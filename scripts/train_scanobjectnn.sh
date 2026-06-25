#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

EXP_NAME="hrcnet_scanobjectnn"
CONFIG="configs/scanobjectnn.yaml"
OUTPUT_DIR="checkpoints/${EXP_NAME}"
LOG_DIR="logs/${EXP_NAME}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

python -c "
import yaml, torch, os, sys
from torch.utils.data import DataLoader
from models.hrcnet import build_model
from utils.train import train_one_epoch_cls, validate_cls

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('${CONFIG}') as f:
    config = yaml.safe_load(f)

if not os.path.isdir(config['data_root']):
    print(f'Please prepare ScanObjectNN data at {config[\"data_root\"]} first.')
    sys.exit(1)

# Use same ModelNet40Cls loader assuming ScanObjectNN has compatible h5 format
from utils.data import ModelNet40Cls
train_set = ModelNet40Cls(config['data_root'], 'train', config['num_points'])
val_set   = ModelNet40Cls(config['data_root'], 'test',  config['num_points'])
train_loader = DataLoader(train_set, config['batch_size'], shuffle=True,  num_workers=config.get('num_workers', 4), pin_memory=True)
val_loader   = DataLoader(val_set,   config['batch_size'], shuffle=False, num_workers=config.get('num_workers', 4), pin_memory=True)

model = build_model(config).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['scheduler']['min_lr'])

best_oa = 0
for epoch in range(config['epochs']):
    loss, acc = train_one_epoch_cls(model, train_loader, optimizer, criterion, device)
    val_loss, oa, macc = validate_cls(model, val_loader, criterion, device, config['num_classes'])
    scheduler.step()
    print(f'Epoch {epoch:03d} | Train Loss: {loss:.4f} Acc: {acc:.4f} | Val Loss: {val_loss:.4f} OA: {oa:.4f} mAcc: {macc:.4f}')

    if oa > best_oa:
        best_oa = oa
        torch.save(model.state_dict(), '${OUTPUT_DIR}/best.pth')
        print(f'  -> new best OA: {oa:.4f}')

print(f'Best OA: {best_oa:.4f}')
" 2>&1 | tee "${LOG_DIR}/train.log"