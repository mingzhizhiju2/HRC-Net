#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

EXP_NAME="hrcnet_shapenetpart"
CONFIG="configs/shapenetpart.yaml"
OUTPUT_DIR="checkpoints/${EXP_NAME}"
LOG_DIR="logs/${EXP_NAME}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

python -c "
import yaml, torch, os, sys
from torch.utils.data import DataLoader
from models.hrcnet import build_model
from utils.data import ShapeNetPartSeg
from utils.train import train_one_epoch, validate_seg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('${CONFIG}') as f:
    config = yaml.safe_load(f)

train_set = ShapeNetPartSeg(config['data_root'], 'trainval', config['num_points'])
val_set   = ShapeNetPartSeg(config['data_root'], 'test',     config['num_points'])
train_loader = DataLoader(train_set, config['batch_size'], shuffle=True,  num_workers=config.get('num_workers', 4), pin_memory=True)
val_loader   = DataLoader(val_set,   config['batch_size'], shuffle=False, num_workers=config.get('num_workers', 4), pin_memory=True)

model = build_model(config).to(device)
criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
optimizer = torch.optim.Adam(model.parameters(), lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['scheduler']['min_lr'])

best_miou = 0
for epoch in range(config['epochs']):
    loss, acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, miou = validate_seg(model, val_loader, criterion, device, config['num_classes'])
    scheduler.step()
    print(f'Epoch {epoch:03d} | Train Loss: {loss:.4f} Acc: {acc:.4f} | Val Loss: {val_loss:.4f} mIoU: {miou:.4f}')

    if miou > best_miou:
        best_miou = miou
        torch.save(model.state_dict(), '${OUTPUT_DIR}/best.pth')
        print(f'  -> new best mIoU: {miou:.4f}')

print(f'Best mIoU: {best_miou:.4f}')
" 2>&1 | tee "${LOG_DIR}/train.log"