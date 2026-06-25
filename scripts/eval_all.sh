#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

echo "========== Evaluating ModelNet40 =========="
python -c "
import yaml, torch, os
from torch.utils.data import DataLoader
from models.hrcnet import build_model
from utils.eval import evaluate_cls
from utils.data import ModelNet40Cls

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('configs/modelnet40.yaml') as f:
    config = yaml.safe_load(f)

model = build_model(config).to(device)
ckpt = 'checkpoints/hrcnet_modelnet40/best.pth'
if os.path.exists(ckpt):
    model.load_state_dict(torch.load(ckpt, map_location=device))
else:
    print(f'Checkpoint not found: {ckpt}, using random init')

val_set = ModelNet40Cls(config['data_root'], 'test', config['num_points'])
val_loader = DataLoader(val_set, config['batch_size'], shuffle=False, num_workers=4)
results = evaluate_cls(model, val_loader, device, config['num_classes'])
print(f'ModelNet40 | OA: {results[\"OA\"]:.4f} | mAcc: {results[\"mAcc\"]:.4f}')
" 2>&1

echo ""
echo "========== Evaluating ShapeNet Part =========="
python -c "
import yaml, torch, os
from torch.utils.data import DataLoader
from models.hrcnet import build_model
from utils.eval import evaluate_partseg
from utils.data import ShapeNetPartSeg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('configs/shapenetpart.yaml') as f:
    config = yaml.safe_load(f)

model = build_model(config).to(device)
ckpt = 'checkpoints/hrcnet_shapenetpart/best.pth'
if os.path.exists(ckpt):
    model.load_state_dict(torch.load(ckpt, map_location=device))
else:
    print(f'Checkpoint not found: {ckpt}, using random init')

val_set = ShapeNetPartSeg(config['data_root'], 'test', config['num_points'])
val_loader = DataLoader(val_set, config['batch_size'], shuffle=False, num_workers=4)
results = evaluate_partseg(model, val_loader, device, config['num_classes'], config['num_categories'])
print(f'ShapeNet Part | mIoU: {results[\"mIoU\"]:.4f}')
" 2>&1