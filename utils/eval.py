import torch
from tqdm import tqdm
import numpy as np


def evaluate_cls(model, loader, device, num_classes):
    model.eval()
    total_correct = 0
    total_seen = 0
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

    with torch.no_grad():
        for pts, lbl in tqdm(loader, desc="Eval"):
            pts, lbl = pts.to(device), lbl.to(device)
            logits = model(pts)
            pred = logits.argmax(dim=-1)
            total_correct += (pred == lbl).sum().item()
            total_seen += lbl.size(0)
            for c in range(num_classes):
                mask = lbl == c
                class_total[c] += mask.sum().item()
                class_correct[c] += (pred[mask] == c).sum().item()

    oa = total_correct / total_seen
    per_class = class_correct / (class_total + 1e-12)
    macc = per_class[class_total > 0].mean()
    return {"OA": oa, "mAcc": macc}


def evaluate_partseg(model, loader, device, num_classes, num_categories):
    model.eval()
    all_preds = []
    all_targets = []
    cat_ious = {i: [] for i in range(num_categories)}

    with torch.no_grad():
        for pts, lbl, cat in tqdm(loader, desc="Eval"):
            pts, lbl, cat = pts.to(device), lbl.to(device), cat.to(device)
            logits = model(pts, cat)
            pred = logits.argmax(dim=-1)
            all_preds.append(pred.cpu())
            all_targets.append(lbl.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    ious = []
    for c in range(num_classes):
        inter = np.logical_and(all_preds == c, all_targets == c).sum()
        union = np.logical_or(all_preds == c, all_targets == c).sum()
        if union > 0:
            ious.append(inter / union)

    miou = float(np.mean(ious))
    return {"mIoU": miou}
