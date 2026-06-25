import os
import time
import torch
import numpy as np
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, criterion, device, config):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    num_batches = len(loader)

    pbar = tqdm(loader, desc="Train")
    for batch in pbar:
        pts = batch[0].to(device)
        lbl = batch[1].to(device)
        cat = batch[2].to(device) if len(batch) > 2 else None

        optimizer.zero_grad()
        if cat is not None:
            logits = model(pts, cat)
        else:
            logits = model(pts)

        logits = logits.view(-1, logits.shape[-1])
        lbl = lbl.view(-1).long()

        loss = criterion(logits, lbl)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=-1)
        correct = (pred == lbl).float().sum().item()
        total_correct += correct
        total_seen += lbl.numel()

        pbar.set_postfix(loss=loss.item())

    return total_loss / num_batches, total_correct / total_seen


def train_one_epoch_cls(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    pbar = tqdm(loader, desc="Train")
    for pts, lbl in pbar:
        pts, lbl = pts.to(device), lbl.to(device)
        optimizer.zero_grad()
        logits = model(pts)
        loss = criterion(logits, lbl)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=-1)
        total_correct += (pred == lbl).float().sum().item()
        total_seen += lbl.size(0)
        pbar.set_postfix(loss=loss.item())

    return total_loss / len(loader), total_correct / total_seen


def validate_cls(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

    with torch.no_grad():
        for pts, lbl in tqdm(loader, desc="Val"):
            pts, lbl = pts.to(device), lbl.to(device)
            logits = model(pts)
            loss = criterion(logits, lbl)
            total_loss += loss.item()
            pred = logits.argmax(dim=-1)
            total_correct += (pred == lbl).sum().item()
            total_seen += lbl.size(0)

            for c in range(num_classes):
                mask = lbl == c
                class_total[c] += mask.sum().item()
                class_correct[c] += (pred[mask] == c).sum().item()

    oa = total_correct / total_seen
    per_class_acc = class_correct / (class_total + 1e-12)
    macc = per_class_acc[class_total > 0].mean()
    return total_loss / len(loader), oa, macc


def validate_seg(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for pts, lbl, cat in tqdm(loader, desc="Val"):
            pts, lbl, cat = pts.to(device), lbl.to(device), cat.to(device)
            logits = model(pts, cat)
            loss = criterion(logits.reshape(-1, num_classes), lbl.reshape(-1))
            total_loss += loss.item()
            all_preds.append(logits.argmax(dim=-1).cpu())
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
    return total_loss / len(loader), miou
