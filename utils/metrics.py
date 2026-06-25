import numpy as np


def accuracy(output, target):
    pred = output.argmax(dim=-1)
    correct = (pred == target).float().sum()
    return correct / target.numel()


def mean_class_accuracy(output, target, num_classes):
    pred = output.argmax(dim=-1)
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    for c in range(num_classes):
        mask = target_np == c
        class_total[c] = mask.sum()
        if class_total[c] > 0:
            class_correct[c] = (pred_np[mask] == c).sum()
    return float(class_correct[class_total > 0].mean())


def part_seg_iou(pred, target, num_classes=50, ignore_label=-1):
    pred = pred.argmax(dim=-1).cpu().numpy()
    target = target.cpu().numpy()
    ious = []
    for c in range(num_classes):
        if c == ignore_label:
            continue
        inter = np.logical_and(pred == c, target == c).sum()
        union = np.logical_or(pred == c, target == c).sum()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious))
