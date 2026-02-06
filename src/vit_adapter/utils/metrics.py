import torch


def fast_hist(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int):
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]
    if pred.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.int64, device=pred.device)
    hist = torch.bincount(
        num_classes * target + pred,
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    return hist


def compute_miou(hist: torch.Tensor):
    hist = hist.float()
    intersection = torch.diag(hist)
    union = hist.sum(1) + hist.sum(0) - intersection
    iou = intersection / torch.clamp(union, min=1.0)
    valid = union > 0
    miou = iou[valid].mean().item() if valid.any() else 0.0
    return miou, iou
