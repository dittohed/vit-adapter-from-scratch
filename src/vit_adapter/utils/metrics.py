import torch


def compute_confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int):
    """
    Compute a `num_classes x num_classes` confusion matrix using a vectorized histogram-based method. 
    Element [i, j] counts how many pixels of class i were predicted as class j.
    """

    # Don't include ignored class pixels in the metric calculation
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]
    if pred.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.int64, device=pred.device)
    
    # Compute a (num_classes, num_classes) confusion matrix
    # by first encoding unique (target, pred) pairs as unique integers (compatible with row-major order)
    # and then counting the occurrences of each pair
    cm = torch.bincount(
        num_classes * target + pred,
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    
    return cm


def compute_miou(cm: torch.Tensor):
    """
    Compute mean Intersection over Union (mIoU) and per-class IoU from a confusion matrix.
    """

    cm = cm.float()

    intersection = torch.diag(cm)
    union = cm.sum(dim=1) + cm.sum(dim=0) - intersection
    per_class_iou = intersection / torch.clamp(union, min=1.0)

    valid = union > 0  # To exclude classes neither present in GT nor predicted
    miou = per_class_iou[valid].mean().item() if valid.any() else 0.0

    return miou, per_class_iou
