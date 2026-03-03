from dataclasses import dataclass

from torch.nn import Module


def build_param_groups(model: Module, base_lr: float, weight_decay: float):
    groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim == 1 or name.endswith(".bias") or "norm" in name.lower():
            decay = 0.0
        else:
            decay = weight_decay

        if decay not in groups:
            groups[decay] = {
                "params": [],
                "weight_decay": decay,
            }
        
        groups[decay]["params"].append(param)

    return [
        {
            "params": g["params"],
            "weight_decay": g["weight_decay"],
            "lr": base_lr,
        }
        for g in groups.values()
    ]


def poly_warmup_factor(step: int, total_steps: int, warmup_steps: int, power: float) -> float:
    step = step + 1  # So the first optimizer step has a small but non-zero LR

    if warmup_steps > 0 and step < warmup_steps:
        return step / warmup_steps
    
    progress = (step - warmup_steps) / (total_steps - warmup_steps)

    return (1.0 - progress) ** power


@dataclass
class DataConfig:
    data_root: str
    crop_size: int
    scale_min: float
    scale_max: float
    reduce_zero_label: bool
    ignore_index: int
    batch_size: int
    num_workers: int
    pin_memory: bool