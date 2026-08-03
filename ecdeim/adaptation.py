"""Target-domain adaptation hooks for DEIM-D-FINE-X."""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch
import torch.nn as nn

from .taxonomy import CLASS_TO_ID, TRACK6_CLASSES


SINGLE_TRUCK_ID = CLASS_TO_ID["Vehicle.Single Truck"]
HEAVY_DUTY_ID = CLASS_TO_ID["Vehicle.Heavy Duty Vehicle"]
TRAILER_ID = CLASS_TO_ID["Vehicle.Trailer"]


class LoRALinear(nn.Module):
    """Low-rank residual wrapper for a frozen linear projection."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        self.base = base
        for parameter in base.parameters():
            parameter.requires_grad = False
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_A = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.lora_A.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_B.to(device=base.weight.device, dtype=base.weight.dtype)

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.lora_B(self.lora_A(inputs)) * self.scaling
        return self.base(inputs) + residual


def _unwrap(model: nn.Module) -> nn.Module:
    return getattr(model, "module", model)


def inject_decoder_lora(model: nn.Module, rank: int = 8, alpha: float = 16.0) -> list[str]:
    """Add LoRA to decoder deformable cross-attention geometry projections."""
    root = _unwrap(model)
    matched: list[str] = []
    for module_name, module in root.named_modules():
        if not module_name.startswith("decoder."):
            continue
        for attribute in ("attention_weights", "sampling_offsets"):
            projection = getattr(module, attribute, None)
            if isinstance(projection, nn.Linear):
                setattr(module, attribute, LoRALinear(projection, rank=rank, alpha=alpha))
                matched.append(f"{module_name}.{attribute}")
    if not matched and not any(isinstance(module, LoRALinear) for module in root.modules()):
        raise RuntimeError("No decoder cross-attention projections were found for LoRA.")
    return matched


def freeze_backbone(model: nn.Module) -> int:
    root = _unwrap(model)
    backbone = getattr(root, "backbone", None)
    if backbone is None:
        raise RuntimeError("DEIM model has no backbone attribute.")
    count = 0
    for parameter in backbone.parameters():
        parameter.requires_grad = False
        count += parameter.numel()
    return count


def _classification_heads(model: nn.Module) -> list[nn.Linear]:
    decoder = getattr(_unwrap(model), "decoder", None)
    if decoder is None:
        raise RuntimeError("DEIM model has no decoder attribute.")
    heads: list[nn.Linear] = []
    encoder_head = getattr(decoder, "enc_score_head", None)
    if isinstance(encoder_head, nn.Linear):
        heads.append(encoder_head)
    for head in getattr(decoder, "dec_score_head", []):
        if isinstance(head, nn.Linear):
            heads.append(head)
    if not heads:
        raise RuntimeError("No DEIM classification heads were found.")
    return heads


def calibrate_class_heads(
    model: nn.Module,
    single_truck_bias: float = -2.0,
    restart_class_ids: Iterable[int] = (HEAVY_DUTY_ID, TRAILER_ID),
) -> int:
    """Lower the Single Truck prior and restart weak target-domain rows."""
    restart_ids = tuple(int(class_id) for class_id in restart_class_ids)
    heads = _classification_heads(model)
    with torch.no_grad():
        for head in heads:
            if head.out_features != len(TRACK6_CLASSES):
                raise ValueError(
                    f"Expected {len(TRACK6_CLASSES)} output rows, got {head.out_features}."
                )
            if head.bias is None:
                raise ValueError("EC-DEIM head calibration requires classification biases.")
            head.bias[SINGLE_TRUCK_ID] += float(single_truck_bias)
            for class_id in restart_ids:
                nn.init.normal_(head.weight[class_id], mean=0.0, std=0.01)
                head.bias[class_id] = 0.0
    return len(heads)


def optimizer_groups(
    class_head_lr: float,
    box_lr: float,
    decoder_lr: float,
) -> tuple[list[dict[str, Any]], float]:
    """Return mutually exclusive regex groups for DEIM's optimizer builder."""
    groups = [
        {
            "params": r"^decoder\..*(?:enc_score_head|dec_score_head|denoising_class_embed).*$",
            "lr": float(class_head_lr),
        },
        {
            "params": r"^decoder\..*(?:enc_bbox_head|dec_bbox_head).*$",
            "lr": float(box_lr),
        },
        {
            "params": r"^decoder\.(?!.*(?:score_head|bbox_head|denoising_class_embed)).*$",
            "lr": float(decoder_lr),
        },
    ]
    return groups, float(box_lr)


def install_adaptation_hooks(config: dict[str, Any], resume: bool = False) -> None:
    """Install LoRA, backbone freezing, and class-head calibration at setup time."""
    from engine.solver._solver import BaseSolver

    use_lora = bool(config.get("lora", {}).get("enabled", True))
    rank = int(config.get("lora", {}).get("rank", 8))
    alpha = float(config.get("lora", {}).get("alpha", 16))
    freeze = bool(config.get("freeze_backbone", True))
    calibration = config.get("head_calibration", {})
    calibrate = bool(calibration.get("enabled", True)) and not resume
    bias = float(calibration.get("single_truck_bias", -2.0))
    restart_names = calibration.get(
        "restart_classes", ["Vehicle.Heavy Duty Vehicle", "Vehicle.Trailer"]
    )
    restart_ids = tuple(CLASS_TO_ID[name] for name in restart_names)

    original_load_tuning = BaseSolver.load_tuning_state

    def load_tuning_with_lora(self: Any, path: str) -> None:
        original_load_tuning(self, path)
        if use_lora:
            inject_decoder_lora(self.model, rank=rank, alpha=alpha)

    BaseSolver.load_tuning_state = load_tuning_with_lora
    original_setup = BaseSolver._setup

    def setup_with_adaptation(self: Any) -> None:
        original_setup(self)
        ema = getattr(getattr(self, "ema", None), "module", None)
        has_lora = any(
            isinstance(module, LoRALinear) for module in _unwrap(self.model).modules()
        )
        if use_lora and not has_lora:
            inject_decoder_lora(self.model, rank=rank, alpha=alpha)
            if ema is not None:
                inject_decoder_lora(ema, rank=rank, alpha=alpha)
        if freeze:
            freeze_backbone(self.model)
        if calibrate:
            calibrate_class_heads(self.model, bias, restart_ids)
            if ema is not None:
                calibrate_class_heads(ema, bias, restart_ids)

    BaseSolver._setup = setup_with_adaptation


def infer_lora_spec(state_dict: dict[str, torch.Tensor]) -> tuple[int, float] | None:
    """Recover the released LoRA rank and alpha from checkpoint metadata or tensor shape."""
    keys = [key for key in state_dict if key.endswith("lora_A.weight")]
    if not keys:
        return None
    return int(state_dict[keys[0]].shape[0]), 16.0
