"""Small compatibility and optimization patches for pinned DEIM."""

from __future__ import annotations

from contextlib import nullcontext
from functools import wraps
import math
import sys
import types
from typing import Any

import torch
import torch.nn.functional as F


def _box_area_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    sizes = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)
    return sizes[:, 0] * sizes[:, 1]


def generalized_box_iou_aligned(
    boxes1: torch.Tensor, boxes2: torch.Tensor
) -> torch.Tensor:
    """Compute GIoU for matched pairs without allocating an N-by-N matrix."""
    if boxes1.shape != boxes2.shape:
        raise ValueError(
            f"Aligned GIoU expects equal shapes, got {boxes1.shape} and {boxes2.shape}."
        )
    if boxes1.numel() == 0:
        return boxes1.new_zeros((0,))
    area1, area2 = _box_area_xyxy(boxes1), _box_area_xyxy(boxes2)
    intersection_lt = torch.maximum(boxes1[:, :2], boxes2[:, :2])
    intersection_rb = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
    intersection_size = (intersection_rb - intersection_lt).clamp(min=0)
    intersection = intersection_size[:, 0] * intersection_size[:, 1]
    union = (area1 + area2 - intersection).clamp(min=1e-8)
    iou = intersection / union
    enclosing_lt = torch.minimum(boxes1[:, :2], boxes2[:, :2])
    enclosing_rb = torch.maximum(boxes1[:, 2:], boxes2[:, 2:])
    enclosing_size = (enclosing_rb - enclosing_lt).clamp(min=0)
    enclosing_area = (enclosing_size[:, 0] * enclosing_size[:, 1]).clamp(min=1e-8)
    return iou - (enclosing_area - union) / enclosing_area


def memory_efficient_loss_boxes(
    criterion: Any,
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    indices: list[tuple[torch.Tensor, torch.Tensor]],
    num_boxes: float,
    boxes_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """DEIM box loss with aligned GIoU, equivalent to diag(pairwise GIoU)."""
    if "pred_boxes" not in outputs:
        raise KeyError("outputs must contain pred_boxes")
    match_index = criterion._get_src_permutation_idx(indices)
    predicted = outputs["pred_boxes"][match_index]
    expected = torch.cat(
        [target["boxes"][target_index] for target, (_, target_index) in zip(targets, indices)]
    )
    loss_bbox = F.l1_loss(predicted, expected, reduction="none")
    box_ops = criterion._ecdeim_box_ops
    loss_giou = 1 - generalized_box_iou_aligned(
        box_ops.box_cxcywh_to_xyxy(predicted),
        box_ops.box_cxcywh_to_xyxy(expected),
    )
    if boxes_weight is not None:
        loss_giou = loss_giou * boxes_weight
    return {
        "loss_bbox": loss_bbox.sum() / num_boxes,
        "loss_giou": loss_giou.sum() / num_boxes,
    }


def install_memory_efficient_box_loss() -> None:
    """Patch the pinned DEIM criterion with an algebraically equivalent O(N) GIoU."""
    from engine.deim import box_ops
    from engine.deim.deim_criterion import DEIMCriterion

    if not hasattr(DEIMCriterion, "_ecdeim_original_loss_boxes"):
        DEIMCriterion._ecdeim_original_loss_boxes = DEIMCriterion.loss_boxes
        DEIMCriterion.loss_boxes = memory_efficient_loss_boxes
    DEIMCriterion._ecdeim_box_ops = box_ops


def install_optional_calflops_fallback() -> bool:
    """Let training and inference run when the optional FLOPs reporter is absent."""
    try:
        import calflops  # noqa: F401
    except ImportError:
        fallback = types.ModuleType("calflops")
        fallback.calculate_flops = lambda *args, **kwargs: (
            "unavailable",
            "unavailable",
            None,
        )
        sys.modules["calflops"] = fallback
        return True
    return False


def install_torchvision_compatibility() -> None:
    """Bridge the torchvision v2 method rename used by the pinned DEIM revision."""
    from engine.data.transforms._transforms import ConvertBoxes, ConvertPILImage, PadToSize

    for transform_class in (ConvertBoxes, ConvertPILImage, PadToSize):
        if "transform" not in transform_class.__dict__ and "_transform" in transform_class.__dict__:
            transform_class.transform = transform_class._transform


def accumulated_train_one_epoch(
    self_lr_scheduler: bool,
    lr_scheduler: Any,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0.0,
    **kwargs: Any,
) -> dict[str, float]:
    """Accumulate exact group means and step optimizer, EMA, and LR together."""
    from engine.misc import MetricLogger, SmoothedValue, dist_utils

    steps = int(getattr(accumulated_train_one_epoch, "steps", 1))
    if steps < 1:
        raise ValueError("Gradient accumulation steps must be positive.")
    model.train()
    criterion.train()
    metrics = MetricLogger(delimiter="  ")
    metrics.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    print_freq = int(kwargs.get("print_freq", 10))
    writer = kwargs.get("writer")
    ema = kwargs.get("ema")
    scaler = kwargs.get("scaler")
    warmup = kwargs.get("lr_warmup_scheduler")
    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    optimizer_steps_per_epoch = math.ceil(len(data_loader) / steps)

    for micro_step, (samples, targets) in enumerate(
        metrics.log_every(data_loader, print_freq, f"Epoch: [{epoch}]")
    ):
        samples = samples.to(device)
        targets = [
            {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in target.items()
            }
            for target in targets
        ]
        group_start = micro_step // steps * steps
        group_size = min(steps, len(data_loader) - group_start)
        should_step = micro_step + 1 == len(data_loader) or (micro_step + 1) % steps == 0
        global_step = epoch * optimizer_steps_per_epoch + optimizer_step
        metadata = {
            "epoch": epoch,
            "step": micro_step,
            "global_step": global_step,
            "epoch_step": optimizer_steps_per_epoch,
        }
        sync_context = (
            model.no_sync()
            if not should_step and hasattr(model, "no_sync")
            else nullcontext()
        )
        with sync_context:
            if scaler is not None:
                with torch.autocast(device_type=device.type, cache_enabled=True):
                    outputs = model(samples, targets=targets)
                with torch.autocast(device_type=device.type, enabled=False):
                    loss_dict = criterion(outputs, targets, **metadata)
                loss = sum(loss_dict.values())
                scaler.scale(loss / group_size).backward()
            else:
                outputs = model(samples, targets=targets)
                loss_dict = criterion(outputs, targets, **metadata)
                loss = sum(loss_dict.values())
                (loss / group_size).backward()

        if should_step:
            if max_norm > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
            if self_lr_scheduler:
                optimizer = lr_scheduler.step(global_step, optimizer)
            elif warmup is not None:
                warmup.step()
            optimizer_step += 1

        reduced = {key: value.detach() for key, value in dist_utils.reduce_dict(loss_dict).items()}
        reduced_loss = sum(reduced.values())
        loss_value = float(reduced_loss.item())
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}: {loss_value}")
        metrics.update(loss=loss_value, **reduced)
        metrics.update(lr=optimizer.param_groups[0]["lr"])
        if writer is not None and dist_utils.is_main_process() and should_step:
            writer.add_scalar("Loss/total", loss_value, global_step)

    metrics.synchronize_between_processes()
    print("Averaged stats:", metrics)
    return {key: meter.global_avg for key, meter in metrics.meters.items()}


def install_gradient_accumulation(steps: int) -> None:
    """Patch both imported train-loop references and correct scheduler step counts."""
    if int(steps) < 1:
        raise ValueError("Gradient accumulation steps must be positive.")
    if int(steps) == 1:
        return
    from engine.solver import det_engine, det_solver

    accumulated_train_one_epoch.steps = int(steps)
    det_engine.train_one_epoch = accumulated_train_one_epoch
    det_solver.train_one_epoch = accumulated_train_one_epoch
    original_scheduler = det_solver.FlatCosineLRScheduler

    @wraps(original_scheduler)
    def accumulation_aware_scheduler(
        optimizer: torch.optim.Optimizer,
        lr_gamma: float,
        iter_per_epoch: int,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        updates_per_epoch = math.ceil(iter_per_epoch / int(steps))
        return original_scheduler(optimizer, lr_gamma, updates_per_epoch, *args, **kwargs)

    det_solver.FlatCosineLRScheduler = accumulation_aware_scheduler
