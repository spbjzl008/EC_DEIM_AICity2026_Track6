"""Local evidence routing for heterogeneous pretraining annotations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from .taxonomy import (
    BICYCLE_ID,
    FINE_TRUCK_IDS,
    GENERIC_TRUCK_ID,
    OTHER_ID,
    POSITIVE_CLASS_WEIGHTS,
    PRETRAIN_CLASSES,
    TRAILER_ID,
)


@dataclass(frozen=True)
class EvidenceConfig:
    generic_positive_weight: float = 0.35
    generic_iou_threshold: float = 0.30
    fine_generic_margin: float = 0.50
    margin_weight: float = 0.20
    bicycle_iou_threshold: float = 0.30
    bicycle_coverage_threshold: float = 0.50
    positive_class_weights: dict[int, float] = field(
        default_factory=lambda: dict(POSITIVE_CLASS_WEIGHTS)
    )

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "EvidenceConfig":
        values = dict(values or {})
        raw_weights = values.pop("positive_class_weights", None)
        if raw_weights is not None:
            name_to_id = {name: index for index, name in enumerate(PRETRAIN_CLASSES)}
            weights = {}
            for key, value in raw_weights.items():
                class_id = (
                    name_to_id[key]
                    if isinstance(key, str) and key in name_to_id
                    else int(key)
                )
                weights[class_id] = float(value)
            values["positive_class_weights"] = weights
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        probabilities = (
            self.generic_positive_weight,
            self.generic_iou_threshold,
            self.bicycle_iou_threshold,
            self.bicycle_coverage_threshold,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("Evidence weights and overlap thresholds must be in [0, 1].")
        if self.fine_generic_margin < 0 or self.margin_weight < 0:
            raise ValueError("Margin values must be non-negative.")
        invalid_ids = (
            class_id < 0 or class_id >= len(PRETRAIN_CLASSES)
            for class_id in self.positive_class_weights
        )
        if any(invalid_ids):
            raise ValueError("Positive class weights contain an invalid class id.")
        if any(value <= 0 for value in self.positive_class_weights.values()):
            raise ValueError("Positive class weights must be positive.")


def _box_area(boxes: torch.Tensor) -> torch.Tensor:
    sizes = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)
    return sizes[:, 0] * sizes[:, 1]


def _aligned_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.shape != boxes2.shape:
        raise ValueError(
            f"Aligned IoU expects equal shapes, got {boxes1.shape} and {boxes2.shape}."
        )
    if boxes1.numel() == 0:
        return boxes1.new_zeros((0,))
    intersection_lt = torch.maximum(boxes1[:, :2], boxes2[:, :2])
    intersection_rb = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
    intersection_size = (intersection_rb - intersection_lt).clamp(min=0)
    intersection = intersection_size[:, 0] * intersection_size[:, 1]
    union = (_box_area(boxes1) + _box_area(boxes2) - intersection).clamp(min=1e-8)
    return intersection / union


def _max_iou(box_ops: Any, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0],))
    maximum = boxes1.new_zeros((boxes1.shape[0],))
    for start in range(0, len(boxes2), 512):
        iou = box_ops.box_iou(boxes1, boxes2[start : start + 512])[0]
        maximum = torch.maximum(maximum, iou.max(dim=1).values)
    return maximum


def _ignore_query_mask(
    criterion: Any,
    predicted_boxes: torch.Tensor,
    ignore_boxes: torch.Tensor | None,
) -> torch.Tensor:
    if ignore_boxes is None or ignore_boxes.numel() == 0:
        return torch.zeros(len(predicted_boxes), dtype=torch.bool, device=predicted_boxes.device)
    box_ops = criterion._ecdeim_box_ops
    predicted_xyxy = box_ops.box_cxcywh_to_xyxy(predicted_boxes.detach())
    ignore_xyxy = box_ops.box_cxcywh_to_xyxy(ignore_boxes)
    iou, union = box_ops.box_iou(predicted_xyxy, ignore_xyxy)
    predicted_area = box_ops.box_area(predicted_xyxy).clamp(min=1e-8)[:, None]
    coverage = (iou * union).clamp(min=0) / predicted_area
    config: EvidenceConfig = criterion._ecdeim_evidence_config
    return (
        (iou >= config.bicycle_iou_threshold)
        | (coverage >= config.bicycle_coverage_threshold)
    ).any(dim=1)


def evidence_conditioned_mal(
    criterion: Any,
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    indices: list[tuple[torch.Tensor, torch.Tensor]],
    num_boxes: float,
    values: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Apply local negative masks and positive-only weights to DEIM MAL."""
    logits = outputs["pred_logits"]
    if logits.shape[-1] != len(PRETRAIN_CLASSES):
        return criterion._ecdeim_original_mal(outputs, targets, indices, num_boxes, values=values)

    match_index = criterion._get_src_permutation_idx(indices)
    matched_labels = torch.cat(
        [target["labels"][target_index] for target, (_, target_index) in zip(targets, indices)]
    )
    if values is None:
        predicted = outputs["pred_boxes"][match_index]
        expected = torch.cat(
            [target["boxes"][target_index] for target, (_, target_index) in zip(targets, indices)]
        )
        iou = _aligned_iou(
            criterion._ecdeim_box_ops.box_cxcywh_to_xyxy(predicted),
            criterion._ecdeim_box_ops.box_cxcywh_to_xyxy(expected),
        ).detach()
    else:
        iou = values

    target_classes = torch.full(
        logits.shape[:2], criterion.num_classes, dtype=torch.int64, device=logits.device
    )
    target_classes[match_index] = matched_labels
    one_hot = F.one_hot(target_classes, num_classes=criterion.num_classes + 1)[..., :-1]
    matched_iou = torch.zeros_like(target_classes, dtype=logits.dtype)
    matched_iou[match_index] = iou.to(logits.dtype)
    target_score = (matched_iou.unsqueeze(-1) * one_hot).pow(criterion.gamma)
    negative_weight = logits.sigmoid().detach().pow(criterion.gamma)
    weight = negative_weight * (1 - one_hot) + one_hot
    if criterion.mal_alpha is not None:
        weight = criterion.mal_alpha * negative_weight * (1 - one_hot) + one_hot
    element_loss = F.binary_cross_entropy_with_logits(
        logits, target_score, weight=weight, reduction="none"
    )
    unmasked_loss = element_loss.clone()

    box_ops = criterion._ecdeim_box_ops
    config: EvidenceConfig = criterion._ecdeim_evidence_config
    fine_columns = sorted(FINE_TRUCK_IDS)
    generic_weak_columns = [column for column in fine_columns if column != TRAILER_ID]
    margin_terms: list[torch.Tensor] = []

    for batch_index, target in enumerate(targets):
        bicycle_mask = _ignore_query_mask(
            criterion,
            outputs["pred_boxes"][batch_index],
            target.get("_ecdeim_ignore_boxes"),
        )
        element_loss[batch_index, bicycle_mask, BICYCLE_ID] = 0.0

        labels = target.get("labels")
        boxes = target.get("boxes")
        if labels is None or boxes is None or labels.numel() == 0:
            continue
        generic = labels == GENERIC_TRUCK_ID
        if not bool(generic.any()):
            continue
        predicted_xyxy = box_ops.box_cxcywh_to_xyxy(outputs["pred_boxes"][batch_index].detach())
        generic_xyxy = box_ops.box_cxcywh_to_xyxy(boxes[generic])
        local_queries = (
            _max_iou(box_ops, predicted_xyxy, generic_xyxy) >= config.generic_iou_threshold
        )
        for column in generic_weak_columns:
            element_loss[batch_index, local_queries, column] = 0.0

    for position, (batch_index, query_index) in enumerate(
        zip(match_index[0].tolist(), match_index[1].tolist())
    ):
        target_id = int(matched_labels[position])
        if target_id == GENERIC_TRUCK_ID:
            element_loss[batch_index, query_index, generic_weak_columns] = 0.0
            element_loss[
                batch_index, query_index, GENERIC_TRUCK_ID
            ] *= config.generic_positive_weight
        elif target_id in FINE_TRUCK_IDS:
            element_loss[batch_index, query_index, fine_columns] = unmasked_loss[
                batch_index, query_index, fine_columns
            ]
            margin_terms.append(
                F.relu(
                    logits[batch_index, query_index, GENERIC_TRUCK_ID]
                    - logits[batch_index, query_index, target_id]
                    + config.fine_generic_margin
                )
            )
        if target_id == BICYCLE_ID:
            element_loss[batch_index, query_index, BICYCLE_ID] = unmasked_loss[
                batch_index, query_index, BICYCLE_ID
            ]
        element_loss[batch_index, query_index, target_id] *= config.positive_class_weights.get(
            target_id, 1.0
        )

    loss = element_loss.sum() / num_boxes
    if margin_terms:
        loss = loss + config.margin_weight * torch.stack(margin_terms).mean()
    return {"loss_mal": loss}


def split_ignore_regions(target: dict[str, Any]) -> dict[str, Any]:
    """Move protocol-specific ignore boxes out of positive targets while retaining geometry."""
    labels = target["labels"]
    iscrowd = target.get("iscrowd")
    if iscrowd is None or len(iscrowd) != len(labels):
        ignore_mask = torch.zeros_like(labels, dtype=torch.bool)
    else:
        ignore_mask = iscrowd.to(dtype=torch.bool)
    per_instance = {"boxes", "labels", "area", "iscrowd", "masks", "keypoints"}
    clean: dict[str, Any] = {}
    for key, value in target.items():
        if (
            key in per_instance
            and isinstance(value, torch.Tensor)
            and value.ndim > 0
            and value.shape[0] == labels.shape[0]
        ):
            clean[key] = value[~ignore_mask]
        else:
            clean[key] = value
    ignored = target["boxes"][ignore_mask]
    previous = target.get("_ecdeim_ignore_boxes")
    if isinstance(previous, torch.Tensor) and previous.numel():
        ignored = torch.cat([previous, ignored])
    clean["_ecdeim_ignore_boxes"] = ignored
    return clean


def _convert_coco_with_ignore(
    self: Any,
    image: Image.Image,
    target: dict[str, Any],
    **kwargs: Any,
):
    """Preserve protocol-specific ignore regions through DEIM's COCO conversion."""
    width, height = image.size
    annotations = target["annotations"]
    boxes = torch.as_tensor(
        [annotation["bbox"] for annotation in annotations], dtype=torch.float32
    ).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2].clamp_(0, width)
    boxes[:, 1::2].clamp_(0, height)

    category_to_label = kwargs.get("category2label")
    labels = []
    for annotation in annotations:
        label = annotation["category_id"]
        if category_to_label is not None:
            label = category_to_label[label]
        labels.append(OTHER_ID if annotation.get("iscrowd", 0) else label)
    labels_tensor = torch.tensor(labels, dtype=torch.int64)
    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
    converted = {
        "boxes": boxes[keep],
        "labels": labels_tensor[keep],
        "image_id": torch.tensor([target["image_id"]]),
        "area": torch.tensor(
            [
                annotation.get("area", annotation["bbox"][2] * annotation["bbox"][3])
                for annotation in annotations
            ],
            dtype=torch.float32,
        )[keep],
        "iscrowd": torch.tensor(
            [annotation.get("iscrowd", 0) for annotation in annotations], dtype=torch.int64
        )[keep],
        "orig_size": torch.as_tensor([int(width), int(height)]),
    }
    if self.return_masks:
        from engine.data.dataset.coco_dataset import convert_coco_poly_to_mask

        segmentations = [annotation.get("segmentation", []) for annotation in annotations]
        converted["masks"] = convert_coco_poly_to_mask(segmentations, height, width)[keep]
    return image, converted


def _getitem_with_ignore(self: Any, index: int):
    image, target = self._ecdeim_original_getitem(index)
    return image, split_ignore_regions(target)


def _criterion_with_ignore(self: Any, outputs: Any, targets: list[dict[str, Any]], **kwargs: Any):
    return self._ecdeim_original_forward(
        outputs, [split_ignore_regions(target) for target in targets], **kwargs
    )


def _mixup_with_ignore(self: Any, images: torch.Tensor, targets: list[dict[str, Any]]):
    mixed_images, mixed_targets = self._ecdeim_original_mixup(images, targets)
    if mixed_targets is targets:
        return mixed_images, mixed_targets
    shifted = targets[-1:] + targets[:-1]
    for target, other, mixed in zip(targets, shifted, mixed_targets):
        empty = target["boxes"].new_zeros((0, 4))
        mixed["_ecdeim_ignore_boxes"] = torch.cat(
            [target.get("_ecdeim_ignore_boxes", empty), other.get("_ecdeim_ignore_boxes", empty)]
        )
    return mixed_images, mixed_targets


def install_evidence_routing(config: EvidenceConfig) -> None:
    """Install evidence routing into the pinned DEIM data and loss interfaces."""
    from engine.data.dataloader import BatchImageCollateFunction
    from engine.data.dataset.coco_dataset import CocoDetection, ConvertCocoPolysToMask
    from engine.deim import box_ops
    from engine.deim.deim_criterion import DEIMCriterion

    config.validate()
    if not hasattr(ConvertCocoPolysToMask, "_ecdeim_original_call"):
        ConvertCocoPolysToMask._ecdeim_original_call = ConvertCocoPolysToMask.__call__
        ConvertCocoPolysToMask.__call__ = _convert_coco_with_ignore
    if not hasattr(CocoDetection, "_ecdeim_original_getitem"):
        CocoDetection._ecdeim_original_getitem = CocoDetection.__getitem__
        CocoDetection.__getitem__ = _getitem_with_ignore
    if not hasattr(BatchImageCollateFunction, "_ecdeim_original_mixup"):
        BatchImageCollateFunction._ecdeim_original_mixup = BatchImageCollateFunction.apply_mixup
        BatchImageCollateFunction.apply_mixup = _mixup_with_ignore
    if not hasattr(DEIMCriterion, "_ecdeim_original_mal"):
        DEIMCriterion._ecdeim_original_mal = DEIMCriterion.loss_labels_mal
        DEIMCriterion.loss_labels_mal = evidence_conditioned_mal
    if not hasattr(DEIMCriterion, "_ecdeim_original_forward"):
        DEIMCriterion._ecdeim_original_forward = DEIMCriterion.forward
        DEIMCriterion.forward = _criterion_with_ignore
    DEIMCriterion._ecdeim_box_ops = box_ops
    DEIMCriterion._ecdeim_evidence_config = config
