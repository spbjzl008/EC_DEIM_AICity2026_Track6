"""Training-only augmentations used by EC-DEIM."""

from __future__ import annotations

from io import BytesIO
import math
import random
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from torchvision.tv_tensors import BoundingBoxes


def _unpack_sample(inputs: tuple[Any, ...]) -> tuple[Image.Image, dict[str, Any], Any]:
    sample = inputs[0] if len(inputs) == 1 else inputs
    if not isinstance(sample, (tuple, list)) or len(sample) != 3:
        raise TypeError("Expected (image, target, dataset).")
    image, target, dataset = sample
    if not isinstance(image, Image.Image):
        raise TypeError("EC-DEIM augmentations must run before ConvertPILImage.")
    if not isinstance(target, dict):
        raise TypeError("Target must be a dictionary.")
    return image, target, dataset


def _positive_sizes(values: Iterable[int]) -> tuple[int, ...]:
    sizes = tuple(int(value) for value in values)
    if not sizes or any(value <= 0 for value in sizes):
        raise ValueError("Crop-size lists must contain positive integers.")
    return sizes


class ObjectAwareTileSampling(nn.Module):
    """Mix full images with random, rare-object, and low-density crops."""

    _INSTANCE_FIELDS = ("labels", "area", "iscrowd", "masks")

    def __init__(
        self,
        mode_probabilities: dict[str, float] | None = None,
        class_center_weights: dict[int | str, float] | None = None,
        random_crop_sizes: Iterable[int] = (640, 800, 896),
        small_crop_sizes: Iterable[int] = (512, 640),
        large_crop_sizes: Iterable[int] = (800, 896),
        large_class_ids: Iterable[int] = (2, 3, 4, 5),
        small_class_ids: Iterable[int] = (6, 7, 9),
        min_visibility_large: float = 0.60,
        min_visibility_regular: float = 0.30,
        min_visibility_small: float = 0.20,
        center_jitter: float = 0.15,
        max_resample_tries: int = 10,
        max_objects_empty: int = 2,
        min_box_side_after_resize: float = 2.0,
        output_size: int = 896,
    ) -> None:
        super().__init__()
        self.mode_probabilities = mode_probabilities or {
            "whole_image": 0.74,
            "random_tile": 0.14,
            "rare_object_tile": 0.08,
            "empty_tile": 0.04,
        }
        expected = {"whole_image", "random_tile", "rare_object_tile", "empty_tile"}
        if set(self.mode_probabilities) != expected:
            raise ValueError(f"mode_probabilities must contain {sorted(expected)}")
        probability_sum = sum(float(value) for value in self.mode_probabilities.values())
        if not math.isclose(probability_sum, 1.0, abs_tol=1e-6):
            raise ValueError(f"Tile-mode probabilities must sum to one, got {probability_sum}.")
        if any(float(value) < 0 for value in self.mode_probabilities.values()):
            raise ValueError("Tile-mode probabilities must be non-negative.")

        default_weights = {2: 0.55, 4: 0.35, 5: 0.35, 6: 2.0, 7: 2.0, 9: 1.5}
        weights = class_center_weights or default_weights
        self.class_center_weights = {int(key): float(value) for key, value in weights.items()}
        self.random_crop_sizes = _positive_sizes(random_crop_sizes)
        self.small_crop_sizes = _positive_sizes(small_crop_sizes)
        self.large_crop_sizes = _positive_sizes(large_crop_sizes)
        self.large_class_ids = frozenset(map(int, large_class_ids))
        self.small_class_ids = frozenset(map(int, small_class_ids))
        self.min_visibility_large = float(min_visibility_large)
        self.min_visibility_regular = float(min_visibility_regular)
        self.min_visibility_small = float(min_visibility_small)
        self.center_jitter = float(center_jitter)
        self.max_resample_tries = int(max_resample_tries)
        self.max_objects_empty = int(max_objects_empty)
        self.min_box_side_after_resize = float(min_box_side_after_resize)
        self.output_size = int(output_size)

        thresholds = (
            self.min_visibility_large,
            self.min_visibility_regular,
            self.min_visibility_small,
        )
        if any(not 0.0 <= value <= 1.0 for value in thresholds):
            raise ValueError("Visibility thresholds must be in [0, 1].")
        if self.center_jitter < 0 or self.min_box_side_after_resize < 0:
            raise ValueError("Jitter and projected box size must be non-negative.")
        if self.max_resample_tries < 1 or self.output_size < 1:
            raise ValueError("max_resample_tries and output_size must be positive.")
        if self.max_objects_empty < 0:
            raise ValueError("max_objects_empty must be non-negative.")

    @staticmethod
    def _size_multiplier(area_ratio: float) -> float:
        if area_ratio < 0.0005:
            return 1.8
        if area_ratio < 0.0015:
            return 1.5
        if area_ratio < 0.0030:
            return 1.2
        return 1.0

    def _visibility_threshold(self, class_id: int) -> float:
        if class_id in self.large_class_ids:
            return self.min_visibility_large
        if class_id in self.small_class_ids:
            return self.min_visibility_small
        return self.min_visibility_regular

    @staticmethod
    def _valid_sizes(values: tuple[int, ...], width: int, height: int) -> list[int]:
        return [value for value in values if value <= min(width, height)]

    @staticmethod
    def _random_crop(width: int, height: int, size: int) -> tuple[int, int, int, int]:
        left = random.randint(0, width - size) if width > size else 0
        top = random.randint(0, height - size) if height > size else 0
        return left, top, left + size, top + size

    def _rare_crop(
        self,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        iscrowd: torch.Tensor,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int] | None:
        candidates: list[int] = []
        weights: list[float] = []
        image_area = max(float(width * height), 1.0)
        for index, (box, label_value, crowd_value) in enumerate(
            zip(boxes, labels, iscrowd, strict=True)
        ):
            if bool(crowd_value):
                continue
            label = int(label_value)
            class_weight = self.class_center_weights.get(label, 0.0)
            if class_weight <= 0:
                continue
            box_width = max(0.0, float(box[2] - box[0]))
            box_height = max(0.0, float(box[3] - box[1]))
            candidates.append(index)
            weights.append(
                class_weight * self._size_multiplier(box_width * box_height / image_area)
            )
        if not candidates or sum(weights) <= 0:
            return None

        selected = random.choices(candidates, weights=weights, k=1)[0]
        label = int(labels[selected])
        box = boxes[selected]
        area_ratio = (
            max(0.0, float(box[2] - box[0]))
            * max(0.0, float(box[3] - box[1]))
            / image_area
        )
        if label in self.large_class_ids:
            sizes = self._valid_sizes(self.large_crop_sizes, width, height)
        elif label in self.small_class_ids or area_ratio < 0.003:
            sizes = self._valid_sizes(self.small_crop_sizes, width, height)
        else:
            sizes = self._valid_sizes(self.random_crop_sizes, width, height)
        if not sizes:
            return None

        size = random.choice(sizes)
        jitter = self.center_jitter * size
        center_x = 0.5 * float(box[0] + box[2]) + random.uniform(-jitter, jitter)
        center_y = 0.5 * float(box[1] + box[3]) + random.uniform(-jitter, jitter)
        left = max(0, min(round(center_x - size / 2), width - size))
        top = max(0, min(round(center_y - size / 2), height - size))
        return left, top, left + size, top + size

    def _crop_target(
        self,
        image: Image.Image,
        target: dict[str, Any],
        crop: tuple[int, int, int, int],
    ) -> tuple[Image.Image, dict[str, Any], int]:
        if "keypoints" in target:
            raise NotImplementedError("OATS supports box detection targets only.")
        left, top, right, bottom = crop
        crop_width, crop_height = right - left, bottom - top
        boxes = torch.as_tensor(target.get("boxes", torch.empty((0, 4))), dtype=torch.float32)
        labels = torch.as_tensor(target.get("labels", torch.empty((0,), dtype=torch.int64)))
        if boxes.ndim != 2 or boxes.shape[-1] != 4 or len(boxes) != len(labels):
            raise ValueError("Target boxes must have shape [N, 4] and align with labels.")

        old_boxes = target.get("boxes")
        box_format = getattr(old_boxes, "format", "XYXY")
        result = target.copy()
        if len(boxes) == 0:
            result["boxes"] = BoundingBoxes(
                boxes.reshape(0, 4), format=box_format, canvas_size=(crop_height, crop_width)
            )
            result["area"] = torch.empty((0,), dtype=torch.float32)
            result["size"] = torch.tensor([crop_height, crop_width])
            return image.crop(crop), result, 0

        clipped = boxes.clone()
        clipped[:, 0::2].clamp_(min=float(left), max=float(right))
        clipped[:, 1::2].clamp_(min=float(top), max=float(bottom))
        original_area = (
            (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
            * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        )
        visible_area = (
            (clipped[:, 2] - clipped[:, 0]).clamp(min=0)
            * (clipped[:, 3] - clipped[:, 1]).clamp(min=0)
        )
        visibility = visible_area / original_area.clamp(min=1e-12)
        thresholds = torch.tensor(
            [self._visibility_threshold(int(label)) for label in labels], dtype=torch.float32
        )
        projected_width = (clipped[:, 2] - clipped[:, 0]) * self.output_size / max(crop_width, 1)
        projected_height = (clipped[:, 3] - clipped[:, 1]) * self.output_size / max(crop_height, 1)
        keep = (
            (visibility >= thresholds)
            & (projected_width >= self.min_box_side_after_resize)
            & (projected_height >= self.min_box_side_after_resize)
        )

        shifted = clipped[keep] - torch.tensor([left, top, left, top], dtype=torch.float32)
        result["boxes"] = BoundingBoxes(
            shifted, format=box_format, canvas_size=(crop_height, crop_width)
        )
        for field in self._INSTANCE_FIELDS:
            value = target.get(field)
            if value is None or field == "area":
                continue
            if isinstance(value, torch.Tensor) and value.ndim > 0 and len(value) == len(keep):
                cropped_value = value[keep]
                if field == "masks":
                    cropped_value = cropped_value[:, top:bottom, left:right]
                result[field] = cropped_value
        result["area"] = visible_area[keep]
        result["size"] = torch.tensor([crop_height, crop_width])
        return image.crop(crop), result, int(keep.sum())

    def _choose_crop(
        self,
        mode: str,
        image: Image.Image,
        target: dict[str, Any],
    ) -> tuple[int, int, int, int] | None:
        width, height = image.size
        boxes = torch.as_tensor(target.get("boxes", torch.empty((0, 4))), dtype=torch.float32)
        labels = torch.as_tensor(target.get("labels", torch.empty((0,), dtype=torch.int64)))
        iscrowd = torch.as_tensor(
            target.get("iscrowd", torch.zeros(len(labels), dtype=torch.int64)), dtype=torch.int64
        )
        if len(iscrowd) != len(labels):
            iscrowd = torch.zeros(len(labels), dtype=torch.int64)
        if mode == "rare_object_tile":
            return self._rare_crop(boxes, labels, iscrowd, width, height)
        sizes = self._valid_sizes(self.random_crop_sizes, width, height)
        return self._random_crop(width, height, random.choice(sizes)) if sizes else None

    def forward(self, *inputs: Any) -> tuple[Image.Image, dict[str, Any], Any]:
        image, target, dataset = _unpack_sample(inputs)
        modes = list(self.mode_probabilities)
        mode = random.choices(
            modes, weights=[self.mode_probabilities[name] for name in modes], k=1
        )[0]
        if mode == "whole_image":
            return image, target, dataset

        fallback: tuple[Image.Image, dict[str, Any], int] | None = None
        for _ in range(self.max_resample_tries):
            crop = self._choose_crop(mode, image, target)
            if crop is None:
                break
            candidate = self._crop_target(image, target, crop)
            fallback = candidate
            retained = candidate[2]
            if mode in {"rare_object_tile", "random_tile"} and retained == 0:
                continue
            if mode == "empty_tile" and retained > self.max_objects_empty:
                continue
            return candidate[0], candidate[1], dataset
        if mode == "empty_tile" and fallback is not None and fallback[2] <= self.max_objects_empty:
            return fallback[0], fallback[1], dataset
        return image, target, dataset


class ObjectAwareDomainCoverage(nn.Module):
    """Blend a sampled appearance shift through a box-aware protection mask."""

    SCENARIOS = (
        "day_clear",
        "low_light",
        "haze_fog",
        "compression_heavy",
        "motion_blur",
        "shadow_glare",
    )

    def __init__(
        self,
        max_probability: float = 0.90,
        constant_until_epoch: int = 4,
        zero_at_epoch: int = 8,
        background_strength: Iterable[float] = (0.45, 0.85),
        object_strength: Iterable[float] = (0.08, 0.28),
        small_object_strength: Iterable[float] = (0.02, 0.10),
        small_object_area_ratio: float = 0.0021,
        box_padding_ratio: float = 0.12,
        feather_ratio: float = 0.08,
        force_scenario: str | None = None,
    ) -> None:
        super().__init__()
        self.max_probability = float(max_probability)
        self.constant_until_epoch = int(constant_until_epoch)
        self.zero_at_epoch = int(zero_at_epoch)
        self.background_strength = self._range(background_strength, "background_strength")
        self.object_strength = self._range(object_strength, "object_strength")
        self.small_object_strength = self._range(small_object_strength, "small_object_strength")
        self.small_object_area_ratio = float(small_object_area_ratio)
        self.box_padding_ratio = float(box_padding_ratio)
        self.feather_ratio = float(feather_ratio)
        self.force_scenario = force_scenario

        if not 0.0 <= self.max_probability <= 1.0:
            raise ValueError("max_probability must be in [0, 1].")
        if self.constant_until_epoch < 0 or self.zero_at_epoch <= self.constant_until_epoch:
            raise ValueError("OADC epoch boundaries must satisfy 0 <= constant < zero.")
        if min(self.small_object_area_ratio, self.box_padding_ratio, self.feather_ratio) < 0:
            raise ValueError("OADC area, padding, and feather values must be non-negative.")
        if self.object_strength[1] > self.background_strength[0]:
            raise ValueError("object_strength must not exceed the background-strength range.")
        if self.small_object_strength[1] > self.background_strength[0]:
            raise ValueError("small_object_strength must not exceed the background-strength range.")
        if force_scenario is not None and force_scenario not in self.SCENARIOS:
            raise ValueError(f"Unknown OADC scenario: {force_scenario}")

    @staticmethod
    def _range(values: Iterable[float], name: str) -> tuple[float, float]:
        pair = tuple(float(value) for value in values)
        if len(pair) != 2 or not 0.0 <= pair[0] <= pair[1] <= 1.0:
            raise ValueError(f"{name} must be an ordered pair in [0, 1].")
        return pair

    def probability_at(self, epoch: int) -> float:
        epoch = max(0, int(epoch))
        if epoch <= self.constant_until_epoch:
            return self.max_probability
        if epoch >= self.zero_at_epoch:
            return 0.0
        span = self.zero_at_epoch - self.constant_until_epoch
        return self.max_probability * (self.zero_at_epoch - epoch) / span

    @staticmethod
    def _enhance(image: Image.Image, enhancer: type, low: float, high: float) -> Image.Image:
        return enhancer(image).enhance(random.uniform(low, high))

    @staticmethod
    def _gamma(image: Image.Image, low: float, high: float) -> Image.Image:
        gamma = random.uniform(low, high)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return Image.fromarray(np.clip(array**gamma * 255.0, 0, 255).astype(np.uint8))

    @staticmethod
    def _noise(image: Image.Image, low: float, high: float) -> Image.Image:
        array = np.asarray(image, dtype=np.float32)
        noise = np.random.normal(0.0, random.uniform(low, high) * 255.0, array.shape)
        return Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8))

    @staticmethod
    def _motion_blur(image: Image.Image) -> Image.Image:
        length = random.choice((5, 7, 9, 11))
        axis = random.choice((0, 1))
        array = np.asarray(image, dtype=np.float32)
        pad = length // 2
        padding = ((pad, pad), (0, 0), (0, 0)) if axis == 0 else ((0, 0), (pad, pad), (0, 0))
        padded = np.pad(array, padding, mode="edge")
        blurred = np.zeros_like(array)
        for offset in range(length):
            if axis == 0:
                blurred += padded[offset : offset + array.shape[0], :, :]
            else:
                blurred += padded[:, offset : offset + array.shape[1], :]
        return Image.fromarray(np.clip(blurred / length, 0, 255).astype(np.uint8))

    @staticmethod
    def _shadow_glare(image: Image.Image) -> Image.Image:
        array = np.asarray(image, dtype=np.float32) / 255.0
        height, width = array.shape[:2]
        yy, xx = np.mgrid[0:height, 0:width]
        angle = random.uniform(0.0, 2.0 * math.pi)
        projection = xx / max(width, 1) * math.cos(angle) + yy / max(height, 1) * math.sin(angle)
        cutoff = random.uniform(-0.1, 0.8)
        softness = random.uniform(0.04, 0.16)
        shadow = 1.0 / (1.0 + np.exp(-(projection - cutoff) / softness))
        array *= 1.0 - random.uniform(0.18, 0.45) * shadow[..., None]
        center_x, center_y = random.uniform(0, width), random.uniform(0, height)
        radius = random.uniform(0.18, 0.45) * max(width, height)
        glare = np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / max(2 * radius**2, 1.0))
        array += random.uniform(0.12, 0.35) * glare[..., None]
        return Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8))

    def _transform(self, image: Image.Image, scenario: str) -> Image.Image:
        if scenario == "day_clear":
            image = self._enhance(image, ImageEnhance.Brightness, 0.82, 1.25)
            image = self._enhance(image, ImageEnhance.Contrast, 0.80, 1.25)
            image = self._enhance(image, ImageEnhance.Color, 0.80, 1.20)
            return self._enhance(image, ImageEnhance.Sharpness, 1.10, 1.45)
        if scenario == "low_light":
            image = self._enhance(image, ImageEnhance.Brightness, 0.55, 0.90)
            image = self._enhance(image, ImageEnhance.Contrast, 0.75, 1.25)
            return self._noise(self._gamma(image, 0.80, 1.30), 0.03, 0.07)
        if scenario == "haze_fog":
            image = self._enhance(image, ImageEnhance.Contrast, 0.65, 0.90)
            image = self._enhance(image, ImageEnhance.Color, 0.55, 0.80)
            fog = Image.new("RGB", image.size, color=(220, 225, 225))
            image = Image.blend(image, fog, random.uniform(0.10, 0.28))
            return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.6, 2.0)))
        if scenario == "compression_heavy":
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=random.randint(20, 55), optimize=False)
            buffer.seek(0)
            image = Image.open(buffer).convert("RGB")
            width, height = image.size
            scale = random.uniform(0.35, 0.65)
            small = image.resize(
                (max(2, round(width * scale)), max(2, round(height * scale))),
                Image.Resampling.BILINEAR,
            )
            return small.resize((width, height), Image.Resampling.BILINEAR)
        if scenario == "motion_blur":
            return self._motion_blur(image)
        if scenario == "shadow_glare":
            return self._shadow_glare(image)
        raise ValueError(f"Unknown OADC scenario: {scenario}")

    def _alpha_map(self, image_size: tuple[int, int], boxes: torch.Tensor) -> np.ndarray:
        width, height = image_size
        background_alpha = random.uniform(*self.background_strength)
        alpha = np.full((height, width), background_alpha, dtype=np.float32)
        coordinates = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4).cpu().numpy()
        if not np.isfinite(coordinates).all():
            raise ValueError("OADC received non-finite box coordinates.")
        for x1, y1, x2, y2 in coordinates:
            x1 = min(float(width), max(0.0, float(x1)))
            x2 = min(float(width), max(0.0, float(x2)))
            y1 = min(float(height), max(0.0, float(y1)))
            y2 = min(float(height), max(0.0, float(y2)))
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            box_width, box_height = x2 - x1, y2 - y1
            if box_width <= 0 or box_height <= 0:
                continue
            area_ratio = box_width * box_height / max(float(width * height), 1.0)
            strength = (
                self.small_object_strength
                if area_ratio <= self.small_object_area_ratio
                else self.object_strength
            )
            object_alpha = random.uniform(*strength)
            pad_x, pad_y = box_width * self.box_padding_ratio, box_height * self.box_padding_ratio
            mask_image = Image.new("L", (width, height), color=0)
            ImageDraw.Draw(mask_image).rectangle(
                (
                    max(0, math.floor(x1 - pad_x)),
                    max(0, math.floor(y1 - pad_y)),
                    min(width - 1, math.ceil(x2 + pad_x)),
                    min(height - 1, math.ceil(y2 + pad_y)),
                ),
                fill=255,
            )
            sigma = max(1.0, min(box_width, box_height) * self.feather_ratio)
            mask = np.asarray(
                mask_image.filter(ImageFilter.GaussianBlur(radius=sigma)), dtype=np.float32
            ) / 255.0
            alpha = np.minimum(
                alpha,
                background_alpha - (background_alpha - object_alpha) * mask,
            )
        return alpha[..., None]

    def forward(self, *inputs: Any) -> tuple[Image.Image, dict[str, Any], Any]:
        image, target, dataset = _unpack_sample(inputs)
        epoch = int(getattr(dataset, "epoch", 0))
        if random.random() >= self.probability_at(epoch):
            return image, target, dataset
        scenario = self.force_scenario or random.choice(self.SCENARIOS)
        transformed = self._transform(image, scenario)
        boxes = torch.as_tensor(target.get("boxes", torch.empty((0, 4))), dtype=torch.float32)
        alpha = self._alpha_map(image.size, boxes)
        source = np.asarray(image, dtype=np.float32)
        changed = np.asarray(transformed, dtype=np.float32)
        mixed = source * (1.0 - alpha) + changed * alpha
        output = Image.fromarray(np.clip(mixed, 0, 255).astype(np.uint8))
        return output, target, dataset


def register_deim_transforms() -> None:
    """Register the two custom transforms with DEIM's YAML workspace."""
    from engine.core import GLOBAL_CONFIG, register

    for transform in (ObjectAwareTileSampling, ObjectAwareDomainCoverage):
        if transform.__name__ not in GLOBAL_CONFIG:
            register()(transform)
