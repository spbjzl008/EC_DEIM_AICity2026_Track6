"""Configuration loading and deterministic DEIM YAML rendering."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .adaptation import optimizer_groups
from .taxonomy import PRETRAIN_CLASSES, TRACK6_CLASSES


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _read_experiment(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"Circular experiment inheritance: {chain}")
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise TypeError("Experiment configuration must be a YAML mapping.")
    parent = values.pop("extends", None)
    if parent is None:
        return values
    parent_path = Path(str(parent))
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return _deep_merge(_read_experiment(parent_path, (*stack, path)), values)


def load_experiment(path: Path) -> dict[str, Any]:
    values = _read_experiment(path)
    stage = values.get("stage")
    if stage not in {"pretrain", "adapt"}:
        raise ValueError("stage must be either 'pretrain' or 'adapt'.")
    for section in ("model", "training", "optimizer", "augmentation"):
        if not isinstance(values.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")
    return values


def _pretrain_transforms(config: dict[str, Any], image_size: int) -> dict[str, Any]:
    augmentation = config["augmentation"]
    profile = augmentation.get("profile")
    if profile != "oats_plus_deim_native":
        raise ValueError(f"Unsupported pretraining augmentation profile: {profile}")
    oats = deepcopy(augmentation.get("oats", {}))
    ops: list[dict[str, Any]] = [
        {
            "type": "ObjectAwareTileSampling",
            **oats,
            "output_size": image_size,
        }
    ]
    policy_ops = ["ObjectAwareTileSampling"]
    ops.extend(
        [
            {
                "type": "Mosaic",
                "output_size": image_size // 2,
                "rotation_range": 10,
                "translation_range": [0.1, 0.1],
                "scaling_range": [0.5, 1.5],
                "probability": 1.0,
                "fill_value": 0,
                "use_cache": False,
                "max_cached_images": 50,
                "random_pop": True,
            },
            {"type": "RandomPhotometricDistort", "p": 0.5},
            {"type": "RandomZoomOut", "fill": 0},
            {"type": "RandomIoUCrop", "p": 0.8},
            {"type": "SanitizeBoundingBoxes", "min_size": 1},
            {"type": "RandomHorizontalFlip"},
            {"type": "Resize", "size": [image_size, image_size]},
            {"type": "SanitizeBoundingBoxes", "min_size": 1},
            {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
            {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
        ]
    )
    policy_ops.extend(["Mosaic", "RandomPhotometricDistort", "RandomZoomOut", "RandomIoUCrop"])
    return {
        "type": "Compose",
        "ops": ops,
        "policy": {
            "name": "stop_epoch",
            "epoch": list(augmentation["policy_epochs"]),
            "ops": policy_ops,
        },
        "mosaic_prob": float(augmentation.get("mosaic_probability", 0.5)),
    }


def _adapt_transforms(config: dict[str, Any], image_size: int) -> dict[str, Any]:
    augmentation = config["augmentation"]
    profile = augmentation.get("profile")
    if profile != "oadc_plus_deim_native_light":
        raise ValueError(f"Unsupported adaptation augmentation profile: {profile}")
    oadc = deepcopy(augmentation.get("oadc", {}))
    ops: list[dict[str, Any]] = [{"type": "ObjectAwareDomainCoverage", **oadc}]
    ops.extend(
        [
            {"type": "RandomPhotometricDistort", "p": 0.5},
            {"type": "RandomZoomOut", "fill": 0},
            {"type": "SanitizeBoundingBoxes", "min_size": 1},
            {"type": "RandomHorizontalFlip"},
            {"type": "Resize", "size": [image_size, image_size]},
            {"type": "SanitizeBoundingBoxes", "min_size": 1},
            {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
            {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
        ]
    )
    policy_epoch = augmentation.get("policy_epochs", [2, 6, 12])
    if isinstance(policy_epoch, (list, tuple)):
        policy_epoch = list(policy_epoch)
    else:
        policy_epoch = int(policy_epoch)
    return {
        "type": "Compose",
        "ops": ops,
        "policy": {
            "name": "stop_epoch",
            "epoch": policy_epoch,
            "ops": ["RandomPhotometricDistort", "RandomZoomOut"],
        },
        # DEIM names the middle-stage selection probability `mosaic_prob`.
        "mosaic_prob": float(augmentation.get("native_stage_probability", 0.5)),
    }


def _validation_transforms(image_size: int) -> dict[str, Any]:
    return {
        "type": "Compose",
        "ops": [
            {"type": "Resize", "size": [image_size, image_size]},
            {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
        ],
    }


def _optimizer(config: dict[str, Any]) -> dict[str, Any]:
    values = config["optimizer"]
    if config["stage"] == "adapt":
        groups, base_lr = optimizer_groups(
            values["class_head_lr"], values["box_lr"], values["decoder_lr"]
        )
    else:
        groups = deepcopy(values.get("parameter_groups", []))
        base_lr = float(values["lr"])
    return {
        "type": "AdamW",
        "params": groups,
        "lr": float(base_lr),
        "betas": [0.9, 0.999],
        "weight_decay": float(values["weight_decay"]),
    }


def render_deim_yaml(
    experiment: dict[str, Any],
    deim_root: Path,
    output_dir: Path,
    train_images: Path,
    train_annotations: Path,
    val_images: Path,
    val_annotations: Path,
    world_size: int,
) -> dict[str, Any]:
    """Build the upstream YAML without modifying the vendored DEIM checkout."""
    if world_size < 1:
        raise ValueError("world_size must be positive.")
    base_model = deim_root / experiment["model"].get(
        "base_config", "configs/deim_dfine/dfine_hgnetv2_x_coco.yml"
    )
    base_runtime = deim_root / "configs/base/deim.yml"
    required_paths = (
        base_model,
        base_runtime,
        train_images,
        train_annotations,
        val_images,
        val_annotations,
    )
    for required in required_paths:
        if not required.exists():
            raise FileNotFoundError(required)

    stage = experiment["stage"]
    training = experiment["training"]
    image_size = int(experiment["model"]["image_size"])
    classes = PRETRAIN_CLASSES if stage == "pretrain" else TRACK6_CLASSES
    declared_classes = int(experiment["model"]["num_classes"])
    if declared_classes != len(classes):
        raise ValueError(f"{stage} requires {len(classes)} model classes, got {declared_classes}.")
    micro_batch = int(training["micro_batch_per_gpu"])
    if micro_batch < 1:
        raise ValueError("micro_batch_per_gpu must be positive.")
    transforms = (
        _pretrain_transforms(experiment, image_size)
        if stage == "pretrain"
        else _adapt_transforms(experiment, image_size)
    )
    no_aug = int(training["no_aug_epochs"])
    stop_epoch = int(training.get("collate_stop_epoch", int(training["epochs"]) - no_aug))
    if stop_epoch <= 0:
        raise ValueError("no_aug_epochs must be smaller than epochs.")

    return {
        "__include__": [str(base_model.resolve()), str(base_runtime.resolve())],
        "output_dir": str(output_dir.resolve()),
        "num_classes": len(classes),
        "remap_mscoco_category": False,
        "eval_spatial_size": [image_size, image_size],
        "sync_bn": bool(experiment["model"].get("sync_bn", world_size > 1)),
        "use_ema": bool(training.get("use_ema", True)),
        "find_unused_parameters": bool(
            experiment["model"].get("find_unused_parameters", stage == "adapt")
        ),
        "checkpoint_freq": int(training.get("checkpoint_frequency", 1)),
        "print_freq": int(training.get("print_frequency", 100)),
        "epoches": int(training["epochs"]),
        "flat_epoch": int(training["flat_epochs"]),
        "no_aug_epoch": no_aug,
        "warmup_iter": int(training.get("warmup_iterations", 500)),
        "scaler": {
            "type": "GradScaler",
            "enabled": True,
        },
        "HGNetv2": {"pretrained": False},
        "optimizer": _optimizer(experiment),
        "lrsheduler": "flatcosine",
        "lr_gamma": float(training.get("lr_gamma", 0.5)),
        "train_dataloader": {
            "dataset": {
                "img_folder": str(train_images.resolve()),
                "ann_file": str(train_annotations.resolve()),
                "transforms": transforms,
            },
            "collate_fn": {
                "base_size": image_size,
                "base_size_repeat": experiment["augmentation"].get("base_size_repeat"),
                "stop_epoch": stop_epoch,
                "ema_restart_decay": float(training.get("ema_restart_decay", 0.9998)),
                "mixup_prob": float(experiment["augmentation"].get("mixup_probability", 0.0)),
                "mixup_epochs": list(experiment["augmentation"].get("mixup_epochs", [0, 0])),
            },
            "total_batch_size": micro_batch * world_size,
            "num_workers": int(training.get("workers_per_gpu", 4)),
        },
        "val_dataloader": {
            "dataset": {
                "img_folder": str(val_images.resolve()),
                "ann_file": str(val_annotations.resolve()),
                "transforms": _validation_transforms(image_size),
            },
            "total_batch_size": micro_batch * world_size,
            "num_workers": int(training.get("workers_per_gpu", 4)),
        },
    }


def dump_yaml(values: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
