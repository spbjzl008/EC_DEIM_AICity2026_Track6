"""Small Hafnia adapters shared by export and cloud training entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .data import validate_coco_categories
from .taxonomy import TRACK6_CLASSES


def load_dataset(dataset_path: Path | None = None) -> Any:
    """Load the mounted cloud dataset or an explicit local HafniaDataset."""
    try:
        from hafnia import utils as hafnia_utils
        from hafnia.dataset.hafnia_dataset import HafniaDataset
    except ImportError as error:
        raise RuntimeError("Run this entrypoint inside a Hafnia environment.") from error

    if hafnia_utils.is_hafnia_cloud_job():
        return HafniaDataset.from_path(hafnia_utils.get_dataset_path_in_hafnia_cloud())
    if dataset_path is None:
        raise ValueError("--dataset-path is required outside Hafnia cloud jobs.")
    return HafniaDataset.from_path(dataset_path)


def prepare_training_splits(
    dataset: Any,
    samples: int | None = None,
    keep_empty_ratio: float = 0.15,
    seed: int = 2026,
) -> Any:
    """Select TRAIN+VAL, retain deterministic background samples, and cap a smoke set."""
    try:
        import polars as pl
        from hafnia.dataset.dataset_names import SplitName
        from hafnia.dataset.primitives import Bbox
    except ImportError as error:
        raise RuntimeError("Hafnia and Polars are required for cloud data preparation.") from error
    if not 0.0 <= keep_empty_ratio <= 1.0:
        raise ValueError("keep_empty_ratio must be in [0, 1].")
    if samples is not None and samples < 2:
        raise ValueError("A capped training export needs at least two samples (train and valid).")
    if not dataset.has_primitive(Bbox):
        raise ValueError("The Hafnia dataset has no bounding-box task.")

    selected = dataset.create_split_dataset(split_name=[SplitName.TRAIN, SplitName.VAL])
    bbox_column = Bbox.column_name()
    positives = selected.samples.filter(pl.col(bbox_column).list.len() > 0)
    empty = selected.samples.filter(pl.col(bbox_column).list.len() == 0)
    frames = [positives]
    if keep_empty_ratio > 0 and len(empty) > 0:
        keep = min(max(round(len(empty) * keep_empty_ratio), 1), len(empty))
        frames.append(empty.sample(n=keep, seed=seed, shuffle=True))
    selected = selected.update_samples(pl.concat(frames, how="diagonal"))

    if samples is not None and len(selected) > samples:
        train = selected.create_split_dataset(split_name=SplitName.TRAIN)
        valid = selected.create_split_dataset(split_name=SplitName.VAL)
        if len(train) == 0 or len(valid) == 0:
            raise RuntimeError("Both train and validation splits must be non-empty.")
        valid_count = min(len(valid), max(1, samples // 5))
        train_count = min(len(train), samples - valid_count)
        if train_count < 1:
            raise RuntimeError("The sample cap leaves no training image.")
        capped = [
            train.select_samples(n_samples=train_count, seed=seed).samples,
            valid.select_samples(n_samples=valid_count, seed=seed).samples,
        ]
        selected = selected.update_samples(pl.concat(capped, how="diagonal"))

    train_count = len(selected.create_split_dataset(split_name=SplitName.TRAIN))
    valid_count = len(selected.create_split_dataset(split_name=SplitName.VAL))
    if train_count == 0 or valid_count == 0:
        raise RuntimeError(
            f"Hafnia training export needs both splits, got train={train_count}, valid={valid_count}."
        )
    return selected


def get_bbox_task_name(dataset: Any, task_name: str | None = None) -> str:
    try:
        from hafnia.dataset.primitives import Bbox
    except ImportError as error:
        raise RuntimeError("Hafnia is required for task discovery.") from error
    task = dataset.info.get_task_by_name(task_name) if task_name else None
    if task is None:
        task = dataset.info.get_task_by_primitive(Bbox)
    if task is None:
        raise ValueError("The Hafnia dataset has no bounding-box task.")
    return str(task.name)


def export_coco(dataset: Any, output: Path, task_name: str) -> dict[str, Path]:
    """Export and validate the zero-based Track 6 COCO train/valid contract."""
    output = output.resolve()
    if output.exists():
        if not output.is_dir():
            raise NotADirectoryError(output)
        if any(output.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    dataset.to_coco_format(output, task_name=task_name)

    annotations = sorted(output.rglob("_annotations.coco.json"))
    by_split: dict[str, Path] = {}
    for annotation in annotations:
        validate_coco_categories(annotation, TRACK6_CLASSES)
        name = annotation.parent.name.lower()
        if name == "train":
            by_split["train"] = annotation
        elif name in {"valid", "validation", "val"}:
            by_split["valid"] = annotation
    if set(by_split) != {"train", "valid"}:
        raise RuntimeError(f"Hafnia COCO export did not create train and valid splits: {annotations}")
    return {
        "train_images": by_split["train"].parent,
        "train_annotations": by_split["train"],
        "val_images": by_split["valid"].parent,
        "val_annotations": by_split["valid"],
    }
