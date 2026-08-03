#!/usr/bin/env python3
"""Export the official Hafnia dataset to the zero-based COCO contract used by DEIM."""

from __future__ import annotations

import argparse
from pathlib import Path

from ecdeim.hafnia import export_coco, get_bbox_task_name, load_dataset, prepare_training_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-name")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--empty-image-keep-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    dataset = prepare_training_splits(
        load_dataset(args.dataset_path),
        samples=args.samples,
        keep_empty_ratio=args.empty_image_keep_ratio,
        seed=args.seed,
    )
    paths = export_coco(dataset, args.output, get_bbox_task_name(dataset, args.task_name))
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
