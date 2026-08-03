#!/usr/bin/env python3
"""Render an EC-DEIM experiment and launch upstream DEIM training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from ecdeim.config import dump_yaml, load_experiment, render_deim_yaml
from ecdeim.data import validate_coco_categories
from ecdeim.taxonomy import PRETRAIN_CLASSES, TRACK6_CLASSES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--deim-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--val-images", type=Path, required=True)
    parser.add_argument("--val-annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", default="0", help="Comma-separated CUDA device ids.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.checkpoint is not None and args.resume is not None:
        raise ValueError("Use either --checkpoint or --resume, not both.")
    if args.resume is None and args.checkpoint is None and not args.dry_run:
        raise ValueError("A semantic initialization checkpoint is required for a new run.")
    for optional_path in (args.checkpoint, args.resume):
        if optional_path is not None and not optional_path.is_file():
            raise FileNotFoundError(optional_path)
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if not devices:
        raise ValueError("At least one CUDA device must be specified.")

    experiment = load_experiment(args.config.resolve())
    expected_names = PRETRAIN_CLASSES if experiment["stage"] == "pretrain" else TRACK6_CLASSES
    validate_coco_categories(args.train_annotations.resolve(), expected_names)
    validate_coco_categories(args.val_annotations.resolve(), expected_names)
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = render_deim_yaml(
        experiment,
        args.deim_root.resolve(),
        output_dir,
        args.train_images,
        args.train_annotations,
        args.val_images,
        args.val_annotations,
        len(devices),
    )
    rendered_path = output_dir / "generated_deim.yml"
    dump_yaml(rendered, rendered_path)
    effective_batch = (
        int(experiment["training"]["micro_batch_per_gpu"])
        * len(devices)
        * int(experiment["training"]["gradient_accumulation_steps"])
    )
    manifest = {
        "stage": experiment["stage"],
        "experiment": str(args.config.resolve()),
        "deim_revision": experiment["model"].get("deim_revision"),
        "full_image_training": bool(
            experiment["training"].get("full_image_training", True)
        ),
        "augmentation_profile": experiment["augmentation"].get("profile"),
        "devices": devices,
        "effective_batch_size": effective_batch,
        "seed": args.seed,
        "command": " ".join(sys.argv),
    }
    (output_dir / "run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Rendered DEIM configuration: {rendered_path}")
    if args.dry_run:
        return

    command = [sys.executable]
    if len(devices) > 1:
        command.extend(
            ["-m", "torch.distributed.run", f"--nproc_per_node={len(devices)}", "-m"]
        )
    else:
        command.append("-m")
    command.extend(
        [
            "ecdeim.runner",
            "--deim-root",
            str(args.deim_root.resolve()),
            "--deim-config",
            str(rendered_path),
            "--experiment",
            str(args.config.resolve()),
            "--seed",
            str(args.seed),
        ]
    )
    if args.checkpoint:
        command.extend(["--checkpoint", str(args.checkpoint.resolve())])
    if args.resume:
        command.extend(["--resume", str(args.resume.resolve())])
    if not args.no_amp:
        command.append("--use-amp")
    if args.test_only:
        command.append("--test-only")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    subprocess.run(command, check=True, env=environment)


if __name__ == "__main__":
    main()
