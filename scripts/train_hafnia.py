#!/usr/bin/env python3
"""Export Track 6 from Hafnia and launch the shared EC-DEIM training path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from ecdeim.hafnia import export_coco, get_bbox_task_name, load_dataset, prepare_training_splits
from ecdeim.taxonomy import TRACK6_CLASSES


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "adapt.yaml")
    parser.add_argument("--deim-root", type=Path, default=ROOT / "third_party" / "DEIM")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ROOT / "pretrained_models" / "DEIM_track6_896",
        help="Directory containing model_config.json and the 10-class bridge checkpoint.",
    )
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--task-name")
    parser.add_argument("--samples", type=int, help="Deterministic TRAIN+VAL cap for smoke tests.")
    parser.add_argument("--empty-image-keep-ratio", type=float, default=0.15)
    parser.add_argument("--data-output", type=Path, default=Path(".data/ecdeim_track6_coco"))
    parser.add_argument("--output", type=Path, default=Path("outputs/adapt"))
    parser.add_argument("--devices", default="0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def resolve_checkpoint(checkpoint: Path | None, model_path: Path) -> Path:
    if checkpoint is not None:
        return checkpoint.resolve()
    model_path = model_path.resolve()
    config_path = model_path / "model_config.json"
    if config_path.is_file():
        metadata = json.loads(config_path.read_text(encoding="utf-8"))
        configured = metadata.get("checkpoint")
        if configured:
            candidate = model_path / str(configured)
            if candidate.is_file():
                return candidate
    for name in ("bridge_track6_10c.pth", "ecdeim_10c_bridge.pth"):
        candidate = model_path / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No bridge checkpoint found in {model_path}; pass --checkpoint explicitly."
    )


def _log_deim_metrics(output: Path, logger: Any) -> None:
    log_path = output / "log.txt"
    if not log_path.is_file():
        return
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        step = int(row.get("epoch", 0))
        if "train_loss" in row:
            logger.log_metric("train_loss", float(row["train_loss"]), step=step)
        values = row.get("test_coco_eval_bbox")
        if isinstance(values, list):
            for index, value in enumerate(values):
                logger.log_metric(f"test_coco_eval_bbox_{index}", float(value), step=step)


def _collect_artifacts(output: Path, config: Path, logger: Any) -> dict[str, Any]:
    model_dir = Path(logger.path_model())
    checkpoint_dir = Path(logger.path_model_checkpoints())
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    names = ["best_stg2.pth", "best_stg1.pth", "last.pth"]
    available = [name for name in names if (output / name).is_file()]
    if not available:
        raise RuntimeError(f"Training completed without a checkpoint in {output}.")
    best = available[0]
    shutil.copy2(output / best, model_dir / "checkpoint_best_deim.pth")
    for name in available:
        shutil.copy2(output / name, checkpoint_dir / name)
    for path in (output / "generated_deim.yml", output / "run.json", config):
        if path.is_file():
            shutil.copy2(path, model_dir / path.name)
    metadata = {
        "name": "EC-DEIM-D-FINE-X",
        "checkpoint": "checkpoint_best_deim.pth",
        "deim_config": "generated_deim.yml",
        "experiment_config": config.name,
        "num_classes": len(TRACK6_CLASSES),
        "resolution": 896,
        "class_names": TRACK6_CLASSES,
        "best_source": best,
    }
    (model_dir / "model_config.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {"model_dir": str(model_dir), "best_source": best, "checkpoints": available}


def main() -> None:
    try:
        from hafnia.experiment import HafniaLogger
    except ImportError as error:
        raise RuntimeError("Run this entrypoint inside a Hafnia environment.") from error
    args = parse_args()
    if args.resume is not None and args.checkpoint is not None:
        raise ValueError("Use either --checkpoint or --resume, not both.")
    logger = HafniaLogger(project_name="ec-deim")
    try:
        dataset = prepare_training_splits(
            load_dataset(args.dataset_path),
            samples=args.samples,
            keep_empty_ratio=args.empty_image_keep_ratio,
            seed=args.seed,
        )
        paths = export_coco(
            dataset,
            args.data_output,
            get_bbox_task_name(dataset, args.task_name),
        )
        initialization = None if args.resume else resolve_checkpoint(args.checkpoint, args.model_path)
        command = [
            sys.executable,
            str(ROOT / "scripts" / "train.py"),
            str(args.config.resolve()),
            "--deim-root",
            str(args.deim_root.resolve()),
            "--train-images",
            str(paths["train_images"]),
            "--train-annotations",
            str(paths["train_annotations"]),
            "--val-images",
            str(paths["val_images"]),
            "--val-annotations",
            str(paths["val_annotations"]),
            "--output",
            str(args.output.resolve()),
            "--devices",
            args.devices,
            "--seed",
            str(args.seed),
        ]
        if initialization is not None:
            command.extend(["--checkpoint", str(initialization)])
        if args.resume is not None:
            command.extend(["--resume", str(args.resume.resolve())])
        if args.no_amp:
            command.append("--no-amp")
        logger.log_configuration(
            {
                "command": command,
                "samples": args.samples,
                "empty_image_keep_ratio": args.empty_image_keep_ratio,
                "augmentation_profile": "oadc_plus_deim_native_light",
            }
        )
        subprocess.run(command, check=True, cwd=ROOT)
        output = args.output.resolve()
        _log_deim_metrics(output, logger)
        manifest = _collect_artifacts(output, args.config.resolve(), logger)
        logger.log_configuration({"artifacts": manifest})
        print(json.dumps(manifest, indent=2))
    finally:
        logger.end_run()


if __name__ == "__main__":
    main()
