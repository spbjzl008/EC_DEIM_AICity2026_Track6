"""In-process bridge between EC-DEIM components and the pinned DEIM engine."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import torch

from .adaptation import install_adaptation_hooks
from .augmentations import register_deim_transforms
from .evidence import EvidenceConfig, install_evidence_routing


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deim-root", type=Path, required=True)
    parser.add_argument("--deim-config", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    return parser


def _load_experiment(path: Path) -> dict[str, Any]:
    from .config import load_experiment

    return load_experiment(path)


def _annotate_checkpoints(output_dir: Path, experiment: dict[str, Any]) -> None:
    from engine.misc import dist_utils
    from .taxonomy import PRETRAIN_CLASSES, TRACK6_CLASSES

    if not dist_utils.is_main_process():
        return
    classes = PRETRAIN_CLASSES if experiment["stage"] == "pretrain" else TRACK6_CLASSES
    metadata = {
        "stage": experiment["stage"],
        "class_names": list(classes),
        "image_size": int(experiment["model"]["image_size"]),
    }
    if experiment["stage"] == "adapt":
        metadata["lora"] = dict(experiment.get("adaptation", {}).get("lora", {}))
    checkpoint_paths = [
        output_dir / name for name in ("best_stg1.pth", "best_stg2.pth", "last.pth")
    ]
    for checkpoint_path in checkpoint_paths:
        if not checkpoint_path.is_file():
            continue
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "model" not in payload:
            continue
        payload["ecdeim_training"] = metadata
        if experiment["stage"] == "pretrain":
            payload["ecdeim_semantic_head"] = {
                "class_names": list(PRETRAIN_CLASSES),
                "source": checkpoint_path.name,
            }
        torch.save(payload, checkpoint_path)


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    deim_root = args.deim_root.resolve()
    if not (deim_root / "engine" / "__init__.py").is_file():
        raise FileNotFoundError(f"DEIM engine not found at {deim_root}")
    sys.path.insert(0, str(deim_root))

    from .runtime import (
        install_gradient_accumulation,
        install_memory_efficient_box_loss,
        install_optional_calflops_fallback,
        install_torchvision_compatibility,
    )

    install_optional_calflops_fallback()
    install_torchvision_compatibility()
    install_memory_efficient_box_loss()
    register_deim_transforms()
    experiment = _load_experiment(args.experiment)
    from .checkpoints import validate_checkpoint_taxonomy
    from .taxonomy import PRETRAIN_CLASSES, TRACK6_CLASSES

    checkpoint_to_validate = args.resume or args.checkpoint
    if checkpoint_to_validate is not None:
        expected_names = PRETRAIN_CLASSES if experiment["stage"] == "pretrain" else TRACK6_CLASSES
        validate_checkpoint_taxonomy(checkpoint_to_validate, expected_names)
    install_gradient_accumulation(int(experiment["training"]["gradient_accumulation_steps"]))

    if experiment["stage"] == "pretrain":
        install_evidence_routing(EvidenceConfig.from_dict(experiment.get("evidence")))
    else:
        install_adaptation_hooks(experiment.get("adaptation", {}), resume=args.resume is not None)

    from engine.core import YAMLConfig
    from engine.misc import dist_utils
    from engine.solver import TASKS

    if args.checkpoint is not None and args.resume is not None:
        raise ValueError("Use either --checkpoint or --resume, not both.")
    dist_utils.setup_distributed(print_rank=0, print_method="builtin", seed=args.seed)
    overrides = {
        "tuning": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "resume": str(args.resume.resolve()) if args.resume else None,
        "use_amp": bool(args.use_amp),
        "seed": int(args.seed),
        "test_only": bool(args.test_only),
        "device": "",
    }
    config = YAMLConfig(str(args.deim_config.resolve()), **overrides)
    if args.checkpoint or args.resume:
        config.yaml_cfg.setdefault("HGNetv2", {})["pretrained"] = False
    solver = TASKS[config.yaml_cfg["task"]](config)
    if args.test_only:
        solver.val()
    else:
        solver.fit()
        _annotate_checkpoints(Path(config.yaml_cfg["output_dir"]), experiment)
    dist_utils.cleanup()


if __name__ == "__main__":
    main()
