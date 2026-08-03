"""Semantic checkpoint conversion for the two EC-DEIM training stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch

from .taxonomy import PRETRAIN_CLASSES, TRACK6_CLASSES


COCO_ROWS = {
    "Person": 0,
    "Vehicle.Bicycle": 1,
    "Vehicle.Car": 2,
    "Vehicle.Motorcycle": 3,
}
COCO_BUS_ROW = 5
COCO_TRUCK_ROW = 7
FINE_TRUCK_CLASSES = frozenset(
    {
        "Vehicle.Pickup Truck",
        "Vehicle.Single Truck",
        "Vehicle.Combo Truck",
        "Vehicle.Heavy Duty Vehicle",
        "Vehicle.Trailer",
    }
)
CLASSIFICATION_MARKERS = ("enc_score_head", "dec_score_head")


def checkpoint_states(payload: dict[str, Any]) -> list[tuple[str, dict[str, torch.Tensor]]]:
    """Return all trainable model states carried by a DEIM checkpoint."""
    states: list[tuple[str, dict[str, torch.Tensor]]] = []
    model = payload.get("model")
    if isinstance(model, dict):
        states.append(("model", model))
    ema = payload.get("ema")
    if isinstance(ema, dict) and isinstance(ema.get("module"), dict):
        states.append(("ema.module", ema["module"]))
    if not states:
        raise TypeError("Checkpoint must contain a model or ema.module state dictionary.")
    return states


def _semantic_rows(source: torch.Tensor, seed: int) -> torch.Tensor:
    if source.shape[0] != 80:
        raise ValueError(f"Expected an 80-class COCO tensor, got {tuple(source.shape)}.")
    source_cpu = source.detach().cpu()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    rows: list[torch.Tensor] = []
    for class_name in PRETRAIN_CLASSES:
        if class_name == "Vehicle.Van":
            row = 0.5 * (
                source_cpu[COCO_ROWS["Vehicle.Car"]].float()
                + source_cpu[COCO_TRUCK_ROW].float()
            )
            rows.append(row.to(source.dtype))
        elif class_name in FINE_TRUCK_CLASSES:
            row = source_cpu[COCO_TRUCK_ROW].clone()
            if row.ndim:
                noise = torch.randn(row.shape, generator=generator, dtype=torch.float32)
                scale = row.float().std().clamp_min(1e-6) * 0.01
                row = (row.float() + noise * scale).to(source.dtype)
            rows.append(row)
        elif class_name == "Vehicle.Truck_Generic":
            rows.append(source_cpu[COCO_TRUCK_ROW].clone())
        elif class_name == "_Other":
            rows.append(source_cpu[COCO_BUS_ROW].clone())
        else:
            rows.append(source_cpu[COCO_ROWS[class_name]].clone())
    return torch.stack(rows).to(device=source.device, dtype=source.dtype)


def _remap_coco_state(state: dict[str, torch.Tensor], seed: int) -> list[str]:
    converted: list[str] = []
    for key, value in list(state.items()):
        if not isinstance(value, torch.Tensor):
            continue
        if any(marker in key for marker in CLASSIFICATION_MARKERS):
            if value.ndim in (1, 2) and value.shape[0] == 80:
                state[key] = _semantic_rows(value, seed)
                converted.append(key)
        elif key.endswith("denoising_class_embed.weight"):
            if value.ndim == 2 and value.shape[0] == 81:
                state[key] = torch.cat([_semantic_rows(value[:-1], seed), value[-1:]], dim=0)
                converted.append(key)
    return converted


def initialize_pretrain_checkpoint(
    input_path: Path,
    output_path: Path,
    seed: int = 2026,
) -> dict[str, Any]:
    """Convert an official 80-class DEIM-D-FINE-X checkpoint to 12 classes."""
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    payload = torch.load(input_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload: {type(payload).__name__}.")
    converted: dict[str, list[str]] = {}
    for state_name, state in checkpoint_states(payload):
        converted[state_name] = _remap_coco_state(state, int(seed))
        if not converted[state_name]:
            raise RuntimeError(f"No 80-class DEIM tensors were found in {state_name}.")
    metadata = {
        "class_names": list(PRETRAIN_CLASSES),
        "source": input_path.name,
        "seed": int(seed),
        "fine_truck_noise_scale": 0.01,
        "converted": converted,
    }
    payload["ecdeim_semantic_head"] = metadata
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return metadata


def _source_class_names(payload: dict[str, Any]) -> list[str]:
    for metadata_key in ("ecdeim_semantic_head", "track6_semantic_head"):
        metadata = payload.get(metadata_key)
        if isinstance(metadata, dict) and isinstance(metadata.get("class_names"), (list, tuple)):
            return list(metadata["class_names"])
    raise ValueError(
        "The source checkpoint has no semantic class metadata. "
        "Use the EC-DEIM checkpoint initializer before public pretraining."
    )


def _take_rows(value: torch.Tensor, indices: Iterable[int]) -> torch.Tensor:
    index = torch.as_tensor(list(indices), dtype=torch.long, device=value.device)
    return value.index_select(0, index).clone()


def bridge_pretrain_checkpoint(input_path: Path, output_path: Path) -> dict[str, Any]:
    """Remove auxiliary classes by name while preserving the denoising background row."""
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    payload = torch.load(input_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload: {type(payload).__name__}.")
    source_names = _source_class_names(payload)
    if len(source_names) != len(PRETRAIN_CLASSES) or set(source_names) != set(PRETRAIN_CLASSES):
        raise ValueError(
            "Source classes do not match the 12-class EC-DEIM pretraining taxonomy: "
            f"{source_names}"
        )
    keep = [source_names.index(name) for name in TRACK6_CLASSES]
    converted: dict[str, list[str]] = {}
    for state_name, state in checkpoint_states(payload):
        converted[state_name] = []
        for key, value in list(state.items()):
            if not isinstance(value, torch.Tensor):
                continue
            if any(marker in key for marker in CLASSIFICATION_MARKERS):
                if value.ndim in (1, 2) and value.shape[0] == len(source_names):
                    state[key] = _take_rows(value, keep)
                    converted[state_name].append(key)
            elif key.endswith("denoising_class_embed.weight"):
                if value.ndim == 2 and value.shape[0] == len(source_names) + 1:
                    state[key] = torch.cat([_take_rows(value, keep), value[-1:].clone()], dim=0)
                    converted[state_name].append(key)
        if not converted[state_name]:
            raise RuntimeError(f"No 12-class DEIM tensors were found in {state_name}.")
    metadata = {
        "class_names": list(TRACK6_CLASSES),
        "source_class_names": source_names,
        "source": input_path.name,
        "converted": converted,
    }
    payload["ecdeim_bridge"] = metadata
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return metadata


def model_state(payload: dict[str, Any], prefer_ema: bool = True) -> dict[str, torch.Tensor]:
    """Select the EMA state for evaluation when it is available."""
    if prefer_ema:
        ema = payload.get("ema")
        if isinstance(ema, dict) and isinstance(ema.get("module"), dict):
            return ema["module"]
    model = payload.get("model")
    if not isinstance(model, dict):
        raise TypeError("Checkpoint has no model state dictionary.")
    return model


def validate_checkpoint_taxonomy(path: Path, expected_names: list[str]) -> None:
    """Reject silent partial loading caused by a wrong class width or row order."""
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint must be a dictionary.")
    widths: set[int] = set()
    for _, state in checkpoint_states(payload):
        for key, value in state.items():
            if (
                isinstance(value, torch.Tensor)
                and any(marker in key for marker in CLASSIFICATION_MARKERS)
                and value.ndim in (1, 2)
            ):
                widths.add(int(value.shape[0]))
    if widths != {len(expected_names)}:
        raise ValueError(
            f"Checkpoint classification widths are {sorted(widths)}; "
            f"expected {len(expected_names)}."
        )
    metadata_candidates = (
        payload.get("ecdeim_bridge"),
        payload.get("ecdeim_semantic_head"),
        payload.get("ecdeim_training"),
        payload.get("track6_bridge"),
        payload.get("track6_semantic_head"),
    )
    declared = None
    for metadata in metadata_candidates:
        if isinstance(metadata, dict) and isinstance(metadata.get("class_names"), (list, tuple)):
            if len(metadata["class_names"]) == len(expected_names):
                declared = list(metadata["class_names"])
                break
    if declared is not None and declared != expected_names:
        raise ValueError(
            "Checkpoint class-row order does not match the experiment. "
            f"Declared={declared}, expected={expected_names}"
        )
