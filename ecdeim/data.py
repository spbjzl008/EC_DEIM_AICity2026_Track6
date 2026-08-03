"""Build the heterogeneous public pretraining set from COCO-format sources."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
from typing import Any, Iterable

import yaml

from .taxonomy import CLASS_TO_ID, PRETRAIN_CLASSES, TRACK6_CLASSES


IGNORE_BICYCLE = "__ignore_bicycle__"
COMMON_KEEP_PROBABILITY = {
    "has_bicycle_or_motorcycle": 1.0,
    "has_fine_truck": 1.0,
    "has_van": 0.8,
    "has_truck_generic": 0.6,
    "normal_mixed": 0.5,
    "car_person_common": 0.25,
    "car_only": 0.15,
    "person_only": 0.20,
    "empty": 1.0,
}
REPEAT_CAP = {
    "Vehicle.Car": 1.0,
    "Person": 1.0,
    "Vehicle.Truck_Generic": 1.0,
    "_Other": 1.0,
    "Vehicle.Bicycle": 2.3,
    "Vehicle.Motorcycle": 2.0,
    "Vehicle.Van": 1.5,
    "Vehicle.Pickup Truck": 1.4,
    "Vehicle.Single Truck": 1.35,
    "Vehicle.Combo Truck": 1.30,
    "Vehicle.Heavy Duty Vehicle": 1.20,
    "Vehicle.Trailer": 1.20,
}
MILD_IRFS_CLASSES = (
    "Vehicle.Pickup Truck",
    "Vehicle.Single Truck",
    "Vehicle.Combo Truck",
    "Vehicle.Heavy Duty Vehicle",
    "Vehicle.Trailer",
    "Vehicle.Motorcycle",
    "Vehicle.Bicycle",
    "Vehicle.Van",
    "Person",
)
MILD_IRFS_IDS = frozenset(CLASS_TO_ID[name] for name in MILD_IRFS_CLASSES)
FINE_TRUCK_IDS = frozenset(CLASS_TO_ID[name] for name in TRACK6_CLASSES[1:6])
SMALL_OBJECT_IDS = frozenset(
    CLASS_TO_ID[name] for name in ("Vehicle.Motorcycle", "Vehicle.Bicycle", "Person")
)


@dataclass(frozen=True)
class ObjectLabel:
    category_id: int
    bbox: tuple[float, float, float, float]
    ignore: bool = False


@dataclass(frozen=True)
class Candidate:
    source: str
    image: Path
    width: int
    height: int
    objects: tuple[ObjectLabel, ...]
    digest: str


def _resolve_path(root: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _split_specs(source: dict[str, Any], split: str) -> list[dict[str, Any]]:
    splits = source.get("splits")
    if not isinstance(splits, dict):
        raise TypeError(f"Source {source.get('name')} must define COCO splits.")
    value = splits.get(split)
    if value is None:
        raise ValueError(f"Source {source.get('name')} has no {split} split.")
    specs = value if isinstance(value, list) else [value]
    if not specs:
        raise ValueError(f"Source {source.get('name')} has an empty {split} split.")
    for spec in specs:
        if not isinstance(spec, dict) or not {"images", "annotations"} <= spec.keys():
            raise TypeError(
                f"Each {split} split for {source.get('name')} needs images and annotations."
            )
    return specs


def _normalized_coco_bbox(
    values: Any,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        return None
    try:
        x, y, box_width, box_height = (float(value) for value in values)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x, y, box_width, box_height)):
        return None
    x1, y1 = max(0.0, x), max(0.0, y)
    x2 = min(float(width), x + box_width)
    y2 = min(float(height), y + box_height)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1 / width, y1 / height, (x2 - x1) / width, (y2 - y1) / height


def _load_coco_split(
    source_name: str,
    root: Path,
    spec: dict[str, Any],
    mapping: dict[str, str | None],
) -> list[Candidate]:
    image_root = _resolve_path(root, spec["images"])
    annotation_path = _resolve_path(root, spec["annotations"])
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)

    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"COCO annotation must be an object: {annotation_path}")
    images = payload.get("images")
    annotations = payload.get("annotations")
    categories = payload.get("categories")
    if not all(isinstance(items, list) for items in (images, annotations, categories)):
        raise TypeError(
            f"COCO annotation needs images, annotations, and categories lists: {annotation_path}"
        )

    category_names: dict[int, str] = {}
    for category in categories:
        category_id = int(category["id"])
        if category_id in category_names:
            raise ValueError(f"Duplicate COCO category id {category_id}: {annotation_path}")
        category_names[category_id] = str(category["name"])

    by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        if category_id not in category_names:
            raise ValueError(f"Unknown COCO category id {category_id} in {annotation_path}")
        by_image.setdefault(image_id, []).append(annotation)

    candidates: list[Candidate] = []
    seen_image_ids: set[int] = set()
    for image_record in images:
        image_id = int(image_record["id"])
        if image_id in seen_image_ids:
            raise ValueError(f"Duplicate COCO image id {image_id}: {annotation_path}")
        seen_image_ids.add(image_id)
        width, height = int(image_record["width"]), int(image_record["height"])
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions for id {image_id}: {annotation_path}")
        image = _resolve_path(image_root, image_record["file_name"])
        if not image.is_file():
            raise FileNotFoundError(image)

        objects: list[ObjectLabel] = []
        for annotation in by_image.get(image_id, []):
            source_category = category_names[int(annotation["category_id"])]
            target_name = mapping.get(source_category)
            if target_name is None:
                continue
            protocol_ignore = target_name == IGNORE_BICYCLE
            # Native COCO crowd annotations are not bicycle-protocol conflicts.
            # Match the pinned DEIM loader by filtering them; reserve iscrowd=1
            # in the unified dataset for explicitly mapped rider--bicycle boxes.
            if bool(annotation.get("iscrowd", 0)) and not protocol_ignore:
                continue
            bbox = _normalized_coco_bbox(annotation.get("bbox"), width, height)
            if bbox is None:
                continue
            category_id = CLASS_TO_ID["Vehicle.Bicycle" if protocol_ignore else target_name]
            objects.append(ObjectLabel(category_id, bbox, protocol_ignore))
        candidates.append(
            Candidate(source_name, image, width, height, tuple(objects), _sha256(image))
        )

    orphan_ids = set(by_image) - seen_image_ids
    if orphan_ids:
        raise ValueError(
            f"COCO annotations reference missing image ids {sorted(orphan_ids)[:5]}: "
            f"{annotation_path}"
        )
    return candidates


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bucket(objects: tuple[ObjectLabel, ...]) -> str:
    ids = {obj.category_id for obj in objects if not obj.ignore}
    if not ids:
        return "empty"
    if ids & {CLASS_TO_ID["Vehicle.Bicycle"], CLASS_TO_ID["Vehicle.Motorcycle"]}:
        return "has_bicycle_or_motorcycle"
    if ids & FINE_TRUCK_IDS:
        return "has_fine_truck"
    if CLASS_TO_ID["Vehicle.Van"] in ids:
        return "has_van"
    if CLASS_TO_ID["Vehicle.Truck_Generic"] in ids:
        return "has_truck_generic"
    car, person = CLASS_TO_ID["Vehicle.Car"], CLASS_TO_ID["Person"]
    if ids == {car}:
        return "car_only"
    if ids == {person}:
        return "person_only"
    if ids <= {car, person}:
        return "car_person_common"
    return "normal_mixed"


def _keep_probability(source: str, objects: tuple[ObjectLabel, ...]) -> float:
    class_ids = {obj.category_id for obj in objects if not obj.ignore}
    if source in {"VisDrone2019", "SODA10M"} and class_ids & SMALL_OBJECT_IDS:
        return 1.0
    if source == "MIO-TCD" and class_ids & FINE_TRUCK_IDS:
        return 1.0
    car_person = {CLASS_TO_ID["Vehicle.Car"], CLASS_TO_ID["Person"]}
    if source == "COCO" and class_ids and class_ids <= car_person:
        return 0.15
    if source == "BDD100K":
        if class_ids == {CLASS_TO_ID["Vehicle.Car"]}:
            return 0.15
        if class_ids and class_ids <= car_person:
            return 0.25
    return COMMON_KEEP_PROBABILITY[_bucket(objects)]


def _collect_source(
    source: dict[str, Any],
    split: str,
    rng: random.Random,
) -> list[Candidate]:
    name = str(source["name"])
    root = Path(source["root"]).expanduser().resolve()
    mapping = source.get("mapping")
    if not isinstance(mapping, dict):
        raise TypeError(f"Source {name} has no class mapping.")
    unknown = set(mapping.values()) - set(PRETRAIN_CLASSES) - {None, IGNORE_BICYCLE}
    if unknown:
        raise ValueError(f"Source {name} maps to unknown classes: {sorted(unknown)}")
    candidates = [
        candidate
        for spec in _split_specs(source, split)
        for candidate in _load_coco_split(name, root, spec, mapping)
    ]
    rng.shuffle(candidates)
    nonempty: list[Candidate] = []
    empty: list[Candidate] = []
    for candidate in candidates:
        has_positive = any(not obj.ignore for obj in candidate.objects)
        if split == "valid" and not has_positive:
            continue
        (nonempty if has_positive else empty).append(candidate)
    if split == "train":
        rng.shuffle(empty)
        keep_empty = round(len(nonempty) * float(source.get("empty_keep_ratio", 0.02)))
        candidates = nonempty + empty[:keep_empty]
    else:
        candidates = nonempty
    rng.shuffle(candidates)
    raw_cap = source.get(f"{split}_cap")
    if raw_cap is not None:
        candidates = candidates[: int(raw_cap)]
    return candidates


def _deduplicate(candidates: Iterable[Candidate], seen: set[str]) -> tuple[list[Candidate], int]:
    unique: list[Candidate] = []
    duplicates = 0
    for candidate in candidates:
        if candidate.digest in seen:
            duplicates += 1
            continue
        seen.add(candidate.digest)
        unique.append(candidate)
    return unique, duplicates


def _repeat_factors(candidates: list[Candidate]) -> dict[int, float]:
    image_counts: Counter[int] = Counter()
    instance_counts: Counter[int] = Counter()
    for candidate in candidates:
        class_ids = {obj.category_id for obj in candidate.objects if not obj.ignore}
        image_counts.update(class_ids)
        instance_counts.update(obj.category_id for obj in candidate.objects if not obj.ignore)
    total_images = max(len(candidates), 1)
    total_instances = max(sum(instance_counts.values()), 1)
    frequencies = {
        class_id: math.sqrt(
            image_counts[class_id] / total_images * instance_counts[class_id] / total_instances
        )
        for class_id in MILD_IRFS_IDS
        if image_counts[class_id] > 0 and instance_counts[class_id] > 0
    }
    if not frequencies:
        return {}
    threshold = min(frequencies.values()) * 1.8**2
    return {
        class_id: min(
            2.0,
            REPEAT_CAP[PRETRAIN_CLASSES[class_id]],
            max(1.0, math.sqrt(threshold / frequency)),
        )
        for class_id, frequency in frequencies.items()
    }


def _repeat(
    candidates: list[Candidate],
    rng: random.Random,
    factors: dict[int, float] | None = None,
) -> list[Candidate]:
    factors = _repeat_factors(candidates) if factors is None else factors
    repeated: list[Candidate] = []
    for candidate in candidates:
        class_ids = {obj.category_id for obj in candidate.objects if not obj.ignore}
        factor = max((factors.get(class_id, 1.0) for class_id in class_ids), default=1.0)
        count = math.floor(factor)
        if rng.random() < factor - count:
            count += 1
        repeated.extend([candidate] * count)
    rng.shuffle(repeated)
    return repeated


def _materialize(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        target.symlink_to(source)
    elif mode == "hardlink":
        target.hardlink_to(source)
    elif mode == "copy":
        shutil.copy2(source, target)
    else:
        raise ValueError("link_mode must be symlink, hardlink, or copy.")


def _write_split(
    candidates: list[Candidate], output: Path, split: str, mode: str
) -> dict[str, int]:
    image_dir = output / split / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    class_counts: Counter[int] = Counter()
    occurrences: Counter[str] = Counter()
    annotation_id = 1
    for image_id, candidate in enumerate(candidates, 1):
        repeat_index = occurrences[candidate.digest]
        occurrences[candidate.digest] += 1
        filename = (
            f"{candidate.source.lower().replace(' ', '_')}_"
            f"{candidate.digest[:16]}_{repeat_index}{candidate.image.suffix.lower()}"
        )
        target = image_dir / filename
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        _materialize(candidate.image, target, mode)
        images.append(
            {
                "id": image_id,
                "file_name": filename,
                "width": candidate.width,
                "height": candidate.height,
                "source": candidate.source,
            }
        )
        for obj in candidate.objects:
            x, y, width, height = obj.bbox
            box = [
                x * candidate.width,
                y * candidate.height,
                width * candidate.width,
                height * candidate.height,
            ]
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": obj.category_id,
                    "bbox": box,
                    "area": box[2] * box[3],
                    "iscrowd": int(obj.ignore),
                }
            )
            if not obj.ignore:
                class_counts[obj.category_id] += 1
            annotation_id += 1
    categories = [{"id": index, "name": name} for index, name in enumerate(PRETRAIN_CLASSES)]
    annotation_dir = output / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    payload = {"images": images, "annotations": annotations, "categories": categories}
    (annotation_dir / f"{split}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "images": len(images),
        "annotations": len(annotations),
        **{PRETRAIN_CLASSES[key]: value for key, value in sorted(class_counts.items())},
    }


def build_public_pretraining_set(
    config_path: Path,
    output: Path,
    seed: int = 2026,
    link_mode: str = "symlink",
) -> dict[str, Any]:
    """Build deterministic COCO splits while retaining source images unchanged."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("sources"), list):
        raise TypeError("Public-source configuration must contain a sources list.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(seed))
    collected_train: list[Candidate] = []
    collected_valid: list[Candidate] = []
    enabled_sources = [source for source in config["sources"] if source.get("enabled", True)]
    for source in enabled_sources:
        train_candidates = _collect_source(source, "train", rng)
        valid_candidates = _collect_source(source, "valid", rng)
        collected_train.extend(train_candidates)
        collected_valid.extend(valid_candidates)

    seen: set[str] = set()
    valid, valid_duplicates = _deduplicate(collected_valid, seen)
    unique_train, train_duplicates = _deduplicate(collected_train, seen)
    duplicate_count = valid_duplicates + train_duplicates
    train = [
        candidate
        for candidate in unique_train
        if rng.random() <= _keep_probability(candidate.source, candidate.objects)
    ]
    # Validation is reserved first so an exact duplicate can never leak into training.
    repeat_factors = _repeat_factors(train)
    splits = {"valid": valid, "train": _repeat(train, rng, repeat_factors)}
    summary = {
        "seed": int(seed),
        "link_mode": link_mode,
        "full_image_training": True,
        "train_cap_used": any(source.get("train_cap") is not None for source in enabled_sources),
        "exact_duplicates_removed": duplicate_count,
        "unique_train_images_before_sampling": len(unique_train),
        "train_images_after_common_downsampling": len(train),
        "sampling": {
            "type": "mild_irfs",
            "enabled": True,
            "alpha": 0.5,
            "target_max_repeat": 1.8,
            "max_repeat": 2.0,
            "repeat_classes": list(MILD_IRFS_CLASSES),
            "class_repeat_factors": {
                PRETRAIN_CLASSES[class_id]: factor
                for class_id, factor in sorted(repeat_factors.items())
            },
            "per_class_repeat_max": dict(REPEAT_CAP),
        },
        "train": _write_split(splits["train"], output, "train", link_mode),
        "valid": _write_split(splits["valid"], output, "valid", link_mode),
    }
    (output / "build_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    policy = {
        "classes": PRETRAIN_CLASSES,
        "generic_truck": "Positive localization evidence without a fabricated fine subtype.",
        "ignore_regions": "Protocol-conflicting cyclist or bicycle boxes use iscrowd=1.",
    }
    (output / "pretrain_label_policy.json").write_text(
        json.dumps(policy, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def validate_coco_categories(annotation_path: Path, expected: list[str]) -> None:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = sorted(payload.get("categories", []), key=lambda item: int(item["id"]))
    ids = [int(item["id"]) for item in categories]
    names = [str(item["name"]) for item in categories]
    if ids != list(range(len(expected))) or names != expected:
        raise ValueError(
            f"COCO categories do not match the required order. Got ids={ids}, names={names}."
        )
