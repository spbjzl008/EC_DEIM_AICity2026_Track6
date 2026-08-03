#!/usr/bin/env python3
"""Create a deterministic group holdout without splitting any camera across sets."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import re
from typing import Any

from ecdeim.data import validate_coco_categories
from ecdeim.taxonomy import TRACK6_CLASSES


def camera_id(record: dict[str, Any], pattern: re.Pattern[str] | None) -> str:
    for field in ("camera_id", "camera", "video_id"):
        value = record.get(field)
        if value is not None and str(value):
            return str(value)
    if pattern is not None:
        match = pattern.search(str(record.get("file_name", "")))
        if match:
            return match.group("camera") if "camera" in match.groupdict() else match.group(1)
    raise ValueError(
        f"No camera id for image {record.get('id')}. Add a camera field or pass --camera-regex."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument(
        "--camera-regex", help="Regex with one capture group or a named 'camera' group."
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation-fraction must be in (0, 1).")
    validate_coco_categories(args.annotations, TRACK6_CLASSES)
    payload = json.loads(args.annotations.read_text(encoding="utf-8"))
    pattern = re.compile(args.camera_regex) if args.camera_regex else None
    if pattern is not None and pattern.groups < 1:
        raise ValueError("camera-regex must contain at least one capture group.")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for image in payload["images"]:
        groups[camera_id(image, pattern)].append(image)
    if len(groups) < 2:
        raise ValueError("At least two camera groups are required for a holdout.")
    rng = random.Random(args.seed)
    names = sorted(groups)
    rng.shuffle(names)
    target_size = round(len(payload["images"]) * args.validation_fraction)
    validation_cameras: set[str] = set()
    validation_size = 0
    for name in names:
        group_size = len(groups[name])
        current_gap = abs(target_size - validation_size)
        new_gap = abs(target_size - (validation_size + group_size))
        if new_gap <= current_gap or not validation_cameras:
            validation_cameras.add(name)
            validation_size += group_size
    if validation_cameras == set(groups):
        validation_cameras.remove(max(validation_cameras, key=lambda name: len(groups[name])))

    valid_ids = {
        int(image["id"])
        for name in validation_cameras
        for image in groups[name]
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for split, selected_ids in (
        ("valid", valid_ids),
        ("train", {int(image["id"]) for image in payload["images"]} - valid_ids),
    ):
        split_payload = {
            **{
                key: value
                for key, value in payload.items()
                if key not in {"images", "annotations"}
            },
            "images": [image for image in payload["images"] if int(image["id"]) in selected_ids],
            "annotations": [
                annotation
                for annotation in payload.get("annotations", [])
                if int(annotation["image_id"]) in selected_ids
            ],
        }
        (args.output / f"{split}.json").write_text(
            json.dumps(split_payload, indent=2) + "\n", encoding="utf-8"
        )
    summary = {
        "train_images": len(payload["images"]) - len(valid_ids),
        "valid_images": len(valid_ids),
        "train_cameras": sorted(set(groups) - validation_cameras),
        "valid_cameras": sorted(validation_cameras),
        "camera_overlap": [],
    }
    (args.output / "split_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
