#!/usr/bin/env python3
"""Run one 896-pixel full-image pass and write COCO detections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from ecdeim.inference import Predictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deim-root", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, required=True, help="EC-DEIM adaptation or generated DEIM YAML."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True, help="COCO image manifest.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--image-size", type=int, default=896)
    parser.add_argument("--device")
    args = parser.parse_args()

    manifest = json.loads(args.annotations.read_text(encoding="utf-8"))
    images = manifest.get("images")
    if not isinstance(images, list):
        raise TypeError("COCO annotations must contain an images list.")
    predictor = Predictor(
        args.deim_root,
        args.config,
        args.checkpoint,
        args.threshold,
        args.image_size,
        args.device,
    )
    detections = []
    for index, record in enumerate(images, 1):
        image_path = args.images / record["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image:
            predictions = predictor.predict(image)
        for prediction in predictions:
            prediction.pop("category_name")
            prediction["image_id"] = int(record["id"])
            detections.append(prediction)
        if index % 100 == 0:
            print(f"Processed {index}/{len(images)} images.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(detections, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(detections)} detections to {args.output}")


if __name__ == "__main__":
    main()
