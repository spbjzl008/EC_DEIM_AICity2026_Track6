from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import yaml

from ecdeim.data import IGNORE_BICYCLE, build_public_pretraining_set, validate_coco_categories
from ecdeim.taxonomy import PRETRAIN_CLASSES


def source(root: Path, name: str, color: tuple[int, int, int]) -> None:
    for split in ("train", "val"):
        image_dir = root / "images" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        image = image_dir / f"{name}_{split}.jpg"
        split_color = color if split == "train" else tuple(min(value + 1, 255) for value in color)
        Image.new("RGB", (64, 48), split_color).save(image)
        annotation_dir = root / "annotations"
        annotation_dir.mkdir(exist_ok=True)
        (annotation_dir / f"{split}.json").write_text(
            json.dumps(
                {
                    "images": [
                        {
                            "id": 1,
                            "file_name": image.name,
                            "width": 64,
                            "height": 48,
                        }
                    ],
                    "annotations": [
                        {
                            "id": 1,
                            "image_id": 1,
                            "category_id": 42,
                            "bbox": [16, 12, 32, 24],
                            "area": 768,
                            "iscrowd": 0,
                        }
                    ],
                    "categories": [{"id": 42, "name": "bicycle"}],
                }
            )
        )


class DataTest(unittest.TestCase):
    def test_builder_outputs_valid_coco_and_removes_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first", root / "second"
            source(first, "first", (10, 20, 30))
            source(second, "second", (10, 20, 30))
            config = {
                "sources": [
                    {
                        "name": "first",
                        "root": str(first),
                        "empty_keep_ratio": 1.0,
                        "splits": {
                            "train": {
                                "images": "images/train",
                                "annotations": "annotations/train.json",
                            },
                            "valid": {
                                "images": "images/val",
                                "annotations": "annotations/val.json",
                            },
                        },
                        "mapping": {"bicycle": "Vehicle.Bicycle"},
                    },
                    {
                        "name": "second",
                        "root": str(second),
                        "empty_keep_ratio": 1.0,
                        "splits": {
                            "train": {
                                "images": "images/train",
                                "annotations": "annotations/train.json",
                            },
                            "valid": {
                                "images": "images/val",
                                "annotations": "annotations/val.json",
                            },
                        },
                        "mapping": {"bicycle": "Vehicle.Bicycle"},
                    },
                ]
            }
            config_path = root / "sources.yaml"
            config_path.write_text(yaml.safe_dump(config))
            output = root / "output"
            summary = build_public_pretraining_set(config_path, output, link_mode="copy")
            self.assertGreaterEqual(summary["exact_duplicates_removed"], 1)
            self.assertTrue(summary["full_image_training"])
            self.assertFalse(summary["train_cap_used"])
            self.assertEqual(summary["sampling"]["type"], "mild_irfs")
            self.assertTrue(summary["sampling"]["enabled"])
            self.assertIn("Vehicle.Bicycle", summary["sampling"]["repeat_classes"])
            validate_coco_categories(output / "annotations" / "train.json", PRETRAIN_CLASSES)
            train = json.loads((output / "annotations" / "train.json").read_text())
            self.assertTrue(train["images"])
            self.assertTrue(
                all(annotation["category_id"] == 7 for annotation in train["annotations"])
            )

    def test_native_coco_crowd_is_not_bicycle_protocol_ignore(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "source"
            source(dataset, "crowd", (30, 40, 50))
            for split in ("train", "val"):
                annotation_path = dataset / "annotations" / f"{split}.json"
                payload = json.loads(annotation_path.read_text())
                payload["categories"].extend(
                    [
                        {"id": 43, "name": "person"},
                        {"id": 44, "name": "Cyclist"},
                    ]
                )
                payload["annotations"].extend(
                    [
                        {
                            "id": 2,
                            "image_id": 1,
                            "category_id": 43,
                            "bbox": [2, 3, 10, 12],
                            "area": 120,
                            "iscrowd": 1,
                        },
                        {
                            "id": 3,
                            "image_id": 1,
                            "category_id": 44,
                            "bbox": [4, 5, 12, 14],
                            "area": 168,
                            "iscrowd": 0,
                        },
                    ]
                )
                annotation_path.write_text(json.dumps(payload))

            config = {
                "sources": [
                    {
                        "name": "crowd",
                        "root": str(dataset),
                        "splits": {
                            "train": {
                                "images": "images/train",
                                "annotations": "annotations/train.json",
                            },
                            "valid": {
                                "images": "images/val",
                                "annotations": "annotations/val.json",
                            },
                        },
                        "mapping": {
                            "bicycle": "Vehicle.Bicycle",
                            "person": "Person",
                            "Cyclist": IGNORE_BICYCLE,
                        },
                    }
                ]
            }
            config_path = root / "sources.yaml"
            config_path.write_text(yaml.safe_dump(config))
            output = root / "output"
            build_public_pretraining_set(config_path, output, link_mode="copy")

            valid = json.loads((output / "annotations" / "valid.json").read_text())
            self.assertEqual(len(valid["annotations"]), 2)
            self.assertEqual(
                sorted(annotation["iscrowd"] for annotation in valid["annotations"]),
                [0, 1],
            )
            protocol_ignore = next(
                annotation for annotation in valid["annotations"] if annotation["iscrowd"] == 1
            )
            self.assertEqual(protocol_ignore["category_id"], 7)
            self.assertEqual(protocol_ignore["bbox"], [4.0, 5.0, 12.0, 14.0])


if __name__ == "__main__":
    unittest.main()
