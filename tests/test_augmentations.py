from __future__ import annotations

import random
import unittest

import numpy as np
from PIL import Image
import torch
from torchvision.tv_tensors import BoundingBoxes

from ecdeim.augmentations import ObjectAwareDomainCoverage, ObjectAwareTileSampling


class DatasetStub:
    epoch = 0


class AugmentationTest(unittest.TestCase):
    def test_oadc_schedule_and_small_object_protection(self) -> None:
        transform = ObjectAwareDomainCoverage(force_scenario="low_light")
        self.assertAlmostEqual(transform.probability_at(0), 0.9)
        self.assertAlmostEqual(transform.probability_at(4), 0.9)
        self.assertAlmostEqual(transform.probability_at(6), 0.45)
        self.assertEqual(transform.probability_at(8), 0.0)
        random.seed(3)
        alpha = transform._alpha_map((100, 100), torch.tensor([[40, 40, 45, 45]]))
        self.assertLess(float(alpha[42, 42, 0]), float(alpha[0, 0, 0]))

    def test_oats_keeps_instance_fields_aligned(self) -> None:
        transform = ObjectAwareTileSampling(
            mode_probabilities={
                "whole_image": 0.0,
                "random_tile": 1.0,
                "rare_object_tile": 0.0,
                "empty_tile": 0.0,
            },
            random_crop_sizes=[64],
            output_size=64,
        )
        image = Image.fromarray(np.full((96, 96, 3), 127, dtype=np.uint8))
        target = {
            "boxes": BoundingBoxes(
                torch.tensor([[20.0, 20.0, 60.0, 60.0]]),
                format="XYXY",
                canvas_size=(96, 96),
            ),
            "labels": torch.tensor([0]),
            "area": torch.tensor([1600.0]),
            "iscrowd": torch.tensor([0]),
        }
        cropped, output, retained = transform._crop_target(image, target, (0, 0, 64, 64))
        self.assertEqual(cropped.size, (64, 64))
        self.assertEqual(retained, 1)
        self.assertEqual(len(output["boxes"]), len(output["labels"]))
        self.assertEqual(len(output["boxes"]), len(output["area"]))


if __name__ == "__main__":
    unittest.main()
