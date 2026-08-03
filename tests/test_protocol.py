from __future__ import annotations

from pathlib import Path
import unittest

from ecdeim.config import _adapt_transforms, _pretrain_transforms, load_experiment


ROOT = Path(__file__).resolve().parents[1]


class ProtocolTest(unittest.TestCase):
    def test_released_training_contract(self) -> None:
        pretrain = load_experiment(ROOT / "configs" / "pretrain.yaml")
        adapt = load_experiment(ROOT / "configs" / "adapt.yaml")
        self.assertEqual(pretrain["model"]["num_classes"], 12)
        self.assertEqual(pretrain["model"]["image_size"], 896)
        self.assertEqual(
            pretrain["augmentation"]["oats"]["mode_probabilities"],
            {
                "whole_image": 0.74,
                "random_tile": 0.14,
                "rare_object_tile": 0.08,
                "empty_tile": 0.04,
            },
        )
        self.assertEqual(adapt["model"]["num_classes"], 10)
        self.assertEqual(adapt["model"]["image_size"], 896)
        self.assertEqual(adapt["training"]["epochs"], 16)
        self.assertEqual(adapt["training"]["micro_batch_per_gpu"], 3)
        self.assertEqual(adapt["training"]["gradient_accumulation_steps"], 10)
        self.assertEqual(adapt["optimizer"]["weight_decay"], 0.000125)
        self.assertEqual(
            (
                adapt["optimizer"]["class_head_lr"],
                adapt["optimizer"]["box_lr"],
                adapt["optimizer"]["decoder_lr"],
            ),
            (0.00005, 0.00002, 0.000014),
        )
        self.assertEqual(adapt["adaptation"]["lora"], {"enabled": True, "rank": 8, "alpha": 16})
        self.assertEqual(adapt["augmentation"]["oadc"]["zero_at_epoch"], 8)
        self.assertTrue(pretrain["training"]["full_image_training"])
        self.assertTrue(pretrain["training"]["use_ema"])
        self.assertTrue(adapt["training"]["use_ema"])
        self.assertEqual(pretrain["augmentation"]["profile"], "oats_plus_deim_native")
        self.assertEqual(pretrain["augmentation"]["policy_epochs"], [0, 6, 7])
        self.assertEqual(pretrain["augmentation"]["mixup_epochs"], [0, 3])
        self.assertEqual(adapt["augmentation"]["profile"], "oadc_plus_deim_native_light")
        self.assertEqual(adapt["augmentation"]["mixup_probability"], 0.5)
        self.assertEqual(adapt["augmentation"]["base_size_repeat"], 3)
        self.assertEqual(adapt["augmentation"]["policy_epochs"], [2, 6, 12])
        self.assertEqual(adapt["augmentation"]["oadc"]["max_probability"], 0.30)

    def test_new_transforms_keep_non_conflicting_deim_native_ops(self) -> None:
        pretrain = load_experiment(ROOT / "configs" / "pretrain.yaml")
        pretrain_transforms = _pretrain_transforms(pretrain, 896)
        self.assertEqual(
            [item["type"] for item in pretrain_transforms["ops"][:6]],
            [
                "ObjectAwareTileSampling",
                "Mosaic",
                "RandomPhotometricDistort",
                "RandomZoomOut",
                "RandomIoUCrop",
                "SanitizeBoundingBoxes",
            ],
        )
        self.assertEqual(pretrain_transforms["policy"]["epoch"], [0, 6, 7])

        adapt = load_experiment(ROOT / "configs" / "adapt.yaml")
        adapt_transforms = _adapt_transforms(adapt, 896)
        adapt_types = [item["type"] for item in adapt_transforms["ops"]]
        self.assertEqual(
            adapt_types[:6],
            [
                "ObjectAwareDomainCoverage",
                "RandomPhotometricDistort",
                "RandomZoomOut",
                "SanitizeBoundingBoxes",
                "RandomHorizontalFlip",
                "Resize",
            ],
        )
        self.assertNotIn("Mosaic", adapt_types)
        self.assertNotIn("RandomIoUCrop", adapt_types)
        self.assertEqual(adapt_transforms["policy"]["epoch"], [2, 6, 12])
        self.assertEqual(adapt_transforms["mosaic_prob"], 0.5)
        self.assertEqual(
            adapt_transforms["policy"]["ops"],
            ["RandomPhotometricDistort", "RandomZoomOut"],
        )

    def test_smoke_config_only_overrides_schedule(self) -> None:
        smoke = load_experiment(ROOT / "configs" / "adapt_smoke.yaml")
        self.assertEqual(smoke["training"]["epochs"], 1)
        self.assertEqual(smoke["augmentation"]["profile"], "oadc_plus_deim_native_light")
        self.assertTrue(smoke["adaptation"]["lora"]["enabled"])


if __name__ == "__main__":
    unittest.main()
