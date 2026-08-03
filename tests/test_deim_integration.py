from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image
import torch

from ecdeim.config import dump_yaml, load_experiment, render_deim_yaml
from ecdeim.taxonomy import PRETRAIN_CLASSES


DEIM_ROOT = os.environ.get("DEIM_ROOT")


@unittest.skipUnless(DEIM_ROOT, "Set DEIM_ROOT to run upstream integration checks.")
class DEIMIntegrationTest(unittest.TestCase):
    @staticmethod
    def prepare_engine() -> Path:
        deim_root = Path(DEIM_ROOT).resolve()
        sys.path.insert(0, str(deim_root))
        from ecdeim.augmentations import register_deim_transforms
        from ecdeim.runtime import (
            install_optional_calflops_fallback,
            install_torchvision_compatibility,
        )

        install_optional_calflops_fallback()
        install_torchvision_compatibility()
        register_deim_transforms()
        return deim_root

    def test_gradient_accumulation_steps_optimizer_ema_and_scheduler_together(self) -> None:
        self.prepare_engine()
        from ecdeim.runtime import accumulated_train_one_epoch

        class ToyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, samples, targets=None):
                return {"value": self.weight * samples}

        class ToyCriterion(torch.nn.Module):
            def forward(self, outputs, targets, **metadata):
                return {"toy": outputs["value"].square().sum()}

        class Counter:
            def __init__(self) -> None:
                self.steps = 0

            def update(self, model) -> None:
                self.steps += 1

            def step(self, global_step, optimizer):
                self.steps += 1
                return optimizer

        model = ToyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loader = [(torch.tensor([1.0]), [{"dummy": torch.tensor(0.0)}]) for _ in range(3)]
        ema, scheduler = Counter(), Counter()
        accumulated_train_one_epoch.steps = 2
        accumulated_train_one_epoch(
            True,
            scheduler,
            model,
            ToyCriterion(),
            loader,
            optimizer,
            torch.device("cpu"),
            epoch=0,
            print_freq=10,
            ema=ema,
        )
        self.assertEqual(ema.steps, 2)
        self.assertEqual(scheduler.steps, 2)

    def test_adaptation_model_and_optimizer_contract(self) -> None:
        deim_root = self.prepare_engine()
        from ecdeim.adaptation import (
            LoRALinear,
            calibrate_class_heads,
            freeze_backbone,
            inject_decoder_lora,
        )
        from engine.core import YAMLConfig

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            annotations = root / "annotations.json"
            annotations.write_text("{}")
            experiment_path = Path(__file__).resolve().parents[1] / "configs" / "adapt.yaml"
            experiment = load_experiment(experiment_path)
            rendered = render_deim_yaml(
                experiment,
                deim_root,
                root / "output",
                images,
                annotations,
                images,
                annotations,
                1,
            )
            self.assertTrue(rendered["use_ema"])
            self.assertEqual(
                rendered["train_dataloader"]["dataset"]["transforms"]["mosaic_prob"],
                0.5,
            )
            config_path = root / "generated.yml"
            dump_yaml(rendered, config_path)
            config = YAMLConfig(str(config_path))
            model = config.model
            matched = inject_decoder_lora(model, rank=8, alpha=16)
            self.assertEqual(len(matched), 12)
            freeze_backbone(model)
            calibrate_class_heads(model)
            optimizer = config.optimizer
            optimized = {
                id(parameter) for group in optimizer.param_groups for parameter in group["params"]
            }
            lora_parameters = [
                parameter
                for module in model.modules()
                if isinstance(module, LoRALinear)
                for parameter in (module.lora_A.weight, module.lora_B.weight)
            ]
            self.assertTrue(lora_parameters)
            self.assertTrue(all(id(parameter) in optimized for parameter in lora_parameters))
            self.assertTrue(
                all(id(parameter) not in optimized for parameter in model.backbone.parameters())
            )

    def test_pretrain_yaml_builds_model_loss_and_dataset(self) -> None:
        deim_root = self.prepare_engine()
        from ecdeim.evidence import EvidenceConfig, install_evidence_routing

        install_evidence_routing(EvidenceConfig())
        from engine.core import YAMLConfig

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            Image.new("RGB", (96, 64), (30, 40, 50)).save(images / "sample.jpg")
            annotations = root / "annotations.json"
            annotations.write_text(
                json.dumps(
                    {
                        "images": [{"id": 1, "file_name": "sample.jpg", "width": 96, "height": 64}],
                        "annotations": [
                            {
                                "id": 1,
                                "image_id": 1,
                                "category_id": 0,
                                "bbox": [10, 10, 30, 20],
                                "area": 600,
                                "iscrowd": 0,
                            },
                            {
                                "id": 2,
                                "image_id": 1,
                                "category_id": 7,
                                "bbox": [50, 10, 20, 20],
                                "area": 400,
                                "iscrowd": 1,
                            },
                        ],
                        "categories": [
                            {"id": index, "name": name}
                            for index, name in enumerate(PRETRAIN_CLASSES)
                        ],
                    }
                )
            )
            experiment_path = Path(__file__).resolve().parents[1] / "configs" / "pretrain.yaml"
            experiment = load_experiment(experiment_path)
            rendered = render_deim_yaml(
                experiment,
                deim_root,
                root / "output",
                images,
                annotations,
                images,
                annotations,
                1,
            )
            config_path = root / "generated.yml"
            dump_yaml(rendered, config_path)
            config = YAMLConfig(str(config_path))
            self.assertEqual(config.yaml_cfg["num_classes"], 12)
            self.assertEqual(config.model.decoder.num_classes, 12)
            self.assertEqual(config.criterion.num_classes, 12)
            dataset = config.train_dataloader.dataset
            self.assertEqual(len(dataset), 1)
            transform_names = [
                type(transform).__name__ for transform in dataset._transforms.transforms
            ]
            self.assertEqual(transform_names[0], "ObjectAwareTileSampling")
            _, target = dataset[0]
            self.assertEqual(target["labels"].tolist(), [0])
            self.assertEqual(tuple(target["_ecdeim_ignore_boxes"].shape), (1, 4))


if __name__ == "__main__":
    unittest.main()
