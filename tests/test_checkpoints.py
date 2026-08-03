from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from ecdeim.checkpoints import (
    bridge_pretrain_checkpoint,
    initialize_pretrain_checkpoint,
    validate_checkpoint_taxonomy,
)
from ecdeim.taxonomy import PRETRAIN_CLASSES, TRACK6_CLASSES


def state() -> dict[str, torch.Tensor]:
    weight = torch.arange(80 * 4, dtype=torch.float32).reshape(80, 4)
    return {
        "decoder.enc_score_head.weight": weight.clone(),
        "decoder.enc_score_head.bias": torch.arange(80, dtype=torch.float32),
        "decoder.dec_score_head.0.weight": weight.clone(),
        "decoder.dec_score_head.0.bias": torch.arange(80, dtype=torch.float32),
        "decoder.denoising_class_embed.weight": torch.arange(
            81 * 4, dtype=torch.float32
        ).reshape(81, 4),
        "backbone.stub": torch.ones(1),
    }


class CheckpointTest(unittest.TestCase):
    def test_initialize_and_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pth"
            initialized = root / "initialized.pth"
            bridge = root / "bridge.pth"
            torch.save({"model": state(), "ema": {"module": state()}}, source)
            initialize_pretrain_checkpoint(source, initialized, seed=2026)
            payload = torch.load(initialized, map_location="cpu", weights_only=False)
            self.assertEqual(payload["ecdeim_semantic_head"]["class_names"], PRETRAIN_CLASSES)
            head = payload["model"]["decoder.enc_score_head.weight"]
            self.assertEqual(tuple(head.shape), (12, 4))
            self.assertTrue(torch.equal(head[0], state()["decoder.enc_score_head.weight"][2]))
            background = payload["model"]["decoder.denoising_class_embed.weight"][-1].clone()

            bridge_pretrain_checkpoint(initialized, bridge)
            bridged = torch.load(bridge, map_location="cpu", weights_only=False)
            self.assertEqual(bridged["ecdeim_bridge"]["class_names"], TRACK6_CLASSES)
            self.assertEqual(
                tuple(bridged["model"]["decoder.enc_score_head.weight"].shape), (10, 4)
            )
            denoising = bridged["model"]["decoder.denoising_class_embed.weight"]
            self.assertEqual(tuple(denoising.shape), (11, 4))
            self.assertTrue(torch.equal(denoising[-1], background))
            validate_checkpoint_taxonomy(initialized, PRETRAIN_CLASSES)
            validate_checkpoint_taxonomy(bridge, TRACK6_CLASSES)
            with self.assertRaises(ValueError):
                validate_checkpoint_taxonomy(bridge, PRETRAIN_CLASSES)

    def test_bridge_uses_names_not_positions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pth"
            output = root / "output.pth"
            names = list(reversed(PRETRAIN_CLASSES))
            state_dict = {
                "decoder.enc_score_head.bias": torch.arange(12, dtype=torch.float32),
                "decoder.denoising_class_embed.weight": torch.arange(13).reshape(13, 1).float(),
            }
            torch.save(
                {"model": state_dict, "ecdeim_semantic_head": {"class_names": names}}, source
            )
            bridge_pretrain_checkpoint(source, output)
            payload = torch.load(output, map_location="cpu", weights_only=False)
            values = payload["model"]["decoder.enc_score_head.bias"].tolist()
            self.assertEqual(values, [float(names.index(name)) for name in TRACK6_CLASSES])


if __name__ == "__main__":
    unittest.main()
