from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from ecdeim.adaptation import (
    LoRALinear,
    calibrate_class_heads,
    freeze_backbone,
    inject_decoder_lora,
)


class CrossAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention_weights = nn.Linear(4, 4)
        self.sampling_offsets = nn.Linear(4, 8)


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cross_attn = CrossAttention()
        self.enc_score_head = nn.Linear(4, 10)
        self.dec_score_head = nn.ModuleList([nn.Linear(4, 10), nn.Linear(4, 10)])


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.decoder = Decoder()


class AdaptationTest(unittest.TestCase):
    def test_lora_freeze_and_calibration(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Model().to(device)
        matched = inject_decoder_lora(model, rank=2, alpha=4)
        self.assertEqual(len(matched), 2)
        self.assertIsInstance(model.decoder.cross_attn.attention_weights, LoRALinear)
        self.assertEqual(
            model.decoder.cross_attn.attention_weights.lora_A.weight.device,
            model.decoder.cross_attn.attention_weights.base.weight.device,
        )
        frozen = freeze_backbone(model)
        self.assertEqual(
            frozen, sum(parameter.numel() for parameter in model.backbone.parameters())
        )
        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.backbone.parameters())
        )
        before = model.decoder.enc_score_head.bias[2].item()
        touched = calibrate_class_heads(model, single_truck_bias=-2.0)
        self.assertEqual(touched, 3)
        self.assertAlmostEqual(model.decoder.enc_score_head.bias[2].item(), before - 2.0, places=6)
        self.assertEqual(model.decoder.enc_score_head.bias[4].item(), 0.0)
        self.assertEqual(model.decoder.enc_score_head.bias[5].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
