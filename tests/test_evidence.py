from __future__ import annotations

import unittest

import torch
from torchvision.ops import box_area, box_iou, generalized_box_iou

from ecdeim.evidence import EvidenceConfig, evidence_conditioned_mal, split_ignore_regions
from ecdeim.runtime import generalized_box_iou_aligned
from ecdeim.taxonomy import GENERIC_TRUCK_ID, PRETRAIN_CLASSES, TRAILER_ID


class EvidenceTest(unittest.TestCase):
    def test_ignore_regions_are_not_positive_targets(self) -> None:
        target = {
            "boxes": torch.tensor([[0.1, 0.1, 0.2, 0.2], [0.4, 0.4, 0.2, 0.2]]),
            "labels": torch.tensor([7, 0]),
            "area": torch.tensor([0.04, 0.04]),
            "iscrowd": torch.tensor([1, 0]),
            "image_id": torch.tensor([1]),
        }
        clean = split_ignore_regions(target)
        self.assertEqual(clean["labels"].tolist(), [0])
        self.assertEqual(tuple(clean["_ecdeim_ignore_boxes"].shape), (1, 4))
        clean_again = split_ignore_regions(clean)
        self.assertTrue(
            torch.equal(clean_again["_ecdeim_ignore_boxes"], clean["_ecdeim_ignore_boxes"])
        )

    def test_aligned_giou_matches_upstream_pairwise_diagonal(self) -> None:
        boxes1 = torch.tensor([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 4.0]])
        boxes2 = torch.tensor([[0.5, 0.5, 2.5, 2.5], [0.0, 2.0, 4.0, 5.0]])
        expected = torch.diag(generalized_box_iou(boxes1, boxes2))
        self.assertTrue(torch.allclose(generalized_box_iou_aligned(boxes1, boxes2), expected))

    def test_generic_truck_does_not_mask_trailer_negative_supervision(self) -> None:
        class BoxOps:
            box_area = staticmethod(box_area)

            @staticmethod
            def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
                iou = box_iou(boxes1, boxes2)
                area1 = box_area(boxes1)[:, None]
                area2 = box_area(boxes2)[None, :]
                intersection = iou * (area1 + area2) / (1 + iou).clamp(min=1e-8)
                return iou, area1 + area2 - intersection

            @staticmethod
            def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
                center, size = boxes[..., :2], boxes[..., 2:]
                return torch.cat([center - size / 2, center + size / 2], dim=-1)

        class Criterion:
            num_classes = len(PRETRAIN_CLASSES)
            gamma = 1.5
            mal_alpha = 0.75
            _ecdeim_box_ops = BoxOps
            _ecdeim_evidence_config = EvidenceConfig()

            @staticmethod
            def _get_src_permutation_idx(indices):
                return torch.tensor([0]), torch.tensor([0])

        target = {
            "labels": torch.tensor([GENERIC_TRUCK_ID]),
            "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]]),
        }
        indices = [(torch.tensor([0]), torch.tensor([0]))]

        def loss(trailer_logit: float) -> torch.Tensor:
            logits = torch.zeros((1, 1, len(PRETRAIN_CLASSES)))
            logits[0, 0, TRAILER_ID] = trailer_logit
            outputs = {
                "pred_logits": logits,
                "pred_boxes": torch.tensor([[[0.5, 0.5, 0.4, 0.4]]]),
            }
            return evidence_conditioned_mal(
                Criterion(), outputs, [target], indices, 1.0, values=torch.ones(1)
            )["loss_mal"]

        self.assertGreater(float(loss(5.0)), float(loss(0.0)))


if __name__ == "__main__":
    unittest.main()
