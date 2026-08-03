#!/usr/bin/env python3
"""Run EC-DEIM on a Hafnia split and write evaluator-ready annotations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from ecdeim.inference import Predictor
from ecdeim.taxonomy import TRACK6_CLASSES


def main() -> None:
    try:
        from hafnia import utils as hafnia_utils
        from hafnia.dataset.benchmark.benchmark import run_inference_on_dataset
        from hafnia.dataset.benchmark.inference_model import InferenceModel
        from hafnia.dataset.dataset_names import SampleField, SplitName
        from hafnia.dataset.hafnia_dataset import HafniaDataset
        from hafnia.dataset.hafnia_dataset_types import ClassInfo, ModelInfo, TaskInfo
        from hafnia.dataset.primitives import Bbox
        from hafnia.experiment import HafniaLogger
    except ImportError as error:
        raise RuntimeError("Run this script inside the official Hafnia environment.") from error

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deim-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/adapt.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--split", default=SplitName.TEST)
    parser.add_argument("--output", type=Path, default=Path("outputs/hafnia_submission"))
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--image-size", type=int, default=896)
    parser.add_argument("--samples", type=int)
    args = parser.parse_args()

    class HafniaECDEIM(InferenceModel):
        def __init__(self) -> None:
            self.predictor = Predictor(
                args.deim_root,
                args.config,
                args.checkpoint,
                args.threshold,
                args.image_size,
            )
            class_info = [ClassInfo(name=name, attributes=None) for name in TRACK6_CLASSES]
            self.task = TaskInfo(name="object_detection", primitive=Bbox, classes=class_info)

        def get_model_info(self) -> Any:
            return ModelInfo(name="EC-DEIM", tasks=[self.task])

        def optimize_for_inference(self) -> None:
            self.predictor.model.eval()

        def predict(self, images: Any, sample_dict: dict[str, Any] | None = None) -> list[Any]:
            values = images if isinstance(images, list) else [images]
            output = []
            for value in values:
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()
                if isinstance(value, np.ndarray):
                    if value.ndim == 3 and value.shape[0] in {1, 3, 4}:
                        value = np.moveaxis(value, 0, -1)
                    if value.ndim == 3 and value.shape[-1] == 1:
                        value = np.repeat(value, 3, axis=-1)
                    value = Image.fromarray(value.astype(np.uint8))
                if not isinstance(value, Image.Image):
                    raise TypeError(f"Unsupported image type: {type(value).__name__}")
                width, height = value.size
                for prediction in self.predictor.predict(value):
                    x, y, box_width, box_height = prediction["bbox"]
                    output.append(
                        Bbox(
                            height=box_height / height,
                            width=box_width / width,
                            top_left_x=x / width,
                            top_left_y=y / height,
                            class_idx=prediction["category_id"],
                            class_name=prediction["category_name"],
                            confidence=prediction["score"],
                            ground_truth=False,
                        )
                    )
            return output

    logger = HafniaLogger(project_name="ec-deim")
    if hafnia_utils.is_hafnia_cloud_job():
        dataset = HafniaDataset.from_path(hafnia_utils.get_dataset_path_in_hafnia_cloud())
        output = Path(logger._path_artifacts())
    elif args.dataset_path:
        dataset = HafniaDataset.from_path(args.dataset_path)
        output = args.output
    else:
        raise ValueError("--dataset-path is required outside Hafnia cloud jobs.")
    split = dataset.create_split_dataset(split_name=args.split)
    if args.samples is not None:
        split = split.select_samples(n_samples=args.samples)
    if len(split) == 0:
        raise RuntimeError(f"The requested split is empty: {args.split}")
    model = HafniaECDEIM()
    model.optimize_for_inference()
    predictions = run_inference_on_dataset(dataset=split, model=model)
    predictions.samples = predictions.samples.drop(
        [SampleField.FILE_PATH, SampleField.VIDEO_INFO, SampleField.CAMERA_INFO, SampleField.META],
        strict=False,
    )
    output.mkdir(parents=True, exist_ok=True)
    predictions.write_annotations(output)
    logger.end_run()
    print(f"Wrote Hafnia annotations to {output}")


if __name__ == "__main__":
    main()
