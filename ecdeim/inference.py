"""Single-pass, full-image EC-DEIM inference."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
from PIL import Image
import torch
import yaml

from .adaptation import infer_lora_spec, inject_decoder_lora
from .checkpoints import model_state
from .taxonomy import TRACK6_CLASSES


class Predictor:
    """Load one 10-class checkpoint and retain upstream DEIM's top-300 outputs."""

    def __init__(
        self,
        deim_root: Path,
        config_path: Path,
        checkpoint_path: Path,
        threshold: float = 0.001,
        image_size: int = 896,
        device: str | None = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1].")
        if image_size < 1:
            raise ValueError("image_size must be positive.")
        deim_root = deim_root.resolve()
        if not (deim_root / "engine" / "__init__.py").is_file():
            raise FileNotFoundError(f"DEIM engine not found at {deim_root}")
        sys.path.insert(0, str(deim_root))
        from .runtime import install_optional_calflops_fallback, install_torchvision_compatibility

        install_optional_calflops_fallback()
        install_torchvision_compatibility()
        from engine.core import YAMLConfig

        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        temporary_path: Path | None = None
        if isinstance(raw_config, dict) and raw_config.get("stage") == "adapt":
            base_config = deim_root / raw_config["model"]["base_config"]
            portable = {
                "__include__": [
                    str(base_config.resolve()),
                    str((deim_root / "configs/base/deim.yml").resolve()),
                ],
                "num_classes": len(TRACK6_CLASSES),
                "remap_mscoco_category": False,
                "eval_spatial_size": [int(image_size), int(image_size)],
                "HGNetv2": {"pretrained": False},
            }
            temporary = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False)
            with temporary:
                yaml.safe_dump(portable, temporary, sort_keys=False)
            temporary_path = Path(temporary.name)
            model_config_path = temporary_path
        else:
            model_config_path = config_path.resolve()
        try:
            config = YAMLConfig(str(model_config_path))
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        model = config.model
        postprocessor = config.postprocessor
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError("Checkpoint must be a dictionary.")
        state = model_state(payload)
        lora = infer_lora_spec(state)
        if lora is not None:
            metadata = payload.get("ecdeim_training", {})
            lora_metadata = metadata.get("lora", {}) if isinstance(metadata, dict) else {}
            rank = int(lora_metadata.get("rank", lora[0]))
            alpha = float(lora_metadata.get("alpha", lora[1]))
            inject_decoder_lora(model, rank=rank, alpha=alpha)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Checkpoint and model configuration disagree. "
                f"Missing={missing[:10]}, unexpected={unexpected[:10]}"
            )
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device).eval()
        self.postprocessor = postprocessor.to(self.device).eval()
        self.threshold = float(threshold)
        self.image_size = int(image_size)

    def _prepare(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        image = image.convert("RGB")
        width, height = image.size
        resized = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1).div_(255.0).unsqueeze(0)
        original_size = torch.tensor([[width, height]], dtype=torch.float32)
        return tensor.to(self.device), original_size.to(self.device)

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> list[dict[str, Any]]:
        sample, original_size = self._prepare(image)
        result = self.postprocessor(self.model(sample), original_size)[0]
        image_width, image_height = (float(value) for value in original_size[0])
        predictions: list[dict[str, Any]] = []
        for label, box, score in zip(result["labels"], result["boxes"], result["scores"]):
            confidence = float(score)
            class_id = int(label)
            if confidence < self.threshold or not 0 <= class_id < len(TRACK6_CLASSES):
                continue
            x1, y1, x2, y2 = (float(value) for value in box)
            x1, x2 = sorted(
                (min(image_width, max(0.0, x1)), min(image_width, max(0.0, x2)))
            )
            y1, y2 = sorted(
                (min(image_height, max(0.0, y1)), min(image_height, max(0.0, y2)))
            )
            if x2 <= x1 or y2 <= y1:
                continue
            predictions.append(
                {
                    "category_id": class_id,
                    "category_name": TRACK6_CLASSES[class_id],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": confidence,
                }
            )
        return predictions
