# EC-DEIM

**Evidence-Conditioned Multi-Source Pretraining for Cross-City Fine-Grained Object Detection**

Official implementation of our AI City Challenge 2026 Track 6 solution.

This public release was rebuilt from the original competition codebase by removing redundant experimental and debugging paths, consolidating the retained training and inference logic, and repairing release-specific data and compatibility issues.

The released detector uses a single model and one full-image 896 × 896 pass, with no temporal cues, test-time augmentation, model ensemble, SAHI, extra NMS, or test-time update.

[Overview](#overview) · [Installation](#installation) · [Data](#data) · [Training](#training) · [Inference](#inference) · [Acknowledgements](#acknowledgements)

## Overview

EC-DEIM extends [DEIM](https://github.com/Intellindust-AI-Lab/DEIM) with a compact two-stage training pipeline:

```text
80-class DEIM checkpoint
        ↓
12-class public pretraining ── OATS + DEIM native aug + mild IRFS
                              └ evidence routing + full-image model path
        ↓
name-based 12 → 10 checkpoint bridge
        ↓
Track 6 adaptation ── OADC + DEIM native aug + decoder LoRA/head calibration
        ↓
single-pass full-image inference
```

Public pretraining uses seven driving datasets, a 12-class semantic head, Object-Aware Tile Sampling (OATS), evidence-conditioned loss routing, and the retained full-image/mild-IRFS data path. Non-conflicting DEIM-native transforms remain in the same pipeline. Track 6 adaptation uses only Track 6 images and combines Object-Aware Domain Coverage (OADC) with DEIM's native augmentations, decoder LoRA, a frozen backbone, and class-head calibration. The pinned upstream checkout is never edited; compact in-process hooks add the released loss routing, memory-safe aligned GIoU, LoRA, and gradient accumulation.

## Installation

The local reference used Python 3.11.9; the verified Hafnia image uses Python 3.12.13. Both pin DEIM revision `09d35d53d39ee3145a1e61e3a989b28b9468d1dd`.

```bash
git clone https://github.com/Intellindust-AI-Lab/DEIM third_party/DEIM
git -C third_party/DEIM checkout 09d35d53d39ee3145a1e61e3a989b28b9468d1dd

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r third_party/DEIM/requirements.txt
python -m pip install -e .
```

For the Hafnia build, the checked-in `Dockerfile` installs the same pinned DEIM revision, PyTorch 2.10/cu128, and Hafnia 0.7.8:

```bash
docker build -t ec-deim-hafnia .
```

Download `deim_dfine_hgnetv2_x_obj2coco_24e.pth` from the upstream DEIM model zoo and place it in `weights/`. This is the Objects365-to-COCO checkpoint used by the released initialization path. Its released classifier has 80 COCO rows; the initializer does not accept a raw 365-class head.

## Data

### Public pretraining

Download [BDD100K](https://bdd-data.berkeley.edu/), [COCO](https://cocodataset.org/), [UA-DETRAC](http://detrac-db.rit.albany.edu/), [MIO-TCD](http://tcd.miovision.com/), [nuImages](https://www.nuscenes.org/nuimages), [SODA10M](https://soda-2d.github.io/), and [VisDrone2019](https://github.com/VisDrone/VisDrone-Dataset) under their respective terms. Images and annotations are not redistributed. Convert each dataset's training and validation annotations to COCO object-detection format before using this repository; native dataset formats are not converted here.

Configure each converted source with separate image and annotation paths:

```text
SOURCE_ROOT/
├── images/
│   ├── train/...
│   └── val/...
└── annotations/
    ├── train.json
    └── val.json
```

Each JSON file must contain the standard COCO `images`, `annotations`, and `categories` lists. Bounding boxes use absolute-pixel `[x, y, width, height]`; every annotation references an image and category id, and each image `file_name` is relative to the configured image directory. Source category ids may be arbitrary because [`configs/public_sources.yaml`](configs/public_sources.yaml) maps categories by name.

Set the source paths in `configs/public_sources.yaml`, then merge the converted datasets into the shared 12-class COCO dataset:

```bash
python scripts/prepare_public_data.py \
  --config configs/public_sources.yaml \
  --output data/public_pretrain \
  --link-mode symlink
```

The builder remaps category names, writes linked images and unified COCO annotations, and records audit manifests. It refuses to overwrite a non-empty output directory.

<details>
<summary><strong>Public-data label and sampling policy</strong></summary>

- Coarse `Truck` labels map to `Vehicle.Truck_Generic`, providing group and localization evidence without an invented fine subtype.
- Protocol-conflicting `Cyclist` and aerial `bicycle` boxes become `iscrowd=1` ignore regions.
- Exact duplicate images are removed across sources and splits using SHA-256.
- Training splits are scanned without a global source cap; common-only images are then downsampled before mild image- and instance-frequency repeat sampling (mild IRFS/mRFS, target maximum repeat 1.8).

The checked-in source mappings are part of the released protocol. Change paths as needed; review mapping changes carefully.

</details>

### Track 6

Track 6 data uses standard COCO JSON with zero-based category ids. In an official Hafnia job:

```bash
python scripts/export_hafnia_coco.py --output data/track6_coco
```

For a local holdout, split complete camera groups rather than frames from the same camera:

```bash
python scripts/split_by_camera.py \
  --annotations data/track6_coco/train/_annotations.coco.json \
  --output data/track6_camera_split \
  --camera-regex 'camera[_-]?(?P<camera>[A-Za-z0-9]+)'
```

The optional splitter verifies that the camera sets do not overlap.

<details>
<summary><strong>Required Track 6 category order</strong></summary>

```text
0  Vehicle.Car
1  Vehicle.Pickup Truck
2  Vehicle.Single Truck
3  Vehicle.Combo Truck
4  Vehicle.Heavy Duty Vehicle
5  Vehicle.Trailer
6  Vehicle.Motorcycle
7  Vehicle.Bicycle
8  Vehicle.Van
9  Person
```

</details>

## Training

All commands below reproduce the one-GPU schedule. Use `--dry-run` first to check paths, class counts, world size, and effective batch size.

### 1. Public pretraining

Initialize the 12-class semantic head from the official 80-class checkpoint:

```bash
python scripts/prepare_checkpoint.py initialize \
  --input weights/deim_dfine_hgnetv2_x_obj2coco_24e.pth \
  --output weights/ecdeim_12c_init.pth
```

The combined 896 recipe gives the new OATS schedule priority while retaining non-conflicting DEIM-native Mosaic, photometric distortion, ZoomOut, IoU crop, horizontal flip, early MixUp, full-image loading, common-class downsampling, and mild IRFS. Train with micro-batch 5 and six accumulation steps (effective batch 30):

```bash
python scripts/train.py configs/pretrain.yaml \
  --deim-root third_party/DEIM \
  --checkpoint weights/ecdeim_12c_init.pth \
  --train-images data/public_pretrain/train/images \
  --train-annotations data/public_pretrain/annotations/train.json \
  --val-images data/public_pretrain/valid/images \
  --val-annotations data/public_pretrain/annotations/valid.json \
  --output outputs/pretrain \
  --devices 0
```

OATS runs in the original image coordinates before the native transform chain. Evidence routing covers both main and auxiliary loss terms; the data manifest records full-image mode and the effective mild-IRFS factors.

Bridge the selected checkpoint to the ten Track 6 classes:

```bash
python scripts/prepare_checkpoint.py bridge \
  --input outputs/pretrain/best_stg2.pth \
  --output weights/ecdeim_10c_bridge.pth
```

The bridge matches rows by class name and preserves the denoising background embedding.

### 2. Track 6 adaptation

The released schedule uses 16 epochs at 896 × 896, micro-batch 3, and ten accumulation steps (effective batch 30):

```bash
python scripts/train.py configs/adapt.yaml \
  --deim-root third_party/DEIM \
  --checkpoint weights/ecdeim_10c_bridge.pth \
  --train-images data/track6_coco/train \
  --train-annotations data/track6_coco/train/_annotations.coco.json \
  --val-images data/track6_coco/valid \
  --val-annotations data/track6_coco/valid/_annotations.coco.json \
  --output outputs/adapt \
  --devices 0
```

Only the bridged weights cross from public pretraining; public images are not read during adaptation. The adaptation recipe combines OADC, early MixUp, DEIM-native appearance and multiscale augmentation, decoder LoRA, class-head calibration, EMA, and gradient accumulation. Exact schedules and probabilities are defined in [`configs/adapt.yaml`](configs/adapt.yaml).

To resume, replace `--checkpoint ...` with:

```bash
--resume outputs/adapt/last.pth
```

Head calibration runs only when starting a new adaptation run. Using multiple GPUs changes the effective batch unless gradient accumulation is adjusted.

### Hafnia cloud training

Attach the 10-class bridge model and run the shared training path; the entrypoint exports TRAIN+VAL to zero-based COCO, keeps 15% of empty/background images, invokes the same `scripts/train.py`, logs metrics, and copies the best checkpoint to Hafnia's model directory:

```bash
python scripts/train_hafnia.py \
  --deim-root third_party/DEIM \
  --checkpoint /path/to/bridge_track6_10c.pth
```

Outside a cloud job add `--dataset-path /path/to/track6-dataset`.

## Inference

Run the full-image predictor on a COCO image manifest:

```bash
python scripts/infer.py \
  --deim-root third_party/DEIM \
  --config configs/adapt.yaml \
  --checkpoint outputs/adapt/best_stg2.pth \
  --images data/track6_coco/test \
  --annotations data/track6_coco/test/_annotations.coco.json \
  --output outputs/test_detections.json \
  --threshold 0.001
```

The predictor resizes each full image to 896 × 896, performs one forward pass, and retains the upstream postprocessor's top 300 candidates above the score threshold. It adds no NMS or tiled inference.

For an official Hafnia job:

```bash
python scripts/make_hafnia_submission.py \
  --deim-root third_party/DEIM \
  --config configs/adapt.yaml \
  --checkpoint outputs/adapt/best_stg2.pth
```

Hafnia annotations are written directly to the platform artifact directory. For local use, pass `--dataset-path` and `--output`.

## Verification

```bash
python -m unittest discover -s tests -v
python -m compileall -q ecdeim scripts tests
```

The tests cover checkpoint conversion, OATS/OADC, LoRA, ignore regions, deduplication, and category order.

## Acknowledgements

EC-DEIM builds on [DEIM](https://github.com/Intellindust-AI-Lab/DEIM) and [D-FINE](https://github.com/Peterande/D-FINE). Please cite these projects when using their implementation or checkpoints.

## License

The EC-DEIM additions are released under the [Apache License 2.0](LICENSE). Upstream code, datasets, pretrained checkpoints, and Hafnia remain subject to their own licenses and terms.
