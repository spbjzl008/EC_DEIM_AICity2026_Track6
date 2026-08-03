#!/usr/bin/env python3
"""Build the 12-class public pretraining set from configured COCO sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecdeim.data import build_public_pretraining_set


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--link-mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    args = parser.parse_args()
    summary = build_public_pretraining_set(
        args.config.resolve(), args.output.resolve(), args.seed, args.link_mode
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
