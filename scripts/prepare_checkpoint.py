#!/usr/bin/env python3
"""Initialize the public head or bridge a public checkpoint to Track 6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecdeim.checkpoints import bridge_pretrain_checkpoint, initialize_pretrain_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser(
        "initialize", help="Convert an 80-class COCO head to 12 classes."
    )
    initialize.add_argument("--input", type=Path, required=True)
    initialize.add_argument("--output", type=Path, required=True)
    initialize.add_argument("--seed", type=int, default=2026)
    bridge = subparsers.add_parser("bridge", help="Convert a trained 12-class head to 10 classes.")
    bridge.add_argument("--input", type=Path, required=True)
    bridge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "initialize":
        result = initialize_pretrain_checkpoint(
            args.input.resolve(), args.output.resolve(), args.seed
        )
    else:
        result = bridge_pretrain_checkpoint(args.input.resolve(), args.output.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
