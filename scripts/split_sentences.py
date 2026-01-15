#!/usr/bin/env python3
import argparse
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Shuffle and split a sentence file into train/test."
    )
    ap.add_argument("--input", required=True, help="Input sentence file.")
    ap.add_argument("--train-out", required=True, help="Output path for train split.")
    ap.add_argument("--test-out", required=True, help="Output path for test split.")
    ap.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of lines to put in test (default: 0.1).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffling.",
    )
    args = ap.parse_args()

    lines = Path(args.input).read_text(encoding="utf-8").splitlines()
    random.Random(args.seed).shuffle(lines)

    split_idx = int(len(lines) * (1.0 - args.test_ratio))
    train = lines[:split_idx]
    test = lines[split_idx:]

    Path(args.train_out).write_text("\n".join(train) + "\n", encoding="utf-8")
    Path(args.test_out).write_text("\n".join(test) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
