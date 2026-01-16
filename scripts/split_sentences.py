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
        "--parses",
        help="Optional parse file aligned to --input (one parse per sentence).",
    )
    ap.add_argument(
        "--parses-train-out",
        help="Output path for train parses (required if --parses is set).",
    )
    ap.add_argument(
        "--parses-test-out",
        help="Output path for test parses (required if --parses is set).",
    )
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
    parses = None
    if args.parses:
        if not args.parses_train_out or not args.parses_test_out:
            raise SystemExit(
                "--parses-train-out and --parses-test-out are required when --parses is set."
            )
        parses = Path(args.parses).read_text(encoding="utf-8").splitlines()
        if len(parses) != len(lines):
            raise SystemExit(
                f"Parses ({len(parses)}) and sentences ({len(lines)}) counts differ."
            )

    rng = random.Random(args.seed)
    if parses is None:
        rng.shuffle(lines)
    else:
        paired = list(zip(lines, parses))
        rng.shuffle(paired)
        lines, parses = zip(*paired) if paired else ([], [])
        lines = list(lines)
        parses = list(parses)

    split_idx = int(len(lines) * (1.0 - args.test_ratio))
    train = lines[:split_idx]
    test = lines[split_idx:]

    Path(args.train_out).write_text("\n".join(train) + "\n", encoding="utf-8")
    Path(args.test_out).write_text("\n".join(test) + "\n", encoding="utf-8")
    if parses is not None:
        train_parses = parses[:split_idx]
        test_parses = parses[split_idx:]
        Path(args.parses_train_out).write_text(
            "\n".join(train_parses) + "\n", encoding="utf-8"
        )
        Path(args.parses_test_out).write_text(
            "\n".join(test_parses) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
