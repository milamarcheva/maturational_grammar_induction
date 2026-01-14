#!/usr/bin/env python3
"""
Batch extract grammars from filtered CTB parses with multiple weight modes and min-freqs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


def normalize_weight_modes(modes: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    for mode in modes:
        if mode == "percentages":
            mode = "percentage"
        normalized.append(mode)
    return normalized


def build_output_name(input_path: Path, weight_mode: str, min_freq: int) -> str:
    stem = input_path.stem
    if stem.endswith("_parses"):
        base = stem.replace("_parses", "_grammar_withUnary")
    else:
        base = f"{stem}_grammar_withUnary"
    mode_label = "counts" if weight_mode == "counts" else "percentage"
    return f"{base}__{mode_label}_min{min_freq}{input_path.suffix}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract grammars over multiple weight modes and min-freq values."
    )
    ap.add_argument(
        "--input",
        default="/Users/milamarcheva/Desktop/morphemic_tokenisation/data/filtered_ctb/filtered_ctb_parses.txt",
        help="Input parse file (one tree per line).",
    )
    ap.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent),
        help="Directory to write extracted grammars.",
    )
    ap.add_argument(
        "--weight-modes",
        nargs="+",
        default=["counts", "percentage"],
        help="Weight modes to use (e.g., counts percentage).",
    )
    ap.add_argument(
        "--min-freqs",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5, 10],
        help="Minimum frequency values to use.",
    )
    ap.add_argument(
        "--extract-script",
        default=str(Path(__file__).resolve().parents[2] / "morphemic_tokenisation/extract_grammar_from_tagged.py"),
        help="Path to extract_grammar_from_tagged.py.",
    )
    args = ap.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    extract_script = Path(args.extract_script)

    weight_modes = normalize_weight_modes(args.weight_modes)
    for weight_mode in weight_modes:
        for min_freq in args.min_freqs:
            out_name = build_output_name(input_path, weight_mode, min_freq)
            out_path = output_dir / out_name
            cmd = [
                sys.executable,
                str(extract_script),
                "--input",
                str(input_path),
                "--output",
                str(out_path),
                "--weight-mode",
                weight_mode,
                "--remove-self-unary",
                "--min-freq",
                str(min_freq),
            ]
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
