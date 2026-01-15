#!/usr/bin/env python3
"""
Binary search for the smallest prefix of a yield file that causes io to fail.

Assumes:
- a lower bound that succeeds
- an upper bound that fails

Example:
  python3 milas_scripts/find_io_fail.py \
    --io ./io \
    --grammar milas_grammars/productions_nva_12_pb0p1_lb0p1_primed.lt \
    --yields milas_grammars/data/engall_morphtok_sents.txt \
    --low 250000 \
    --high 450000 \
    --io-args "-d 0 -m 1 -n 1 -p 0 -W 1"
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import Sequence


def write_prefix(src: Path, dst: Path, nlines: int, start: int = 1) -> None:
    """Write nlines from src to dst, starting at line `start` (1-based, inclusive)."""
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin, 1):
            if i < start:
                continue
            if i >= start + nlines:
                break
            fout.write(line)


def run_io(io_bin: Path, grammar: Path, yields: Path, io_args: Sequence[str]) -> int:
    cmd = [str(io_bin)] + list(io_args) + ["-g", str(grammar), str(yields)]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc.returncode


def main() -> None:
    ap = argparse.ArgumentParser(description="Binary search for failing prefix of yield file for io.")
    ap.add_argument("--io", default="./io", help="Path to io binary (default: ./io)")
    ap.add_argument("--grammar", required=True, help="Path to grammar file")
    ap.add_argument("--yields", required=True, help="Path to full yields file")
    ap.add_argument("--low", type=int, required=True, help="Known good prefix length")
    ap.add_argument("--high", type=int, required=True, help="Known failing prefix length")
    ap.add_argument(
        "--tmp", default="/tmp/io_prefix.txt", help="Temp file to store prefixes (default: /tmp/io_prefix.txt)"
    )
    ap.add_argument(
        "--io-args",
        default="-d 0 -m 1 -n 1 -p 0 -W 1",
        help="Extra args for io (default: '-d 0 -m 1 -n 1 -p 0 -W 1')",
    )
    ap.add_argument(
        "--start",
        type=int,
        default=1,
        help="Start line (1-based) for prefixes (default: 1; set to 1000 to skip first 999 lines).",
    )
    args = ap.parse_args()

    io_bin = Path(args.io).resolve()
    grammar = Path(args.grammar).resolve()
    yields_path = Path(args.yields).resolve()
    tmp_path = Path(args.tmp)
    io_args = shlex.split(args.io_args)

    low = args.low
    high = args.high

    print(f"Using io: {io_bin}")
    print(f"Grammar: {grammar}")
    print(f"Yields: {yields_path}")
    print(f"Known good (low): {low}")
    print(f"Known fail (high): {high}")

    # sanity checks
    write_prefix(yields_path, tmp_path, low, start=args.start)
    rc_low = run_io(io_bin, grammar, tmp_path, io_args)
    if rc_low != 0:
        raise SystemExit(f"Low bound {low} failed with rc={rc_low}; need a passing low bound.")

    write_prefix(yields_path, tmp_path, high, start=args.start)
    rc_high = run_io(io_bin, grammar, tmp_path, io_args)
    if rc_high == 0:
        raise SystemExit(f"High bound {high} succeeded; need a failing high bound.")

    while low + 1 < high:
        mid = (low + high) // 2
        print(f"Testing {mid}...", flush=True)
        write_prefix(yields_path, tmp_path, mid, start=args.start)
        rc = run_io(io_bin, grammar, tmp_path, io_args)
        if rc == 0:
            print(f"OK {mid}")
            low = mid
        else:
            print(f"FAIL {mid}")
            high = mid

    print(f"First failing prefix length likely at {high}")


if __name__ == "__main__":
    main()
