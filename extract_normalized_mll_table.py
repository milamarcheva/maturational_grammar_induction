#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path

MLL_RE = re.compile(
    r"- mean normalized log-likelihood "
    r"\(per token, sentence avg\):\s*([-+]?\d+\.\d+)"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a TSV with file and mean normalized log-likelihood "
            "from *.mll.out files in a directory."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing *.mll.out files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output TSV path. Defaults to <input-dir>/normalized_mll_table.tsv.",
    )
    return parser.parse_args()


def extract_normalized_mll(text: str) -> str:
    match = MLL_RE.search(text)
    return match.group(1) if match else ""


def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def collect_rows(input_dir: Path):
    rows = []
    for path in sorted(input_dir.glob("*.mll.out"), key=lambda p: p.name):
        mll = extract_normalized_mll(read_file(path))
        if not mll:
            print(f"Warning: no normalized MLL found in {path.name}")
        rows.append(
            {
                "file": path.name,
                "mll": mll,
            }
        )
    return rows


def write_tsv(rows, output_path: Path):
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "mll"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    output_path = args.output or (input_dir / "normalized_mll_table.tsv")
    rows = collect_rows(input_dir)
    write_tsv(rows, output_path)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
