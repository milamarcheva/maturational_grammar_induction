#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a TSV table with per-nonterminal JSD values from *.jsd_per_lhs.out files."
        )
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Directory with *.jsd_per_lhs.out files "
            "(default: try ./staged_eval_outputs_detailed_jsd_per_lhs or Desktop sibling)."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output TSV path (default: <input>/jsd_per_lhs_table.tsv)",
    )
    return parser.parse_args()


def resolve_default_input() -> Path:
    candidates = [
        Path.cwd() / "staged_eval_outputs_detailed_jsd_per_lhs",
        Path(__file__).resolve().parents[2] / "staged_eval_outputs_detailed_jsd_per_lhs",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def parse_base_from_dir(dir_part: str) -> str:
    if not dir_part:
        return "NA"
    match = re.match(r"^stages_min7_([^_]+)", dir_part)
    if match:
        return match.group(1)
    base_part = re.sub(r"^stages_min\d+_", "", dir_part)
    base = base_part.split("_", 1)[0]
    return base or "NA"


def parse_stage_from_name(name: str) -> str:
    tail = name.rsplit("__", 1)[-1]
    match = re.match(r"(\d+)", tail)
    return match.group(1) if match else "NA"


def parse_decimal(value: str) -> str:
    if not value:
        return "NA"
    return value.replace("p", ".")


def to_float(value: str):
    if value == "NA":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_filename(name: str) -> Tuple[str, str, str, str, str, str]:
    if name.endswith(".jsd_per_lhs.out"):
        name = name[: -len(".jsd_per_lhs.out")]
    parts = name.split("__")
    dir_part = parts[0] if parts else "NA"
    base = parse_base_from_dir(dir_part)
    stage = parse_stage_from_name(name)

    params: Dict[str, str] = {}
    for part in parts[1:]:
        if "-" not in part:
            continue
        key, val = part.split("-", 1)
        params[key] = val

    s_p = parse_decimal(params.get("ps"))
    s_l = parse_decimal(params.get("ls"))
    eta = parse_decimal(params.get("nbe"))
    return dir_part, base, stage, s_p, s_l, eta


def parse_jsd_file(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    in_section = False
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.strip() == "JSD per nonterminal":
                in_section = True
                continue
            if not in_section:
                continue
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            data[parts[0]] = parts[1]
    return data


def build_table(input_dir: Path, output_path: Path) -> Tuple[int, int]:
    files = sorted(input_dir.glob("*.jsd_per_lhs.out"), key=lambda p: p.name)
    rows: List[Dict[str, object]] = []
    nonterminals = set()

    for path in files:
        dir_part, base, stage, s_p, s_l, eta = parse_filename(path.name)
        s_l_num = to_float(s_l)
        if s_l_num == 0.0:
            continue
        jsd_values = parse_jsd_file(path)
        nonterminals.update(jsd_values.keys())
        rows.append(
            {
                "dir": dir_part,
                "base": base,
                "stage": stage,
                "s_p": s_p,
                "s_l": s_l,
                "eta": eta,
                "vals": jsd_values,
            }
        )

    nt_list = sorted(nonterminals)

    with output_path.open("w", encoding="utf-8", newline="") as out:
        header = ["dir", "base", "stage", "s_p", "s_l", "eta"] + nt_list
        out.write("\t".join(header) + "\n")
        for row in rows:
            values = [
                row["dir"],
                row["base"],
                row["stage"],
                row["s_p"],
                row["s_l"],
                row["eta"],
            ]
            values.extend(row["vals"].get(nt, "NA") for nt in nt_list)
            out.write("\t".join(values) + "\n")

    return len(rows), len(nt_list)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input) if args.input else resolve_default_input()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")
    output_path = Path(args.output) if args.output else input_dir / "jsd_per_lhs_table.tsv"

    row_count, nt_count = build_table(input_dir, output_path)
    print(f"Wrote {row_count} rows and {nt_count} nonterminal columns to {output_path}")


if __name__ == "__main__":
    main()
