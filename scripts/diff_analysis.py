#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot distribution of rule probabilities for a nonterminal and run a paired t-test."
        )
    )
    parser.add_argument("csv_path", help="CSV from compare_pcfg_probs.py")
    parser.add_argument("nonterminal", help="LHS nonterminal to filter (e.g., NP)")
    parser.add_argument(
        "--column",
        required=True,
        help=(
            "Which probability column to plot: 1 or 2, or the column name from the CSV header."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output plot path (default: diff_analysis_<NONTERMINAL>.png)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins (default: 50)",
    )
    return parser.parse_args()


def resolve_column(header: List[str], column_arg: str) -> int:
    if column_arg.isdigit():
        idx = int(column_arg)
        if idx not in (1, 2):
            raise ValueError("--column must be 1 or 2 when using a numeric index")
        return idx
    if column_arg in header:
        return header.index(column_arg)
    raise ValueError(
        f"--column must be 1 or 2 or one of the header names: {', '.join(header)}"
    )


def parse_rule_lhs(rule: str) -> str:
    if "-->" not in rule:
        return ""
    lhs, _ = rule.split("-->", 1)
    return lhs.strip()


def safe_float(val: str):
    val = val.strip()
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def load_rows(
    csv_path: Path, nonterminal: str
) -> Tuple[List[float], List[float]]:
    col1_vals: List[float] = []
    col2_vals: List[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if len(header) < 3:
            raise ValueError("CSV must have at least three columns: rule, file1, file2")
        for row in reader:
            if not row:
                continue
            rule = row[0]
            if parse_rule_lhs(rule) != nonterminal:
                continue
            v1 = safe_float(row[1]) if len(row) > 1 else None
            v2 = safe_float(row[2]) if len(row) > 2 else None
            if v1 is not None:
                col1_vals.append(v1)
            if v2 is not None:
                col2_vals.append(v2)
    return col1_vals, col2_vals


def load_pairs(csv_path: Path, nonterminal: str) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if len(header) < 3:
            raise ValueError("CSV must have at least three columns: rule, file1, file2")
        for row in reader:
            if not row:
                continue
            rule = row[0]
            if parse_rule_lhs(rule) != nonterminal:
                continue
            v1 = safe_float(row[1]) if len(row) > 1 else None
            v2 = safe_float(row[2]) if len(row) > 2 else None
            if v1 is None or v2 is None:
                continue
            pairs.append((v1, v2))
    return pairs


def paired_t_test(pairs: List[Tuple[float, float]]) -> Tuple[float, float, int]:
    if len(pairs) < 2:
        raise ValueError("Need at least two paired observations for a t-test")
    diffs = [b - a for a, b in pairs]
    mean_diff = sum(diffs) / len(diffs)
    if len(diffs) == 1:
        sd = 0.0
    else:
        var = sum((d - mean_diff) ** 2 for d in diffs) / (len(diffs) - 1)
        sd = math.sqrt(var)
    if sd == 0.0:
        t_stat = float("inf") if mean_diff != 0 else 0.0
    else:
        t_stat = mean_diff / (sd / math.sqrt(len(diffs)))
    return t_stat, mean_diff, len(diffs) - 1


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

    col_idx = resolve_column(header, args.column)

    col1_vals, col2_vals = load_rows(csv_path, args.nonterminal)
    vals = col1_vals if col_idx == 1 else col2_vals

    if not vals:
        raise SystemExit("No values found for that nonterminal/column combination.")

    out_path = (
        Path(args.out)
        if args.out
        else Path(f"diff_analysis_{args.nonterminal}.png")
    )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.hist(vals, bins=args.bins, color="#3B6EA5", edgecolor="#1E2A38")
        plt.title(f"{args.nonterminal} probability distribution ({header[col_idx]})")
        plt.xlabel("Probability")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting; install it or run in an environment where it is available."
        ) from exc

    pairs = load_pairs(csv_path, args.nonterminal)
    if len(pairs) < 2:
        print("Paired t-test: not enough paired rows (need at least 2).")
        print(f"Pairs found: {len(pairs)}")
        return

    try:
        from scipy import stats  # type: ignore

        v1 = [p[0] for p in pairs]
        v2 = [p[1] for p in pairs]
        t_stat, p_val = stats.ttest_rel(v2, v1, nan_policy="omit")
        print(f"Paired t-test (file2 - file1): t={t_stat:.6g}, p={p_val:.6g}, n={len(pairs)}")
    except ImportError:
        t_stat, mean_diff, df = paired_t_test(pairs)
        print(
            "SciPy not available; computed t-statistic only. "
            f"t={t_stat:.6g}, df={df}, mean_diff={mean_diff:.6g}"
        )

    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
