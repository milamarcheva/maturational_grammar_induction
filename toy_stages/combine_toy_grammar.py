#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Optional, Tuple


RawRule = Tuple[str, List[str], float, Optional[float]]
Rule = Tuple[str, List[str], float, float]


def parse_rules(path: Path) -> List[RawRule]:
    rules: List[RawRule] = []
    if not path.exists():
        return rules
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if "-->" in parts:
            arrow = parts.index("-->")
        elif "->" in parts:
            arrow = parts.index("->")
        else:
            continue
        if arrow < 2 or arrow >= len(parts) - 1:
            continue
        lhs = parts[arrow - 1]
        rhs = parts[arrow + 1 :]
        weight, bias = parse_prefix_numbers(parts[: arrow - 1])
        rules.append((lhs, rhs, weight, bias))
    return rules


def parse_prefix_numbers(prefix: List[str]) -> Tuple[float, Optional[float]]:
    nums: List[float] = []
    for tok in prefix:
        try:
            nums.append(float(tok))
        except ValueError:
            return 1.0, None
    if not nums:
        return 1.0, None
    weight = nums[0]
    bias = nums[1] if len(nums) >= 2 else None
    return weight, bias


def format_float(value: float) -> str:
    return f"{value:.12g}"


def format_rule_line(lhs: str, rhs: List[str], weight: float, bias: float) -> str:
    return f"{format_float(weight)} {format_float(bias)} {lhs} --> {' '.join(rhs)}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Combine VP results with CP rules, only for non-lexical rules."
    )
    ap.add_argument("--vp", default="res_VP.lt", help="VP results file.")
    ap.add_argument("--cp", default="CP_toy.txt", help="CP rules file.")
    ap.add_argument(
        "--out", default="combined_VP_CP.lt", help="Output grammar file."
    )
    ap.add_argument(
        "--new-weight",
        type=float,
        default=1.0,
        help="Weight for new CP rules (default: 1.0).",
    )
    ap.add_argument(
        "--new-bias",
        type=float,
        default=0.1,
        help="Bias for new CP rules (default: 0.1).",
    )
    ap.add_argument(
        "--vp-bias",
        type=float,
        default=0.1,
        help="Bias for VP rules that lack a bias (default: 0.1).",
    )
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    vp_path = (base / args.vp).resolve()
    cp_path = (base / args.cp).resolve()
    out_path = (base / args.out).resolve()

    vp_raw = parse_rules(vp_path)
    cp_raw = parse_rules(cp_path)

    nonterminals = {lhs for lhs, _, _, _ in vp_raw + cp_raw}

    def is_lex(rule: RawRule) -> bool:
        lhs, rhs, _, _ = rule
        return len(rhs) == 1 and rhs[0] not in nonterminals

    vp_rules: List[Rule] = []
    vp_ids = set()
    for lhs, rhs, weight, bias in vp_raw:
        vp_ids.add((lhs, tuple(rhs)))
        vp_rules.append((lhs, rhs, weight, bias if bias is not None else args.vp_bias))

    new_cp_rules: List[Rule] = []
    for lhs, rhs, _, _ in cp_raw:
        rid = (lhs, tuple(rhs))
        if rid in vp_ids:
            continue
        if is_lex((lhs, rhs, 1.0, None)):
            continue
        new_cp_rules.append((lhs, rhs, args.new_weight, args.new_bias))

    nonlex = [r for r in vp_rules if not is_lex((r[0], r[1], r[2], r[3]))]
    nonlex.extend(new_cp_rules)
    lex = [r for r in vp_rules if is_lex((r[0], r[1], r[2], r[3]))]

    out_path.write_text(
        "\n".join(
            format_rule_line(lhs, rhs, weight, bias)
            for lhs, rhs, weight, bias in (nonlex + lex)
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
