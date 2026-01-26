#!/usr/bin/env python3
import argparse
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


Rule = Tuple[str, List[str], str]


def parse_rules(path: Path) -> List[Rule]:
    rules: List[Rule] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if "-->" not in parts and "->" not in parts:
                continue
            if "-->" in parts:
                arrow = parts.index("-->")
            else:
                arrow = parts.index("->")
            if arrow == 0 or arrow >= len(parts) - 1:
                continue
            lhs = parts[arrow - 1]
            rhs = parts[arrow + 1 :]
            rules.append((lhs, rhs, s))
    return rules


def parse_rules_from_lines(lines: Iterable[str]) -> List[Rule]:
    rules: List[Rule] = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if "-->" not in parts and "->" not in parts:
            continue
        if "-->" in parts:
            arrow = parts.index("-->")
        else:
            arrow = parts.index("->")
        if arrow == 0 or arrow >= len(parts) - 1:
            continue
        lhs = parts[arrow - 1]
        rhs = parts[arrow + 1 :]
        rules.append((lhs, rhs, s))
    return rules


def rule_id(rule: Rule) -> Tuple[str, Tuple[str, ...]]:
    lhs, rhs, _ = rule
    return lhs, tuple(rhs)


def avg_weights_by_lhs(rules: Iterable[Rule]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for lhs, _, raw in rules:
        weight = extract_rule_weight(raw)
        totals[lhs] = totals.get(lhs, 0.0) + weight
        counts[lhs] = counts.get(lhs, 0) + 1
    return {
        lhs: (totals[lhs] / counts[lhs]) for lhs in totals if counts[lhs] > 0
    }


def extract_rule_weight(raw: str) -> float:
    parts = raw.split()
    if "-->" in parts:
        arrow = parts.index("-->")
    elif "->" in parts:
        arrow = parts.index("->")
    else:
        return 1.0
    if arrow <= 0:
        return 1.0
    prefix = parts[: arrow - 1]
    nums: List[float] = []
    for tok in prefix:
        try:
            nums.append(float(tok))
        except ValueError:
            return 1.0
    if not nums:
        return 1.0
    return nums[0]


def replace_rule_weight(raw: str, new_weight: float) -> str:
    parts = raw.split()
    if "-->" in parts:
        arrow = parts.index("-->")
    elif "->" in parts:
        arrow = parts.index("->")
    else:
        return raw
    if arrow <= 0 or arrow >= len(parts) - 1:
        return raw
    lhs = parts[arrow - 1]
    rhs = parts[arrow + 1 :]
    prefix = parts[: arrow - 1]

    nums: List[float] = []
    for tok in prefix:
        try:
            nums.append(float(tok))
        except ValueError:
            nums = []
            break

    if len(nums) >= 2:
        return f"{new_weight} {nums[1]} {lhs} --> {' '.join(rhs)}"
    return f"{new_weight} {lhs} --> {' '.join(rhs)}"


def dedupe_rules(rules: Iterable[Rule]) -> List[Rule]:
    seen = set()
    out: List[Rule] = []
    for rule in rules:
        rid = rule_id(rule)
        if rid in seen:
            continue
        seen.add(rid)
        out.append(rule)
    return out


def split_lexicon(
    rules: List[Rule],
) -> List[Rule]:
    if not rules:
        raise SystemExit("No rules found in full grammar.")
    nonterminals = {lhs for lhs, _, _ in rules}
    lex_rules = []
    for lhs, rhs, raw in rules:
        if len(rhs) == 1 and rhs[0] not in nonterminals:
            lex_rules.append((lhs, rhs, raw))
    return lex_rules


def collect_terminals(lex_rules: List[Rule]) -> set[str]:
    terminals = set()
    for _, rhs, _ in lex_rules:
        if rhs:
            terminals.add(rhs[0])
    return terminals


def merge_stage(
    stage_rules: List[Rule],
    prev_rules: Optional[List[Rule]],
    lex_rules: List[Rule],
) -> List[str]:
    seen = set()
    lines: List[str] = []
    base_rules = stage_rules + lex_rules if prev_rules is None else prev_rules + stage_rules
    for rule in base_rules:
        rid = rule_id(rule)
        if rid in seen:
            continue
        seen.add(rid)
        lines.append(rule[2])
    return lines


def prune_rules_missing_rhs(
    rules: List[Rule],
    terminals: set[str],
) -> Tuple[List[Rule], List[str], int]:
    kept = list(rules)
    dropped_total = 0
    missing_syms: set[str] = set()

    while True:
        lhs_set = {lhs for lhs, _, _ in kept}
        new_kept: List[Rule] = []
        dropped = 0
        missing_round = set()
        for lhs, rhs, raw in kept:
            if len(rhs) >= 2:
                bad = [sym for sym in rhs if sym not in lhs_set]
                if bad:
                    missing_round.update(bad)
                    dropped += 1
                    continue
                new_kept.append((lhs, rhs, raw))
                continue
            if len(rhs) == 1:
                sym = rhs[0]
                if sym in lhs_set or sym in terminals:
                    new_kept.append((lhs, rhs, raw))
                else:
                    missing_round.add(sym)
                    dropped += 1
                continue
            new_kept.append((lhs, rhs, raw))
        if dropped == 0:
            break
        dropped_total += dropped
        missing_syms.update(missing_round)
        kept = new_kept

    return kept, sorted(missing_syms), dropped_total


def order_rules_nonlex_then_lex(rules: List[Rule]) -> List[Rule]:
    lhs_set = {lhs for lhs, _, _ in rules}
    nonlex: List[Rule] = []
    lex: List[Rule] = []
    for lhs, rhs, raw in rules:
        if len(rhs) == 1 and rhs[0] not in lhs_set:
            lex.append((lhs, rhs, raw))
        else:
            nonlex.append((lhs, rhs, raw))
    return nonlex + lex


def normalize_rule_line(raw: str, default_bias: float) -> str:
    parts = raw.split()
    if "-->" in parts:
        arrow = parts.index("-->")
    elif "->" in parts:
        arrow = parts.index("->")
    else:
        return raw
    if arrow <= 0 or arrow >= len(parts) - 1:
        return raw
    lhs = parts[arrow - 1]
    rhs = parts[arrow + 1 :]
    prefix = parts[: arrow - 1]

    nums: List[float] = []
    for tok in prefix:
        try:
            nums.append(float(tok))
        except ValueError:
            nums = []
            break

    if len(nums) >= 2:
        return raw
    if len(nums) == 1:
        return f"{nums[0]} {default_bias} {lhs} --> {' '.join(rhs)}"
    return f"1.0 {default_bias} {lhs} --> {' '.join(rhs)}"


def apply_avg_weights_for_new_rules(
    init_rules: List[Rule],
    prev_rules: List[Rule],
    lex_rule_ids: Optional[set[Tuple[str, Tuple[str, ...]]]] = None,
    new_rule_weight: Optional[float] = None,
) -> List[str]:
    prev_ids = {rule_id(r) for r in prev_rules}
    avg_weights = avg_weights_by_lhs(prev_rules)
    adjusted: List[str] = []
    for lhs, rhs, raw in init_rules:
        rid = rule_id((lhs, rhs, raw))
        if rid not in prev_ids:
            if lex_rule_ids and rid in lex_rule_ids:
                adjusted.append(raw)
                continue
            if new_rule_weight is not None:
                raw = replace_rule_weight(raw, new_rule_weight)
            else:
                raw = replace_rule_weight(raw, avg_weights.get(lhs, 1.0))
        adjusted.append(raw)
    return adjusted


def collect_lex_terms(rules: List[Rule]) -> set[str]:
    nonterminals = {lhs for lhs, _, _ in rules}
    lex_terms = set()
    for _, rhs, _ in rules:
        if len(rhs) == 1 and rhs[0] not in nonterminals:
            lex_terms.add(rhs[0])
    return lex_terms


def read_yield_token_counts(yields_path: Path) -> Counter:
    counts: Counter = Counter()
    with yields_path.open(encoding="utf-8") as f:
        for line in f:
            for tok in line.strip().split():
                if tok:
                    counts[tok] += 1
    return counts


def missing_lex_tokens(
    yields_counts: Counter, lex_terms: set[str]
) -> Counter:
    missing: Counter = Counter()
    for tok, cnt in yields_counts.items():
        if tok not in lex_terms:
            missing[tok] = cnt
    return missing


def run_io(
    io_path: Path,
    grammar_path: Path,
    yields_path: Path,
    iterations: int,
    minruleprob: float,
    debug: int,
    out_path: Path,
) -> None:
    cmd = [
        str(io_path),
        "-V",
        "-n",
        str(iterations),
        "-d",
        str(debug),
        "-p",
        str(minruleprob),
        "-g",
        str(grammar_path),
        str(yields_path),
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fout:
        subprocess.run(cmd, check=True, stdout=fout)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run staged Variational Bayes training over grammar subsets."
    )
    ap.add_argument(
        "--full-grammar",
        required=True,
        help="Full grammar file (used for root rules and lexicon).",
    )
    ap.add_argument(
        "--yields",
        required=True,
        help="Yield file (sentences).",
    )
    ap.add_argument(
        "--stages-dir",
        default="stages",
        help="Directory containing stage grammar files (default: stages).",
    )
    ap.add_argument(
        "--order",
        choices=["VP,TP,CP,INTJ", "INTJ,CP,TP,VP"],
        default="VP,TP,CP,INTJ",
        help="Stage order (default: VP,TP,CP,INTJ).",
    )
    ap.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Iterations per stage (default: 20).",
    )
    ap.add_argument(
        "--io-path",
        default="inside-outside/io",
        help="Path to the io binary (default: inside-outside/io).",
    )
    ap.add_argument(
        "--debug",
        type=int,
        default=1000,
        help="Debug level for io (default: 1000).",
    )
    ap.add_argument(
        "--out-dir",
        default="staged_runs",
        help="Output directory for staged grammars (default: staged_runs).",
    )
    ap.add_argument(
        "--bias",
        type=float,
        default=0.0,
        help="Bias (pseudocount) to add when missing in rule lines (default: 0.0).",
    )
    ap.add_argument(
        "--minruleprob",
        type=float,
        default=1e-20,
        help=(
            "Minimum rule probability for pruning inside io (default: 1e-20; "
            "use 0 to disable pruning)."
        ),
    )
    ap.add_argument(
        "--keep-lex",
        action="store_true",
        help=(
            "Keep lexical rules at their full-grammar weights when first "
            "introduced; learned lexical weights are preserved if present."
        ),
    )
    ap.add_argument(
        "--new-rule-weight",
        type=float,
        default=None,
        help=(
            "Set a fixed weight for rules introduced after stage 1 (default: "
            "use average per-LHS from the previous stage)."
        ),
    )
    ap.add_argument(
        "--validate-lex",
        action="store_true",
        help=(
            "Check that every yield token has a lexical rule before running io."
        ),
    )

    args = ap.parse_args()

    full_grammar = Path(args.full_grammar)
    yields_path = Path(args.yields)
    io_path = Path(args.io_path)
    stages_dir = Path(args.stages_dir)
    out_dir = Path(args.out_dir) / args.order.replace(",", "_")

    full_rules = parse_rules(full_grammar)
    lex_rules = split_lexicon(full_rules)
    terminals = collect_terminals(lex_rules)
    lex_rule_ids = {rule_id(r) for r in lex_rules}
    yields_counts = read_yield_token_counts(yields_path)

    stage_names = args.order.split(",")
    stage_files: Dict[str, Path] = {
        name: stages_dir / f"{name}_stage.txt" for name in stage_names
    }

    prev_rules: Optional[List[Rule]] = None
    prev_output: Optional[Path] = None

    for idx, stage_name in enumerate(stage_names, start=1):
        stage_path = stage_files[stage_name]
        stage_rules = parse_rules(stage_path)
        if not stage_rules:
            raise SystemExit(f"No rules found in stage file: {stage_path}")

        init_lines = merge_stage(stage_rules, prev_rules, lex_rules)
        init_rules = parse_rules_from_lines(init_lines)
        if prev_rules is not None:
            init_lines = apply_avg_weights_for_new_rules(
                init_rules,
                prev_rules,
                lex_rule_ids=lex_rule_ids if args.keep_lex else None,
                new_rule_weight=args.new_rule_weight,
            )
            init_rules = parse_rules_from_lines(init_lines)
        init_rules = dedupe_rules(init_rules)
        if args.validate_lex:
            lex_terms = collect_lex_terms(init_rules)
            missing = missing_lex_tokens(yields_counts, lex_terms)
            if missing:
                top_missing = ", ".join(
                    f"{tok}({cnt})" for tok, cnt in missing.most_common(20)
                )
                raise SystemExit(
                    f"[ERROR] Missing {len(missing)} yield tokens in lexicon: {top_missing}"
                )
        pruned_rules, missing_syms, dropped = prune_rules_missing_rhs(
            init_rules, terminals
        )
        kept_ids = {rule_id(r) for r in pruned_rules}
        stage_dir = out_dir / f"{idx:02d}_{stage_name}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        if dropped:
            details = ", ".join(missing_syms[:10])
            more = "..." if len(missing_syms) > 10 else ""
            print(
                f"[WARN] Dropped {dropped} rules with undefined RHS nonterminals: {details}{more}",
                file=sys.stderr,
            )
        ordered_rules = order_rules_nonlex_then_lex(pruned_rules)
        normalized_lines = [
            normalize_rule_line(r[2], args.bias) for r in ordered_rules
        ]
        init_path = stage_dir / "init_grammar.lt"
        init_path.write_text(
            "\n".join(normalized_lines) + "\n", encoding="utf-8"
        )

        out_path = stage_dir / "results.lt"
        run_io(
            io_path=io_path,
            grammar_path=init_path,
            yields_path=yields_path,
            iterations=args.iterations,
            minruleprob=args.minruleprob,
            debug=args.debug,
            out_path=out_path,
        )

        prev_rules = parse_rules(out_path)
        prev_output = out_path

    if prev_output:
        print(f"Final grammar: {prev_output}")


if __name__ == "__main__":
    main()
