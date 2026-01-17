#!/usr/bin/env python3
import argparse
import subprocess
import sys
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


def split_lexicon(
    rules: List[Rule],
) -> Tuple[List[Rule], List[Rule], str]:
    if not rules:
        raise SystemExit("No rules found in full grammar.")
    root_label = rules[0][0]
    nonterminals = {lhs for lhs, _, _ in rules}
    root_rules = [r for r in rules if r[0] == root_label]
    lex_rules = []
    for lhs, rhs, raw in rules:
        if len(rhs) == 1 and rhs[0] not in nonterminals:
            lex_rules.append((lhs, rhs, raw))
    return root_rules, lex_rules, root_label


def collect_terminals(lex_rules: List[Rule]) -> set[str]:
    terminals = set()
    for _, rhs, _ in lex_rules:
        if rhs:
            terminals.add(rhs[0])
    return terminals


def merge_stage(
    stage_rules: List[Rule],
    prev_rules: Optional[List[Rule]],
    root_rules: List[Rule],
    lex_rules: List[Rule],
    root_label: str,
) -> List[str]:
    if prev_rules is None:
        seen = set()
        lines: List[str] = []
        for rule in root_rules + stage_rules + lex_rules:
            rid = rule_id(rule)
            if rid in seen:
                continue
            seen.add(rid)
            lines.append(rule[2])
        return lines

    prev_ids = {rule_id(r) for r in prev_rules}
    prev_root = [r for r in prev_rules if r[0] == root_label]
    prev_other = [r for r in prev_rules if r[0] != root_label]

    new_stage = [r for r in stage_rules if rule_id(r) not in prev_ids]
    new_lex = [r for r in lex_rules if rule_id(r) not in prev_ids]

    lines = [r[2] for r in prev_root + prev_other + new_stage + new_lex]
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


def run_io(
    io_path: Path,
    grammar_path: Path,
    yields_path: Path,
    iterations: int,
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

    args = ap.parse_args()

    full_grammar = Path(args.full_grammar)
    yields_path = Path(args.yields)
    io_path = Path(args.io_path)
    stages_dir = Path(args.stages_dir)
    out_dir = Path(args.out_dir) / args.order.replace(",", "_")

    full_rules = parse_rules(full_grammar)
    root_rules, lex_rules, root_label = split_lexicon(full_rules)
    terminals = collect_terminals(lex_rules)

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

        init_lines = merge_stage(
            stage_rules, prev_rules, root_rules, lex_rules, root_label
        )
        init_rules = parse_rules_from_lines(init_lines)
        pruned_rules, missing_syms, dropped = prune_rules_missing_rhs(
            init_rules, terminals
        )
        if dropped:
            details = ", ".join(missing_syms[:10])
            more = "..." if len(missing_syms) > 10 else ""
            print(
                f"[WARN] Dropped {dropped} rules with undefined RHS nonterminals: {details}{more}",
                file=sys.stderr,
            )
        normalized_lines = [
            normalize_rule_line(r[2], args.bias) for r in pruned_rules
        ]
        stage_dir = out_dir / f"{idx:02d}_{stage_name}"
        init_path = stage_dir / "init_grammar.lt"
        init_path.parent.mkdir(parents=True, exist_ok=True)
        init_path.write_text(
            "\n".join(normalized_lines) + "\n", encoding="utf-8"
        )

        out_path = stage_dir / "results.lt"
        run_io(
            io_path=io_path,
            grammar_path=init_path,
            yields_path=yields_path,
            iterations=args.iterations,
            debug=args.debug,
            out_path=out_path,
        )

        prev_rules = parse_rules(out_path)
        prev_output = out_path

    if prev_output:
        print(f"Final grammar: {prev_output}")


if __name__ == "__main__":
    main()
