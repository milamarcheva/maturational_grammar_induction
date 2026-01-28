#!/usr/bin/env python3
"""
Run staged VB training using per-stage production files and a shared lexicon.
Lexicon rules are reloaded fresh from the lexicon file at every stage.
New stage rules keep their original weights by default.

Stage order is inferred from the stage folder name, e.g.:
  stages/base_VP_TP -> ["base", "VP", "TP"]
"""

from __future__ import annotations

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
            if "-->" in parts:
                arrow = parts.index("-->")
            elif "->" in parts:
                arrow = parts.index("->")
            else:
                continue
            if arrow == 0 or arrow >= len(parts) - 1:
                continue
            lhs = parts[arrow - 1]
            rhs = parts[arrow + 1 :]
            rules.append((lhs, rhs, s))
    return rules


def rule_id(rule: Rule) -> Tuple[str, Tuple[str, ...]]:
    lhs, rhs, _ = rule
    return lhs, tuple(rhs)


def is_lex_token(token: str) -> bool:
    tok = token
    if len(tok) >= 2 and (
        (tok.startswith("'") and tok.endswith("'"))
        or (tok.startswith('"') and tok.endswith('"'))
    ):
        tok = tok[1:-1]
    if not tok:
        return False
    head = tok[0]
    if head.isalpha():
        return head.islower()
    return True


def is_lex(rule: Rule) -> bool:
    _, rhs, _ = rule
    return len(rhs) == 1 and is_lex_token(rhs[0])


def split_rules(rules: List[Rule]) -> Tuple[List[Rule], List[Rule]]:
    nonlex: List[Rule] = []
    lex: List[Rule] = []
    for rule in rules:
        if is_lex(rule):
            lex.append(rule)
        else:
            nonlex.append(rule)
    return nonlex, lex


def strip_lex_rules(rules: List[Rule]) -> List[Rule]:
    if not rules:
        return []
    return [rule for rule in rules if not is_lex(rule)]


def format_lex_rules(
    rules: Iterable[Rule],
    weight: float,
    bias: float,
) -> List[Rule]:
    seen: set[Tuple[str, Tuple[str, ...]]] = set()
    out: List[Rule] = []
    for lhs, rhs, _ in rules:
        if len(rhs) != 1:
            continue
        rid = (lhs, tuple(rhs))
        if rid in seen:
            continue
        seen.add(rid)
        rhs_str = " ".join(rhs)
        raw = f"{weight} {bias} {lhs} --> {rhs_str}"
        out.append((lhs, rhs, raw))
    return out


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


def merge_rules(
    stage_rules: List[Rule],
    prev_rules: Optional[List[Rule]],
    lexicon_rules: List[Rule],
) -> List[Rule]:
    base = stage_rules if prev_rules is None else prev_rules + stage_rules
    merged = base + lexicon_rules
    return dedupe_rules(merged)


def collect_terminals(lex_rules: List[Rule]) -> set[str]:
    terminals = set()
    for _, rhs, _ in lex_rules:
        if rhs:
            terminals.add(rhs[0])
    return terminals


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
    nonlex: List[Rule] = []
    lex: List[Rule] = []
    for lhs, rhs, raw in rules:
        if len(rhs) == 1 and is_lex_token(rhs[0]):
            lex.append((lhs, rhs, raw))
        else:
            nonlex.append((lhs, rhs, raw))
    return nonlex + lex


def normalize_rule_line(
    raw: str,
    default_bias: float,
) -> str:
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
        weight = nums[0]
        bias = default_bias
    else:
        weight = 1.0
        bias = default_bias
    return f"{weight} {bias} {lhs} --> {' '.join(rhs)}"


def set_rule_bias(raw: str, new_bias: float, default_bias: float) -> str:
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

    if len(nums) >= 1:
        weight = nums[0]
    else:
        weight = 1.0

    if not nums:
        _ = default_bias

    return f"{weight} {new_bias} {lhs} --> {' '.join(rhs)}"


def build_init_lines(
    rules: List[Rule],
    terminals: set[str],
    args: argparse.Namespace,
    prev_prod_bias_map: Optional[Dict[Tuple[str, Tuple[str, ...]], float]] = None,
    prev_lex_bias_map: Optional[Dict[Tuple[str, Tuple[str, ...]], float]] = None,
    new_rule_scale: Optional[float] = None,
) -> List[str]:
    pruned_rules, missing_syms, dropped = prune_rules_missing_rhs(
        rules, terminals
    )
    if dropped:
        details = ", ".join(missing_syms[:10])
        more = "..." if len(missing_syms) > 10 else ""
        print(
            f"[WARN] Dropped {dropped} rules with undefined RHS nonterminals: {details}{more}",
            file=sys.stderr,
        )

    ordered_rules = order_rules_nonlex_then_lex(pruned_rules)
    new_lhs_counts: Dict[str, int] = {}
    prev_prod_ids: Optional[set[Tuple[str, Tuple[str, ...]]]] = None
    if prev_prod_bias_map is not None and args.new_prod_bias_mode == "lhs-parsed":
        prev_prod_ids = set(prev_prod_bias_map.keys())
        for lhs, rhs, raw in ordered_rules:
            if is_lex((lhs, rhs, raw)):
                continue
            rid = (lhs, tuple(rhs))
            if rid not in prev_prod_ids:
                new_lhs_counts[lhs] = new_lhs_counts.get(lhs, 0) + 1

    normalized_lines: List[str] = []
    for lhs, rhs, raw in ordered_rules:
        if is_lex((lhs, rhs, raw)):
            rid = (lhs, tuple(rhs))
            if prev_lex_bias_map is not None:
                prev_bias = prev_lex_bias_map.get(rid, args.lex_init_bias)
                if prev_bias <= 0:
                    prev_bias = args.lex_init_bias
                bias = prev_bias
            else:
                bias = args.lex_init_bias
            normalized_lines.append(
                f"{args.lex_init_weight} {bias} {lhs} --> {' '.join(rhs)}"
            )
            continue
        if prev_prod_bias_map is not None:
            rid = (lhs, tuple(rhs))
            if rid in prev_prod_bias_map:
                prev_bias = prev_prod_bias_map[rid]
                if prev_bias <= 0:
                    prev_bias = args.carryover_prod_bias_min
                normalized_lines.append(
                    f"1.0 {prev_bias} {lhs} --> {' '.join(rhs)}"
                )
                continue
            new_bias = args.bias
            if args.new_prod_bias_mode == "lhs-parsed" and new_rule_scale is not None:
                x_new = new_lhs_counts.get(lhs, 0)
                if x_new > 0:
                    computed = (new_rule_scale * args.new_prod_bias_eta) / x_new
                    if computed > 0:
                        new_bias = computed
            if args.new_prod_bias_min is not None and new_bias < args.new_prod_bias_min:
                new_bias = args.new_prod_bias_min
            normalized_lines.append(set_rule_bias(raw, new_bias, args.bias))
            continue
        normalized_lines.append(normalize_rule_line(raw, args.bias))
    return normalized_lines


def collect_lex_terms(rules: List[Rule]) -> set[str]:
    lex_terms = set()
    for _, rhs, _ in rules:
        if len(rhs) == 1 and is_lex_token(rhs[0]):
            lex_terms.add(rhs[0])
    return lex_terms


def read_yield_token_counts(yields_path: Path) -> Tuple[Counter, int]:
    counts: Counter = Counter()
    total_sentences = 0
    with yields_path.open(encoding="utf-8") as f:
        for line in f:
            total_sentences += 1
            for tok in line.strip().split():
                if tok:
                    counts[tok] += 1
    return counts, total_sentences


def missing_lex_tokens(
    yields_counts: Counter, lex_terms: set[str]
) -> Counter:
    missing: Counter = Counter()
    for tok, cnt in yields_counts.items():
        if tok not in lex_terms:
            missing[tok] = cnt
    return missing


def build_prev_prod_bias_map(
    rules: List[Rule],
    scale: float = 1.0,
    base_bias: float = 0.0,
) -> Dict[Tuple[str, Tuple[str, ...]], float]:
    bias_map: Dict[Tuple[str, Tuple[str, ...]], float] = {}
    for lhs, rhs, raw in rules:
        rid = (lhs, tuple(rhs))
        bias_map[rid] = extract_rule_weight(raw) * scale + base_bias
    return bias_map


def build_prev_lex_bias_map(
    rules: List[Rule],
    scale: float = 1.0,
) -> Dict[Tuple[str, Tuple[str, ...]], float]:
    bias_map: Dict[Tuple[str, Tuple[str, ...]], float] = {}
    for lhs, rhs, raw in rules:
        rid = (lhs, tuple(rhs))
        bias_map[rid] = extract_rule_weight(raw) * scale
    return bias_map


def io_log_path(results_path: Path) -> Path:
    return results_path.with_suffix(results_path.suffix + ".log")


def extract_failed_sentences(results_path: Path) -> Optional[int]:
    failed = None
    try:
        lines = results_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None
    for line in lines:
        if "sentences failed to parse" not in line:
            continue
        parts = line.split()
        if "sentences" not in parts:
            continue
        try:
            idx = parts.index("sentences")
            count = int(parts[idx - 1])
        except (ValueError, IndexError):
            continue
        failed = count
    return failed


def parsed_sentence_scale(
    results_path: Path,
    total_sentences: int,
    scale_factor: float,
    label: str,
) -> float:
    log_path = io_log_path(results_path)
    failed = extract_failed_sentences(log_path)
    if failed is None:
        failed = extract_failed_sentences(results_path)
    if failed is None:
        print(
            f"[WARN] Could not read failed sentence count from {log_path}; "
            f"{label} bias scale set to 1.0",
            file=sys.stderr,
        )
        return 1.0
    parsed = total_sentences - failed
    if parsed <= 0:
        print(
            f"[WARN] Non-positive parsed count ({parsed}) from {results_path}; "
            f"{label} bias scale set to 1.0",
            file=sys.stderr,
        )
        return 1.0
    scaled = float(parsed) * scale_factor
    print(
        f"[info] Parsed sentences for {label} bias scaling: {parsed} "
        f"(total {total_sentences} - failed {failed}); "
        f"scale {scale_factor} -> {scaled}",
        file=sys.stderr,
    )
    return scaled


def run_io(
    io_path: Path,
    grammar_path: Path,
    yields_path: Path,
    maxits: int,
    minits: Optional[int],
    minruleprob: float,
    debug: int,
    out_path: Path,
) -> None:
    cmd = [
        str(io_path),
        "-V",
        "-n",
        str(maxits),
        "-d",
        str(debug),
        "-p",
        str(minruleprob),
    ]
    if minits is not None:
        cmd.extend(["-m", str(minits)])
    cmd.extend(
        [
            "-g",
            str(grammar_path),
            str(yields_path),
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = io_log_path(out_path)
    with out_path.open("w", encoding="utf-8") as fout, log_path.open(
        "w", encoding="utf-8"
    ) as ferr:
        subprocess.run(cmd, check=True, stdout=fout, stderr=ferr)


def infer_stage_order(stage_dir: Path) -> List[str]:
    tokens = [tok for tok in stage_dir.name.split("_") if tok]
    if not tokens:
        raise SystemExit(
            "Stage folder name must encode order, e.g. base_VP_TP."
        )
    return tokens


def format_scale_tag(value: float) -> str:
    return f"{value}".replace(".", "p")


def build_run_tag(stage_dir: Path, args: argparse.Namespace) -> str:
    parent_name = stage_dir.parent.name
    if parent_name.startswith("stages"):
        stage_label = f"{parent_name}_{stage_dir.name}"
    elif parent_name:
        stage_label = f"stages_{parent_name}_{stage_dir.name}"
    else:
        stage_label = f"stages_{stage_dir.name}"
    tag = stage_label
    tag += f"__ps-{format_scale_tag(args.prod_bias_scale)}"
    tag += f"__bias-{format_scale_tag(args.bias)}"
    tag += f"__cb-{format_scale_tag(args.carryover_prod_bias_min)}"
    tag += f"__nbm-{args.new_prod_bias_mode}"
    tag += f"__nbe-{format_scale_tag(args.new_prod_bias_eta)}"
    if args.new_prod_bias_min is not None:
        tag += f"__nbmin-{format_scale_tag(args.new_prod_bias_min)}"
    tag += "__lex-from-prev"
    tag += f"__ls-{format_scale_tag(args.lex_bias_scale)}"
    return tag


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run staged VB training over per-stage production files."
    )
    ap.add_argument(
        "--stage-folder",
        required=True,
        help="Folder containing *_stage.txt files; name encodes stage order.",
    )
    ap.add_argument(
        "--yields",
        required=True,
        help="Yield file (sentences).",
    )
    ap.add_argument(
        "--lexicon",
        default=str(Path(__file__).resolve().parents[1] / "grammars/lexicon.txt"),
        help="Lexicon grammar file (default: grammars/lexicon.txt).",
    )
    ap.add_argument(
        "--lex-init-weight",
        type=float,
        default=1.0,
        help="Weight to use when initializing lexical rules (default: 1.0).",
    )
    ap.add_argument(
        "--lex-init-bias",
        type=float,
        default=0.1,
        help="Bias/pseudocount to use when initializing lexical rules (default: 0.1).",
    )
    ap.add_argument(
        "--lex-bias-scale",
        type=float,
        default=1.0,
        help=(
            "Multiplier for parsed-sentence scaling of lexical biases from the "
            "previous stage (default: 1.0)."
        ),
    )
    ap.add_argument(
        "--maxits",
        type=int,
        default=21,
        help="Maximum iterations for io per stage (default: 21).",
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
        default=0.1,
        help=(
            "Bias (pseudocount) for new production rules or missing biases "
            "(default: 0.1)."
        ),
    )
    ap.add_argument(
        "--carryover_prod_bias_min",
        type=float,
        default=None,
        help=(
            "Minimum/additive bias term for carried-over production rules "
            "(default: use --bias)."
        ),
    )
    ap.add_argument(
        "--new-prod-bias-mode",
        choices=["fixed", "lhs-parsed"],
        default="fixed",
        help=(
            "How to set bias for new production rules: fixed uses --bias; "
            "lhs-parsed uses (N * prod-bias-scale * eta) / X_new per LHS."
        ),
    )
    ap.add_argument(
        "--new-prod-bias-eta",
        type=float,
        default=0.01,
        help="Scaling factor (eta) for lhs-parsed new rule bias (default: 0.01).",
    )
    ap.add_argument(
        "--new-prod-bias-min",
        type=float,
        default=None,
        help="Minimum bias for new production rules (default: unset).",
    )
    ap.add_argument(
        "--prod-bias-scale",
        type=float,
        default=1.0,
        help=(
            "Multiplier for parsed-sentence scaling of production biases from "
            "the previous stage (default: 1.0)."
        ),
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
        "--minits",
        type=int,
        default=None,
        help="Minimum iterations for io before stopping early (default: io default).",
    )
    ap.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Deprecated alias for --maxits.",
    )
    ap.add_argument(
        "--validate-lex",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Check that every yield token has a lexical rule before running io.",
    )

    args = ap.parse_args()

    if args.iterations is not None:
        args.maxits = args.iterations
    if args.carryover_prod_bias_min is None:
        args.carryover_prod_bias_min = args.bias

    stage_dir = Path(args.stage_folder).expanduser().resolve()
    yields_path = Path(args.yields).expanduser().resolve()
    io_path = Path(args.io_path)
    lexicon_path = Path(args.lexicon).expanduser().resolve()
    run_tag = build_run_tag(stage_dir, args)
    out_dir = Path(args.out_dir) / run_tag

    stage_order = infer_stage_order(stage_dir)
    stage_files: Dict[str, Path] = {
        name: stage_dir / f"{name}_stage.txt" for name in stage_order
    }
    missing = [name for name, path in stage_files.items() if not path.exists()]
    if missing:
        raise SystemExit(
            f"Missing stage files for: {', '.join(missing)} in {stage_dir}"
        )

    lexicon_rules_all = parse_rules(lexicon_path)
    if not lexicon_rules_all:
        raise SystemExit(f"No rules found in lexicon file: {lexicon_path}")
    lexicon_nonlex, lexicon_lex = split_rules(lexicon_rules_all)
    if not lexicon_lex:
        raise SystemExit(f"No lexical rules found in lexicon file: {lexicon_path}")
    if lexicon_nonlex:
        print(
            f"[WARN] Ignoring {len(lexicon_nonlex)} non-lex rules in {lexicon_path}",
            file=sys.stderr,
        )
    yields_counts, total_sentences = read_yield_token_counts(yields_path)

    prev_rules: Optional[List[Rule]] = None
    prev_lex_rules: Optional[List[Rule]] = None
    prev_output: Optional[Path] = None
    accumulated_stage_lex: List[Rule] = []
    accumulated_stage_lex_ids: set[Tuple[str, Tuple[str, ...]]] = set()

    for idx, stage_name in enumerate(stage_order, start=1):
        stage_path = stage_files[stage_name]
        stage_rules_all = parse_rules(stage_path)
        if not stage_rules_all:
            raise SystemExit(f"No rules found in stage file: {stage_path}")
        stage_rules, stage_lex = split_rules(stage_rules_all)
        new_stage_lex: List[Rule] = []
        for rule in stage_lex:
            rid = rule_id(rule)
            if rid in accumulated_stage_lex_ids:
                continue
            accumulated_stage_lex_ids.add(rid)
            accumulated_stage_lex.append(rule)
            new_stage_lex.append(rule)
        lex_sources = list(lexicon_lex) + list(accumulated_stage_lex)
        lex_rules = format_lex_rules(
            lex_sources,
            weight=args.lex_init_weight,
            bias=args.lex_init_bias,
        )
        lex_rule_map = {rule_id(rule): rule[2] for rule in lex_rules}
        if not lex_rules:
            raise SystemExit(f"No lexical rules available for stage: {stage_path}")
        terminals = collect_terminals(lex_rules)
        if new_stage_lex:
            print(
                f"[info] Added {len(new_stage_lex)} stage lexical rules from {stage_path}",
                file=sys.stderr,
            )
            for lhs, rhs, _ in new_stage_lex:
                rid = (lhs, tuple(rhs))
                raw = lex_rule_map.get(
                    rid,
                    f"{args.lex_init_weight} {args.lex_init_bias} {lhs} --> {' '.join(rhs)}",
                )
                print(f"[info]   {raw}", file=sys.stderr)

        init_rules = merge_rules(stage_rules, prev_rules, [])
        init_rules = dedupe_rules(init_rules + lex_rules)

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

        stage_out_dir = out_dir / f"{idx:02d}_{stage_name}"
        stage_out_dir.mkdir(parents=True, exist_ok=True)
        prev_bias_map = None
        prod_new_rule_scale = None
        if prev_rules is not None:
            scale = 1.0
            if prev_output is not None:
                scale = parsed_sentence_scale(
                    prev_output,
                    total_sentences,
                    args.prod_bias_scale,
                    "prod",
                )
                prod_new_rule_scale = scale
            prev_bias_map = build_prev_prod_bias_map(
                prev_rules,
                scale=scale,
                base_bias=args.carryover_prod_bias_min,
            )
        prev_lex_bias_map = None
        if prev_lex_rules is not None:
            scale = 1.0
            if prev_output is not None:
                scale = parsed_sentence_scale(
                    prev_output,
                    total_sentences,
                    args.lex_bias_scale,
                    "lex",
                )
            prev_lex_bias_map = build_prev_lex_bias_map(prev_lex_rules, scale=scale)
        normalized_lines = build_init_lines(
            init_rules,
            terminals,
            args,
            prev_prod_bias_map=prev_bias_map,
            prev_lex_bias_map=prev_lex_bias_map,
            new_rule_scale=prod_new_rule_scale,
        )
        init_path = stage_out_dir / "init_grammar.lt"
        init_path.write_text("\n".join(normalized_lines) + "\n", encoding="utf-8")

        out_path = stage_out_dir / "results.lt"
        run_io(
            io_path=io_path,
            grammar_path=init_path,
            yields_path=yields_path,
            maxits=args.maxits,
            minits=args.minits,
            minruleprob=args.minruleprob,
            debug=args.debug,
            out_path=out_path,
        )

        prev_all_rules = parse_rules(out_path)
        prev_rules = strip_lex_rules(prev_all_rules)
        prev_lex_rules = [rule for rule in prev_all_rules if is_lex(rule)]
        prev_output = out_path

    if prev_output:
        print(f"Final grammar: {prev_output}")


if __name__ == "__main__":
    main()
