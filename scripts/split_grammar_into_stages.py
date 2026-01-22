#!/usr/bin/env python3
"""
Split a grammar into stage files (base, VP, TP, CP, INTJ) using heuristics.

Notes:
- Stage membership is determined by the earliest stage that licenses the PTs
  appearing in a rule (cumulative across stages).
- Lexical rules are excluded from all stages by default.
- Edit the symbol lists below to tune stage assignment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class StageDef:
    name: str
    pts: Set[str]
    nts: Optional[Set[str]] = None
    exclude: Optional[Set[str]] = None


STAGE_DEFS = [
    StageDef(
        name="base",
        pts={"NN", "NNP", "PRP", "PRP$", "VB", },
        nts={"ROOT", "S", "FRAG", "NP", "VP"},
    ),
    StageDef(
        name="baseINTJ",
        pts={"NN", "VB", "PRP", "PRP$", "VB", "UH"},
        nts={"ROOT", "S", "FRAG", "INTJ"},
    ),
    StageDef(
        name="VP",
        pts={
            "DT",
            "PDT",
            "POS",
            "NNS",
            "CD",
            "JJ",
            "JJR",
            "JJS",
            "RB",
            "RBR",
            "RBS",
            "IN",
            "TO",
            "RP",
            "VBG",
            "VBN",
            "NOT",
            "DIV",
            "EX",
        },
        exclude={"SBAR", "SBARQ", "SQ", "INTJ", "SINV", "WHNP", "WHNPP", "WHPP"},
    ),
    StageDef(
        name="TP",
        pts={"VBD", "VBP", "VBZ", "AUX", "COP", "MD", "ASP", "T", "PRS"},
        exclude={"SBAR", "SBARQ", "SQ", "INTJ", "SINV", "WHNP", "WHNPP", "WHPP"},
    ),
    StageDef(
        name="INTJ",
        pts={"UH"},
        exclude={"SBAR", "SBARQ", "SQ", "SINV", "WHNP", "WHNPP", "WHPP"},
    ),
    StageDef(
        name="CP",
        pts={"COMP", "CC", "WP", "WP$", "WRB", "WDT", "FW"},
    ),
]

STAGE_DEF_MAP = {stage.name: stage for stage in STAGE_DEFS}
DEFAULT_ORDER = ["base", "VP", "TP", "INTJ", "CP"]


@dataclass(frozen=True)
class Rule:
    lhs: str
    rhs: List[str]
    raw: str


def parse_rules(path: Path) -> List[Rule]:
    rules: List[Rule] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        parts = raw.split()
        if "-->" in parts:
            arrow = parts.index("-->")
        elif "->" in parts:
            arrow = parts.index("->")
        else:
            continue
        if arrow <= 0 or arrow >= len(parts) - 1:
            continue
        lhs = parts[arrow - 1]
        rhs = parts[arrow + 1 :]
        rules.append(Rule(lhs=lhs, rhs=rhs, raw=raw))
    return rules


def is_lex(rule: Rule, nonterminals: set[str]) -> bool:
    return len(rule.rhs) == 1 and rule.rhs[0] not in nonterminals


def build_pt_stage(stage_defs: Sequence[StageDef]) -> Dict[str, int]:
    pt_stage: Dict[str, int] = {}
    for idx, stage in enumerate(stage_defs):
        for pt in stage.pts:
            if pt not in pt_stage:
                pt_stage[pt] = idx
    return pt_stage


def symbol_stage(
    sym: str,
    pt_stage: Dict[str, int],
    nt_stage: Dict[str, int],
) -> Optional[int]:
    if sym in pt_stage:
        return pt_stage[sym]
    return nt_stage.get(sym)


def rule_has_excluded_nt(
    rule: Rule,
    exclude: Optional[Set[str]],
    nonterminals: set[str],
) -> bool:
    if not exclude:
        return False
    if rule.lhs in exclude:
        return True
    return any(sym in exclude and sym in nonterminals for sym in rule.rhs)


def compute_nt_stages(
    rules: Sequence[Rule],
    nonterminals: set[str],
    pt_stage: Dict[str, int],
    base_nts: Set[str],
    min_non_base_stage: int,
    max_stage: int,
) -> Tuple[Dict[str, int], List[str]]:
    nt_stage: Dict[str, Optional[int]] = {
        nt: None for nt in nonterminals if nt not in pt_stage
    }
    for nt in base_nts:
        if nt in nt_stage:
            nt_stage[nt] = 0

    changed = True
    while changed:
        changed = False
        for rule in rules:
            if rule.lhs in pt_stage or rule.lhs in base_nts:
                continue
            rhs_stage = 0
            unknown = False
            for sym in rule.rhs:
                if sym in pt_stage:
                    rhs_stage = max(rhs_stage, pt_stage[sym])
                    continue
                stage = nt_stage.get(sym)
                if stage is None:
                    unknown = True
                    break
                rhs_stage = max(rhs_stage, stage)
            if unknown:
                continue
            candidate = max(rhs_stage, min_non_base_stage)
            current = nt_stage.get(rule.lhs)
            if current is None or candidate < current:
                nt_stage[rule.lhs] = candidate
                changed = True

    unresolved = [nt for nt, stage in nt_stage.items() if stage is None]
    for nt in unresolved:
        nt_stage[nt] = max_stage

    return {nt: stage for nt, stage in nt_stage.items() if stage is not None}, unresolved


def parse_stage_order(order: str) -> List[str]:
    names = [name.strip() for name in order.split(",") if name.strip()]
    if not names:
        raise SystemExit("Stage order must be a non-empty comma-separated list.")
    unknown = [name for name in names if name not in STAGE_DEF_MAP]
    if unknown:
        raise SystemExit(f"Unknown stage name(s): {', '.join(unknown)}")
    if len(set(names)) != len(names):
        raise SystemExit("Stage order must not contain duplicates.")
    first_stage = STAGE_DEF_MAP[names[0]]
    if first_stage.nts is None:
        raise SystemExit(
            "Stage order must start with a base stage (e.g., base or baseINTJ)."
        )
    return names


def write_stage(path: Path, rules: Sequence[Rule]) -> None:
    content = "\n".join(rule.raw for rule in rules)
    path.write_text(content + ("\n" if content else ""), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Split a grammar into base/VP/TP/CP/INTJ stage files."
    )
    ap.add_argument(
        "--grammar",
        required=True,
        help="Input grammar file (e.g. grammars/mature/*.txt).",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Base output directory (default: <repo>/stages).",
    )
    ap.add_argument(
        "--lex-out",
        default=None,
        help="Optional path to write excluded lexical rules.",
    )
    ap.add_argument(
        "--order",
        default=",".join(DEFAULT_ORDER),
        help="Stage order (default: base,VP,TP,INTJ,CP).",
    )
    args = ap.parse_args()

    grammar_path = Path(args.grammar).expanduser().resolve()
    if args.out_dir:
        base_out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        base_out_dir = Path(__file__).resolve().parents[1] / "stages"

    rules = parse_rules(grammar_path)
    if not rules:
        raise SystemExit(f"No rules found in {grammar_path}")

    nonterminals = {rule.lhs for rule in rules}
    excluded_lex: List[Rule] = []
    excluded_nonlex: List[Rule] = []

    stage_order = parse_stage_order(args.order)
    stage_defs = [STAGE_DEF_MAP[name] for name in stage_order]
    base_idx = 0
    min_non_base_stage = 1
    max_stage = len(stage_order) - 1
    stage_dir = base_out_dir / "_".join(stage_order)
    stage_dir.mkdir(parents=True, exist_ok=True)

    lex_rules = [rule for rule in rules if is_lex(rule, nonterminals)]
    excluded_lex.extend(lex_rules)

    preterminals = {rule.lhs for rule in lex_rules}
    pt_stage = build_pt_stage(stage_defs)
    allowed_pts = set(pt_stage.keys())
    disallowed_pts = preterminals - allowed_pts

    nonlex_rules = [rule for rule in rules if not is_lex(rule, nonterminals)]
    allowed_nonlex: List[Rule] = []
    for rule in nonlex_rules:
        if rule.lhs in disallowed_pts:
            excluded_nonlex.append(rule)
            continue
        if any(sym in disallowed_pts for sym in rule.rhs):
            excluded_nonlex.append(rule)
            continue
        allowed_nonlex.append(rule)

    base_nts = stage_defs[base_idx].nts or set()
    nt_stage, unresolved_nts = compute_nt_stages(
        allowed_nonlex,
        nonterminals,
        pt_stage,
        base_nts,
        min_non_base_stage=min_non_base_stage,
        max_stage=max_stage,
    )

    stage_rules: Dict[str, List[Rule]] = {stage: [] for stage in stage_order}
    unknown_symbols: set[str] = set()

    for rule in allowed_nonlex:
        rhs_stage = 0
        unknown = False
        for sym in rule.rhs:
            sym_stage = symbol_stage(sym, pt_stage, nt_stage)
            if sym_stage is None:
                unknown = True
                unknown_symbols.add(sym)
                break
            rhs_stage = max(rhs_stage, sym_stage)
        lhs_stage = symbol_stage(rule.lhs, pt_stage, nt_stage)
        if lhs_stage is None:
            unknown = True
            unknown_symbols.add(rule.lhs)
            lhs_stage = max_stage
        stage_idx = max(rhs_stage, lhs_stage)
        while stage_idx < max_stage and rule_has_excluded_nt(
            rule,
            stage_defs[stage_idx].exclude,
            nonterminals,
        ):
            stage_idx += 1
        if unknown:
            stage_idx = max_stage
        stage_name = stage_order[stage_idx]
        stage_rules[stage_name].append(rule)

    for stage in stage_order:
        stage_path = stage_dir / f"{stage}_stage.txt"
        write_stage(stage_path, stage_rules[stage])

    if args.lex_out:
        lex_out = Path(args.lex_out).expanduser().resolve()
        write_stage(lex_out, excluded_lex)

    total = sum(len(rules) for rules in stage_rules.values())
    print(f"Split {total} rules into stages in {stage_dir}")
    if excluded_lex:
        print(f"Excluded lexical rules: {len(excluded_lex)}")
    if excluded_nonlex:
        print(f"Excluded non-lex rules (disallowed PTs): {len(excluded_nonlex)}")
    if unresolved_nts:
        print(f"Unresolved NTs forced to last stage: {len(unresolved_nts)}")
    if unknown_symbols:
        print(f"Unknown symbols forced to last stage: {len(unknown_symbols)}")
    if args.lex_out:
        print(f"Wrote excluded lexicon to {lex_out}")
    for stage in stage_order:
        print(f"{stage}: {len(stage_rules[stage])}")
    if excluded_nonlex:
        print("Excluded non-lex rules (disallowed PTs):")
        for rule in excluded_nonlex:
            print(rule.raw)
    if unresolved_nts:
        print("Unresolved NTs forced to last stage:")
        for nt in unresolved_nts:
            print(nt)


if __name__ == "__main__":
    main()
