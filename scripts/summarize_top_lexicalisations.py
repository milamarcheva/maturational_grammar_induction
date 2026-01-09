#!/usr/bin/env python3
"""
Summarize how often top nouns/verbs are emitted by expected nonterminals.

Example:
  python milas_scripts/summarize_top_lexicalisations.py \
    --run-dir milas_grammars/results/productions_NVA_VB_pb0p1_lb0p01_primed_20251220_0901 \
    --noun-nts N \
    --verb-nts V
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple


def parse_lex_file(path: Path) -> List[Tuple[float, str, str]]:
    """
    Parse lines like:
      0.001234  N --> word
    Returns list of (prob, lhs, rhs_word).
    """
    entries: List[Tuple[float, str, str]] = []
    if not path.exists():
        return entries

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "-->" not in line:
                continue
            parts = line.split()
            try:
                prob = float(parts[0])
            except (ValueError, IndexError):
                continue
            try:
                arrow_idx = parts.index("-->")
            except ValueError:
                continue
            if arrow_idx < 1 or arrow_idx + 1 >= len(parts):
                continue
            lhs = parts[arrow_idx - 1]
            rhs_word = parts[arrow_idx + 1]
            entries.append((prob, lhs, rhs_word))
    return entries


def best_nt_by_word(entries: List[Tuple[float, str, str]]) -> Dict[str, str]:
    """
    Given lex entries sorted by word (not required) but possibly multiple per word,
    pick the highest-probability NT per word.
    """
    best: Dict[str, Tuple[float, str]] = {}
    for prob, lhs, word in entries:
        if (word not in best) or (prob > best[word][0]):
            best[word] = (prob, lhs)
    return {w: nt for w, (p, nt) in best.items()}


def score_file(path: Path, expected_nts: Set[str], label: str) -> None:
    entries = parse_lex_file(path)
    if not entries:
        print(f"[WARN] No entries found in {path}")
        return
    best_map = best_nt_by_word(entries)
    total = len(best_map)
    hits = sum(1 for nt in best_map.values() if nt in expected_nts)
    pct = (hits / total * 100.0) if total else 0.0
    # has_a: word appears with at least one expected NT in any lexicalisation
    all_hits = set(word for prob, lhs, word in entries if lhs in expected_nts)
    has_a = len(all_hits)
    has_a_pct = (has_a / total * 100.0) if total else 0.0
    print(
        f"{label}: {hits}/{total} ({pct:.1f}%) best-only | "
        f"{has_a}/{total} ({has_a_pct:.1f}%) has_a (any expected NT)"
        f" | expected NTs {sorted(expected_nts)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize top lexicalisations alignment with expected NTs.")
    ap.add_argument("--run-dir", required=True, help="Run directory under milas_grammars/results/ containing top_nouns_lexicalisations.txt etc.")
    ap.add_argument("--noun-nts", nargs="+", default=["N"], help="Nonterminal(s) expected to emit nouns. Default: N")
    ap.add_argument("--verb-nts", nargs="+", default=["V"], help="Nonterminal(s) expected to emit verbs. Default: V")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    noun_file = run_dir / "top_nouns_lexicalisations.txt"
    verb_file = run_dir / "top_verbs_lexicalisations.txt"

    score_file(noun_file, set(args.noun_nts), "Nouns")
    score_file(verb_file, set(args.verb_nts), "Verbs")


if __name__ == "__main__":
    main()
