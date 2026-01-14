#!/usr/bin/env python3
"""
Create a grammar by combining input productions with a generated lexicon.
The generated lexicon uses all preterminals from the input grammar and all
terminals from a sentence file (one sentence per line).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


Rule = Tuple[str, List[str]]
ParsedRule = Tuple[str, List[str], List[float]]
LexRule = Tuple[str, str]
OPEN_CLASS = {
    "NN", "NNS", "NNP",
    "VB", "VBD", "VBP", "VBZ", "VBG", "VBN",
    "JJ", "JJR", "JJS",
    "RB", "RBR", "RBS",
    "CD", "FW",
}
BOUND_MORPHEME_PTS = {"ASP", "DIV", "PRS", "T"}


def iter_rules(lines: Iterable[str]) -> List[ParsedRule]:
    rules: List[ParsedRule] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "-->" not in line:
            continue
        left, right = line.split("-->", 1)
        left_tokens = left.split()
        if not left_tokens:
            continue
        rhs = right.split()
        if not rhs:
            continue
        # LHS is always the token immediately before the arrow (counts/weights may precede it).
        lhs = left_tokens[-1]
        prefix_vals: List[float] = []
        for tok in left_tokens[:-1]:
            try:
                prefix_vals.append(float(tok))
            except ValueError:
                prefix_vals = []
                break
        rules.append((lhs, rhs, prefix_vals))
    return rules


def split_rules(
    rules: Sequence[ParsedRule],
) -> Tuple[List[Rule], List[LexRule], List[str], Dict[str, Dict[str, float]]]:
    nonterminals = {lhs for lhs, _, _ in rules}
    productions: List[Rule] = []
    lex_rules: List[LexRule] = []
    preterminals = set()
    lex_counts: Dict[str, Dict[str, float]] = defaultdict(dict)
    for lhs, rhs, prefix_vals in rules:
        is_lex = len(rhs) == 1 and rhs[0] not in nonterminals
        if is_lex:
            preterminals.add(lhs)
            word = rhs[0]
            lex_rules.append((lhs, word))
            count = prefix_vals[0] if prefix_vals else 0.0
            lex_counts.setdefault(lhs, {})
            lex_counts[lhs][word] = lex_counts[lhs].get(word, 0.0) + count
        else:
            productions.append((lhs, rhs))
    return productions, lex_rules, sorted(preterminals), lex_counts


def read_vocab(sent_path: Path) -> List[str]:
    vocab = set()
    with sent_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for tok in line.split():
                if tok:
                    vocab.add(tok)
    return sorted(vocab)


def build_uniform_lexicon(preterminals: Sequence[str], vocab: Sequence[str]) -> List[LexRule]:
    return [(lhs, word) for lhs in preterminals for word in vocab]


def build_postagged_lexicon(
    lex_rules: Sequence[LexRule],
    preterminals: Sequence[str],
    vocab: Sequence[str],
) -> List[LexRule]:
    vocab_set = set(vocab)
    lex_vocab_all = {word for _, word in lex_rules}
    seen_pairs = set()
    kept: List[LexRule] = []
    for lhs, word in lex_rules:
        if word not in vocab_set:
            continue
        pair = (lhs, word)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        kept.append(pair)

    open_class_preterms = [pt for pt in preterminals if pt in OPEN_CLASS]
    missing_tokens = [word for word in vocab if word not in lex_vocab_all]
    for word in missing_tokens:
        for lhs in open_class_preterms:
            kept.append((lhs, word))
    return kept


def build_primed_by_freq_lexicon(
    lex_counts: Dict[str, Dict[str, float]],
    preterminals: Sequence[str],
    vocab: Sequence[str],
    top_n: int = 5,
) -> List[LexRule]:
    vocab_set = set(vocab)
    top_rules: List[LexRule] = []
    top_words = set()
    for pt in preterminals:
        counts = lex_counts.get(pt, {})
        items = [(word, count) for word, count in counts.items() if word in vocab_set]
        items.sort(key=lambda item: (-item[1], item[0]))
        for word, _ in items[:top_n]:
            top_rules.append((pt, word))
            top_words.add(word)

    uniform_tokens = [word for word in vocab if word not in top_words]
    uniform_preterms = [pt for pt in preterminals if pt not in BOUND_MORPHEME_PTS]
    uniform_rules = [(pt, word) for word in uniform_tokens for pt in uniform_preterms]
    return top_rules + uniform_rules


def write_grammar(
    output_path: Path,
    productions: Sequence[Rule],
    lexical_rules: Sequence[LexRule],
    weight_prod: float,
    pseudocount_prod: float,
    weight_lex: float,
    pseudocount_lex: float,
) -> None:
    with output_path.open("w", encoding="utf-8") as fout:
        for lhs, rhs in productions:
            rhs_str = " ".join(rhs)
            fout.write(f"{weight_prod} {pseudocount_prod}  {lhs} --> {rhs_str}\n")
        for lhs, word in lexical_rules:
            fout.write(f"{weight_lex} {pseudocount_lex}  {lhs} --> {word}\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create a grammar from productions plus a generated lexicon."
    )
    ap.add_argument(
        "--grammar",
        required=True,
        help="Input grammar with productions and lexicalisations.",
    )
    ap.add_argument(
        "--sentences",
        required=True,
        help="Sentence file (one sentence per line) to build vocab.",
    )
    ap.add_argument(
        "--output",
        help="Output directory (default: maturational_grammar_induction/grammars).",
    )
    ap.add_argument(
        "--lexicalisation",
        default="uniform",
        choices=["uniform", "postagged", "primed_by_freq"],
        help="Lexicalisation scheme to use (default: uniform).",
    )
    ap.add_argument(
        "--weight-prod",
        type=float,
        default=1.0,
        help="Weight for production rules (default: 1.0).",
    )
    ap.add_argument(
        "--weight-lex",
        type=float,
        default=1.0,
        help="Weight for lexical rules (default: 1.0).",
    )
    ap.add_argument(
        "--pseudocount-prod",
        type=float,
        default=0.1,
        help="Pseudocount for production rules (default: 0.1).",
    )
    ap.add_argument(
        "--pseudocount-lex",
        type=float,
        default=0.1,
        help="Pseudocount for lexical rules (default: 0.1).",
    )

    args = ap.parse_args()

    grammar_path = Path(args.grammar)
    sentences_path = Path(args.sentences)
    out_dir = Path(args.output) if args.output else Path(
        "/Users/milamarcheva/Desktop/maturational_grammar_induction/grammars"
    )
    def safe(value: object) -> str:
        return str(value).replace(" ", "").replace("/", "_").replace(".", "p")
    output_name = (
        f"{grammar_path.stem}"
        f"__lex-{safe(args.lexicalisation)}"
        f"__wp-{safe(args.weight_prod)}"
        f"__wl-{safe(args.weight_lex)}"
        f"__pp-{safe(args.pseudocount_prod)}"
        f"__pl-{safe(args.pseudocount_lex)}"
        f"{grammar_path.suffix}"
    )
    output_path = out_dir / output_name

    with grammar_path.open(encoding="utf-8") as f:
        rules = iter_rules(f)
    if not rules:
        raise SystemExit(f"No rules found in input grammar: {grammar_path}")

    productions, lex_rules, preterminals, lex_counts = split_rules(rules)
    if not preterminals:
        raise SystemExit("No preterminals found in input grammar.")

    vocab = read_vocab(sentences_path)
    if not vocab:
        raise SystemExit(f"No tokens found in sentence file: {sentences_path}")

    if args.lexicalisation == "uniform":
        lexical_rules = build_uniform_lexicon(preterminals, vocab)
    elif args.lexicalisation == "postagged":
        lexical_rules = build_postagged_lexicon(lex_rules, preterminals, vocab)
    elif args.lexicalisation == "primed_by_freq":
        lexical_rules = build_primed_by_freq_lexicon(lex_counts, preterminals, vocab)
    else:
        raise SystemExit(f"Unsupported lexicalisation: {args.lexicalisation}")

    write_grammar(
        output_path,
        productions,
        lexical_rules,
        args.weight_prod,
        args.pseudocount_prod,
        args.weight_lex,
        args.pseudocount_lex,
    )
    print(f"Wrote grammar to {output_path}")
    print(f"- Productions: {len(productions)}")
    print(f"- Preterminals: {len(preterminals)}")
    print(f"- Vocab size: {len(vocab)}")
    print(f"- Lexical rules: {len(lexical_rules)}")


if __name__ == "__main__":
    main()
