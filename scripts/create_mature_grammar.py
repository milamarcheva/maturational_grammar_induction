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
        nargs="+",
        required=True,
        help="Input grammar(s) with productions and lexicalisations.",
    )
    ap.add_argument(
        "--sentences",
        required=True,
        help="Sentence file (one sentence per line) to build vocab. Default: /Users/milamarcheva/Desktop/maturational_grammar_induction/yields/filtered_ctb_yields.txt",
        default = "/Users/milamarcheva/Desktop/maturational_grammar_induction/yields/filtered_ctb_yields.txt"
    )
    ap.add_argument(
        "--output",
        help="Output directory (default: maturational_grammar_induction/grammars/mature).",
    )
    ap.add_argument(
        "--lexicalisation",
        default=argparse.SUPPRESS,
        choices=["uniform", "postagged", "primed_by_freq"],
        help="Lexicalisation scheme to use (default: uniform).",
    )
    ap.add_argument(
        "--weight-prod",
        type=float,
        default=argparse.SUPPRESS,
        help="Weight for production rules (default: 1.0).",
    )
    ap.add_argument(
        "--weight-lex",
        type=float,
        default=argparse.SUPPRESS,
        help="Weight for lexical rules (default: 1.0).",
    )
    ap.add_argument(
        "--pseudocount-prod",
        nargs="+",
        type=float,
        default=argparse.SUPPRESS,
        help="Pseudocount(s) for production rules (default: 0.1).",
    )
    ap.add_argument(
        "--pseudocount-lex",
        nargs="+",
        type=float,
        default=argparse.SUPPRESS,
        help="Pseudocount(s) for lexical rules (default: 0.1).",
    )

    args = ap.parse_args()

    default_out_dir = Path(
        "/Users/milamarcheva/Desktop/maturational_grammar_induction/grammars/mature"
    )
    default_lexicalisation = "uniform"
    default_weight_prod = 1.0
    default_weight_lex = 1.0
    default_pseudocount_prod = 0.1
    default_pseudocount_lex = 0.1

    grammar_paths = [Path(path) for path in args.grammar]
    sentences_path = Path(args.sentences)
    out_dir = Path(args.output) if args.output else default_out_dir
    lexicalisation = getattr(args, "lexicalisation", default_lexicalisation)
    weight_prod = getattr(args, "weight_prod", default_weight_prod)
    weight_lex = getattr(args, "weight_lex", default_weight_lex)
    pseudocount_prod_values = getattr(args, "pseudocount_prod", None)
    pseudocount_lex_values = getattr(args, "pseudocount_lex", None)
    include_lex = hasattr(args, "lexicalisation")
    include_wp = hasattr(args, "weight_prod")
    include_wl = hasattr(args, "weight_lex")
    include_pp = hasattr(args, "pseudocount_prod")
    include_pl = hasattr(args, "pseudocount_lex")
    if pseudocount_prod_values is None and pseudocount_lex_values is None:
        pseudocount_values = [default_pseudocount_prod]
    elif pseudocount_prod_values is not None and pseudocount_lex_values is not None:
        if list(pseudocount_prod_values) != list(pseudocount_lex_values):
            raise SystemExit(
                "--pseudocount-prod and --pseudocount-lex must be identical when both are set."
            )
        pseudocount_values = list(pseudocount_prod_values)
    elif pseudocount_prod_values is not None:
        pseudocount_values = list(pseudocount_prod_values)
    else:
        pseudocount_values = list(pseudocount_lex_values)
    def safe(value: object) -> str:
        return str(value).replace(" ", "").replace("/", "_").replace(".", "p")
    vocab = read_vocab(sentences_path)
    if not vocab:
        raise SystemExit(f"No tokens found in sentence file: {sentences_path}")

    for grammar_path in grammar_paths:
        with grammar_path.open(encoding="utf-8") as f:
            rules = iter_rules(f)
        if not rules:
            raise SystemExit(f"No rules found in input grammar: {grammar_path}")

        productions, lex_rules, preterminals, lex_counts = split_rules(rules)
        if not preterminals:
            raise SystemExit(f"No preterminals found in input grammar: {grammar_path}")

        if lexicalisation == "uniform":
            lexical_rules = build_uniform_lexicon(preterminals, vocab)
        elif lexicalisation == "postagged":
            lexical_rules = build_postagged_lexicon(lex_rules, preterminals, vocab)
        elif lexicalisation == "primed_by_freq":
            lexical_rules = build_primed_by_freq_lexicon(lex_counts, preterminals, vocab)
        else:
            raise SystemExit(f"Unsupported lexicalisation: {lexicalisation}")

        for pseudocount in pseudocount_values:
            name_parts = [grammar_path.stem]
            if include_lex:
                name_parts.append(f"lex-{safe(lexicalisation)}")
            if include_wp:
                name_parts.append(f"wp-{safe(weight_prod)}")
            if include_wl:
                name_parts.append(f"wl-{safe(weight_lex)}")
            if include_pp:
                name_parts.append(f"pp-{safe(pseudocount)}")
            if include_pl:
                name_parts.append(f"pl-{safe(pseudocount)}")
            output_name = "__".join(name_parts) + grammar_path.suffix
            output_path = out_dir / output_name

            write_grammar(
                output_path,
                productions,
                lexical_rules,
                weight_prod,
                pseudocount,
                weight_lex,
                pseudocount,
            )
            print(f"Wrote grammar to {output_path}")
            print(f"- Productions: {len(productions)}")
            print(f"- Preterminals: {len(preterminals)}")
            print(f"- Vocab size: {len(vocab)}")
            print(f"- Lexical rules: {len(lexical_rules)}")


if __name__ == "__main__":
    main()
