#!/usr/bin/env python3
import argparse
from pathlib import Path
from nltk import PCFG
from nltk.parse import ViterbiParser
from nltk.grammar import Nonterminal, ProbabilisticProduction, PCFG


# def load_grammar(pcfg_path):
#     text = Path(pcfg_path).read_text(encoding="utf-8")
#     grammar = PCFG.fromstring(text)
#     return grammar

def load_grammar(path: str) -> PCFG:
    """
    Load a PCFG from a file in the format:

        LHS -> RHS1 RHS2 ... [prob]

    Works even when terminals contain quotes, e.g. "'s", "'d", etc.
    """
    productions = []

    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Expect: LHS -> RHS1 RHS2 ... [prob]
            if "->" not in line:
                continue  # skip malformed lines

            lhs_part, rhs_part = line.split("->", 1)
            lhs_str = lhs_part.strip()

            rhs_part = rhs_part.strip()
            # probability is in the last [ ... ]
            lb = rhs_part.rfind("[")
            rb = rhs_part.rfind("]")
            if lb == -1 or rb == -1 or rb < lb:
                raise ValueError(f"No probability found in line: {line}")

            prob_str = rhs_part[lb + 1:rb].strip()
            prob = float(prob_str)

            rhs_symbols_str = rhs_part[:lb].strip()
            rhs = []
            if rhs_symbols_str:
                for tok in rhs_symbols_str.split():
                    # If token is quoted (single or double), treat it as a terminal
                    if ((tok.startswith("'") and tok.endswith("'")) or
                        (tok.startswith('"') and tok.endswith('"'))):
                        rhs.append(tok[1:-1])  # strip the surrounding quotes
                    else:
                        rhs.append(Nonterminal(tok))

            lhs = Nonterminal(lhs_str)
            productions.append(ProbabilisticProduction(lhs, rhs, prob=prob))

    if not productions:
        raise ValueError(f"No productions read from grammar file: {path}")

    # Use the LHS of the first production as the start symbol by default
    start = productions[0].lhs()
    grammar = PCFG(start, productions)
    return grammar


def extract_tags_from_tree(tree):
    """
    Tree.pos() returns list of (word, tag).
    Here tag is directly N, V, A, etc.
    """
    pairs = tree.pos()
    return [tag for (word, tag) in pairs]

def write_fallback_tags(tags_out, words):
    if not tags_out:
        return
    tags_out.write(" ".join(["N"] * len(words)) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Decode sentences with NLTK Viterbi PCFG parser."
    )
    ap.add_argument(
        "-g", "--grammar", required=True,
        help="PCFG grammar file (output of io_to_nltk_pcfg.py)."
    )
    ap.add_argument(
        "-s", "--sentences", required=True,
        help="Test sentence file (one tokenised sentence per line)."
    )
    ap.add_argument(
        "--trees-out",
        help="Optional output file for bracketed parse trees."
    )
    ap.add_argument(
        "--tags-out",
        help="Optional output file for tag sequences (one line per sentence)."
    )
    args = ap.parse_args()

    grammar = load_grammar(args.grammar)
    parser = ViterbiParser(grammar)

    trees_out = open(args.trees_out, "w", encoding="utf-8") if args.trees_out else None
    tags_out = open(args.tags_out, "w", encoding="utf-8") if args.tags_out else None

    sent_path = Path(args.sentences)
    with sent_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            sent = line.strip()
            if not sent:
                continue
            words = sent.split()

            try:
                parses = list(parser.parse(words))
            except ValueError as e:
                print(f"[WARN] sentence {line_no}: parse error: {e}")
                if trees_out:
                    trees_out.write(f"; PARSE_ERROR: {sent}\n")
                write_fallback_tags(tags_out, words)
                continue

            if not parses:
                print(f"[WARN] sentence {line_no}: NO PARSE")
                if trees_out:
                    trees_out.write(f"; NO_PARSE: {sent}\n")
                write_fallback_tags(tags_out, words)
                continue

            best = parses[0]

            if trees_out:
                trees_out.write(str(best) + "\n")

            if tags_out:
                tags = extract_tags_from_tree(best)
                tags_out.write(" ".join(tags) + "\n")

    if trees_out:
        trees_out.close()
    if tags_out:
        tags_out.close()


if __name__ == "__main__":
    main()
