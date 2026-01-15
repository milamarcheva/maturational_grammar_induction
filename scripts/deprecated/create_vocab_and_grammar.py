#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple


def read_sentences_build_vocab(sent_path):
    vocab = set()
    with Path(sent_path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for tok in line.split():
                if tok:
                    vocab.add(tok)
    return sorted(vocab)


def cmd_vocab(args):
    vocab = read_sentences_build_vocab(args.sentences)
    out_path = Path(args.vocab)
    with out_path.open("w", encoding="utf-8") as f:
        for w in vocab:
            f.write(w + "\n")
    print(f"Vocab written to: {out_path}")


def parse_prime_file(
    path: Optional[Path],
    lexicalised: bool,
    default_weight: Optional[float],
) -> Dict[str, Dict[str, Tuple[float, Optional[float]]]]:
    """
    Returns mapping word -> tag -> (bias, weight or None).
    prime file format (whitespace separated):
        word TAG bias [weight]
    Bias is required; weight is optional.
    """
    primes: Dict[str, Dict[str, Tuple[float, Optional[float]]]] = {}
    if not path:
        return primes
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError("Prime file lines must be: word TAG bias [weight]")
            word, tag, bias_s = parts[:3]
            bias = float(bias_s)
            weight = float(parts[3]) if len(parts) >= 4 else default_weight
            tag_key = "L_" + tag if lexicalised and not tag.startswith("L_") else tag
            primes.setdefault(word, {})[tag_key] = (bias, weight)
    return primes


def cmd_lexicon(args):
    vocab_path = Path(args.vocab)
    out_path = Path(args.lexicon)
    weight = args.weight
    bias = args.bias
    prime_map = parse_prime_file(
        Path(args.prime_file) if args.prime_file else None,
        args.lexicalised,
        args.prime_weight,
    )

    # read tags
    if args.tags_file:
        with Path(args.tags_file).open(encoding="utf-8") as tf:
            tags = [t.strip() for t in tf if t.strip()]
    else:
        tags = args.tags

    def lex_tag(tag):
        if args.lexicalised:
            # lexicalised tag name: L_ + TAG (e.g. N -> L_N, ASP -> L_ASP)
            return "L_" + tag
        else:
            return tag

    with vocab_path.open(encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            w = line.strip()
            if not w:
                continue
            for tag in tags:
                lt = lex_tag(tag)
                if w in prime_map and lt in prime_map[w]:
                    pbias, pweight = prime_map[w][lt]
                    fout.write(f"{pweight or weight} {pbias}  {lt} --> {w}\n")
                else:
                    fout.write(f"{weight} {bias}  {lt} --> {w}\n")

    print(f"Lexicon written to: {out_path}")


def rewrite_bias_line(line: str, new_bias: Optional[float]) -> str:
    """Override bias (second number) if the line looks like a rule."""
    if new_bias is None:
        return line
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "-->" not in line:
        return line
    left, right = line.split("-->", 1)
    tokens = left.split()
    if len(tokens) < 3:
        return line
    try:
        float(tokens[0])
        float(tokens[1])
    except ValueError:
        return line
    tokens[1] = str(new_bias)
    return f"{' '.join(tokens)} -->{right}"


def rewrite_lex_line(
    line: str,
    lex_bias: Optional[float],
    primes: Dict[str, Dict[str, Tuple[float, Optional[float]]]],
    lexicalised: bool,
) -> str:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "-->" not in line:
        return line
    left, right = line.split("-->", 1)
    tokens = left.split()
    if len(tokens) < 3:
        return line
    try:
        weight = float(tokens[0])
        bias = float(tokens[1])
    except ValueError:
        return line
    parent = tokens[2]
    word = right.strip()
    prime_tag = parent
    if lexicalised and not parent.startswith("L_"):
        prime_tag = "L_" + parent
    if word in primes and prime_tag in primes[word]:
        pbias, pweight = primes[word][prime_tag]
        bias = pbias
        if pweight is not None:
            weight = pweight
    elif lex_bias is not None:
        bias = lex_bias
    tokens[0] = str(weight)
    tokens[1] = str(bias)
    return f"{' '.join(tokens)} --> {word}\n"


def cmd_combine(args):
    prod_path = Path(args.productions)
    lex_path = Path(args.lexicon)
    out_path = Path(args.output)
    prime_map = parse_prime_file(
        Path(args.prime_file) if args.prime_file else None,
        args.lexicalised,
        args.prime_weight,
    )

    with out_path.open("w", encoding="utf-8") as fout:
        with prod_path.open(encoding="utf-8") as fp:
            for line in fp:
                fout.write(rewrite_bias_line(line, args.prod_bias))
        fout.write("\n")
        with lex_path.open(encoding="utf-8") as fl:
            for line in fl:
                fout.write(rewrite_lex_line(line, args.lex_bias, prime_map, args.lexicalised))

    print(f"Combined grammar written to: {out_path}")


def cmd_build(args):
    """
    One-shot:
      - read sentences → vocab
      - build lexicon from vocab & tags
      - combine productions + lexicon into final grammar
    """
    sentences = args.sentences
    productions = Path(args.productions)
    output = Path(args.output)
    weight = args.weight
    bias = args.bias

    # tags
    if args.tags_file:
        with Path(args.tags_file).open(encoding="utf-8") as tf:
            tags = [t.strip() for t in tf if t.strip()]
    else:
        tags = args.tags

    if not tags:
        raise ValueError("No tags provided. Use --tags or --tags-file.")

    # 1) build vocab from sentences
    vocab = read_sentences_build_vocab(sentences)

    def lex_tag(tag):
        if args.lexicalised:
            return "L_" + tag
        else:
            return tag

    # 2+3) write productions + generated lexicon directly to final grammar
    prime_map = parse_prime_file(
        Path(args.prime_file) if args.prime_file else None,
        args.lexicalised,
        args.prime_weight,
    )

    with output.open("w", encoding="utf-8") as fout:
        # productions
        with productions.open(encoding="utf-8") as fp:
            for line in fp:
                fout.write(rewrite_bias_line(line, args.prod_bias))
        fout.write("\n")
        # lexicon
        for w in vocab:
            for tag in tags:
                lt = lex_tag(tag)
                if w in prime_map and lt in prime_map[w]:
                    pbias, pweight = prime_map[w][lt]
                    fout.write(f"{pweight or weight} {pbias}  {lt} --> {w}\n")
                else:
                    fout.write(f"{weight} {bias}  {lt} --> {w}\n")

    print(f"Combined grammar (productions + lexicon) written to: {output}")
    print(f"- Tags used: {', '.join(tags)}")
    print(f"- Lexicalised: {args.lexicalised}")
    print(f"- Vocab size: {len(vocab)}")


def main():
    parser = argparse.ArgumentParser(
        description="Tools for building vocab, lexicon, and combined grammar from sentences and tags."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1) vocab: sentences -> vocab
    p_vocab = subparsers.add_parser(
        "vocab",
        help="Extract vocab from sentences (one sentence per line)."
    )
    p_vocab.add_argument(
        "-i", "--sentences", required=True,
        help="Input sentence file (one sentence per line)."
    )
    p_vocab.add_argument(
        "-o", "--vocab", required=True,
        help="Output vocab file (one word per line)."
    )
    p_vocab.set_defaults(func=cmd_vocab)

    # 2) lexicon: vocab + tags -> lexicon
    p_lex = subparsers.add_parser(
        "lexicon",
        help="Create lexicon from vocab and tags."
    )
    p_lex.add_argument(
        "-i", "--vocab", required=True,
        help="Input vocab file (one word per line)."
    )
    p_lex.add_argument(
        "-o", "--lexicon", required=True,
        help="Output lexicon file."
    )
    p_lex.add_argument(
        "--tags", nargs="+", default=[],
        help="List of tags (e.g. --tags N V A)."
    )
    p_lex.add_argument(
        "--tags-file",
        help="File with one tag per line (alternative to --tags)."
    )
    p_lex.add_argument(
        "--weight", type=float, default=1.0,
        help="Initial weight for each lexical rule (default: 1.0)."
    )
    p_lex.add_argument(
        "--bias", type=float, default=0.3,
        help="Pseudo-count (bias) for each lexical rule (default: 0.3)."
    )
    p_lex.add_argument(
        "--lexicalised", action="store_true",
        help="If set, lexical tags are 'L_'+TAG (e.g. N->L_N). Otherwise the tag itself is used."
    )
    p_lex.add_argument(
        "--prime-file",
        help="Optional file with word TAG bias [weight] for priming specific lexical entries."
    )
    p_lex.add_argument(
        "--prime-weight", type=float,
        help="Weight to use for primed entries when the prime file omits it (default: use --weight)."
    )
    p_lex.set_defaults(func=cmd_lexicon)

    # 3) combine: productions + lexicon -> grammar
    p_comb = subparsers.add_parser(
        "combine",
        help="Combine productions and lexicon into one grammar."
    )
    p_comb.add_argument(
        "-p", "--productions", required=True,
        help="File with non-lexical productions (.lt)."
    )
    p_comb.add_argument(
        "-l", "--lexicon", required=True,
        help="Lexicon file to append."
    )
    p_comb.add_argument(
        "-o", "--output", required=True,
        help="Output combined grammar file."
    )
    p_comb.add_argument(
        "--prod-bias", type=float,
        help="Override bias for all production rules (leave unset to keep existing bias values)."
    )
    p_comb.add_argument(
        "--lex-bias", type=float,
        help="Override bias for all lexicon rules (leave unset to keep existing bias values)."
    )
    p_comb.add_argument(
        "--prime-file",
        help="Optional file with word TAG bias [weight] for priming specific lexical entries."
    )
    p_comb.add_argument(
        "--lexicalised", action="store_true",
        help="If set, prime tags are matched against lexicalised tags (L_TAG)."
    )
    p_comb.add_argument(
        "--prime-weight", type=float,
        help="Weight to use for primed entries when the prime file omits it (default: leave weight as-is)."
    )
    p_comb.set_defaults(func=cmd_combine)

    # 4) build: sentences + tags + productions -> final grammar (everything in one go)
    p_build = subparsers.add_parser(
        "build",
        help="Build combined grammar directly from sentences, tags, and productions."
    )
    p_build.add_argument(
        "-s", "--sentences", required=True,
        help="Input sentence file (one sentence per line)."
    )
    p_build.add_argument(
        "-p", "--productions", required=True,
        help="Non-lexical productions file (.lt)."
    )
    p_build.add_argument(
        "-o", "--output", required=True,
        help="Output combined grammar file (.lt)."
    )
    p_build.add_argument(
        "--tags", nargs="+", default=[],
        help="List of tags (e.g. --tags N V A)."
    )
    p_build.add_argument(
        "--tags-file",
        help="File with one tag per line (alternative to --tags)."
    )
    p_build.add_argument(
        "--weight", type=float, default=1.0,
        help="Initial weight for each lexical rule (default: 1.0)."
    )
    p_build.add_argument(
        "--bias", type=float, default=0.3,
        help="Pseudo-count (bias) for each lexical rule (default: 0.3)."
    )
    p_build.add_argument(
        "--prod-bias", type=float,
        help="Override bias for all production rules (leave unset to keep existing bias values)."
    )
    p_build.add_argument(
        "--lexicalised", action="store_true",
        help="If set, lexical tags are 'L_'+TAG (e.g. N->L_N, ASP->L_ASP)."
    )
    p_build.add_argument(
        "--prime-file",
        help="Optional file with word TAG bias [weight] for priming specific lexical entries."
    )
    p_build.add_argument(
        "--prime-weight", type=float,
        help="Weight to use for primed entries when the prime file omits it (default: use --weight)."
    )
    p_build.set_defaults(func=cmd_build)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
