#!/usr/bin/env python3
import argparse
from pathlib import Path


def parse_io_grammar(io_path):
    """
    Parse IO's final grammar lines from an output file.

    Expect lines like:
      0.432835 N --> N V
      0.000273945 V --> 'd

    Ignore lines that don't start with a float.
    """
    rules = []
    with Path(io_path).open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            parts = s.split()
            try:
                p = float(parts[0])
            except ValueError:
                continue  # skip non-rule lines
            if len(parts) < 4 or parts[2] != "-->":
                continue
            lhs = parts[1]
            rhs = parts[3:]
            rules.append({"lhs": lhs, "rhs": rhs, "prob": p})
    return rules


def renormalize_by_lhs(rules):
    """Ensure probs per LHS sum to 1."""
    mass = {}
    for r in rules:
        mass.setdefault(r["lhs"], 0.0)
        mass[r["lhs"]] += r["prob"]

    for r in rules:
        Z = mass[r["lhs"]]
        if Z > 0:
            r["prob"] = r["prob"] / Z
        else:
            r["prob"] = 1e-12


def escape_terminal(tok: str) -> str:
    """
    Return a fully quoted terminal literal for NLTK PCFG.

    Strategy:
      - If the token contains a single quote (like 'd, 're), wrap it in double quotes:
            "'d"
        (so NLTK sees the terminal as `'d`).
      - Otherwise, wrap it in single quotes:
            'dog'
    """

    # First escape backslashes inside the token itself
    tok = tok.replace("\\", "\\\\")

    if "'" in tok:
        # Use double quotes outside; escape any double quotes inside if they ever appear.
        inner = tok.replace('"', '\\"')
        return f"\"{inner}\""
    else:
        # Use single quotes outside; we don't expect single quotes inside now.
        return f"'{tok}'"


def rules_to_pcfg_text(rules, nonterminals):
    lines = []
    nonterminals = set(nonterminals)

    for r in rules:
        lhs = r["lhs"]
        rhs = r["rhs"]
        prob = r["prob"]

        if len(rhs) == 1 and rhs[0] not in nonterminals:
            # lexical rule
            word_literal = escape_terminal(rhs[0])
            line = f"{lhs} -> {word_literal} [{prob}]"
        else:
            # structural rule
            rhs_str = " ".join(rhs)
            line = f"{lhs} -> {rhs_str} [{prob}]"

        lines.append(line)

    return "\n".join(lines)
    

def main():
    ap = argparse.ArgumentParser(
        description="Convert IO-learned grammar output to NLTK PCFG format."
    )
    ap.add_argument(
        "-i", "--input", required=True,
        help="IO output file containing final grammar (rule lines with probabilities)."
    )
    ap.add_argument(
        "-o", "--output", required=True,
        help="Output PCFG file for NLTK."
    )
    ap.add_argument(
        "--no-renorm", action="store_true",
        help="Skip renormalization by LHS (default: renormalize)."
    )
    ap.add_argument(
        "--nonterminals", nargs="+",
        default=["S1", "S", "N", "V", "A"],
        help="List of nonterminal symbols (default: S1 S N V A)."
    )

    args = ap.parse_args()

    rules = parse_io_grammar(args.input)
    if not rules:
        raise SystemExit("No rules found in input. Check the file format.")

    if not args.no_renorm:
        renormalize_by_lhs(rules)

    pcfg_text = rules_to_pcfg_text(rules, args.nonterminals)

    out_path = Path(args.output)
    out_path.write_text(pcfg_text, encoding="utf-8")

    print(f"Wrote NLTK PCFG grammar to: {out_path}")
    print(f"Number of rules: {len(rules)}")
    print("--- sample ---")
    for line in pcfg_text.splitlines()[:10]:
        print(line)


if __name__ == "__main__":
    main()
