import csv
import os
import sys

def read_pcfg_file(path):
    """
    Reads a PCFG rule file of the form:
      PROB <whitespace> RULE
    Returns a dict: {rule: prob}
    """
    rules = {}
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                prob = float(parts[0])
            except ValueError:
                raise ValueError(f"{path}:{lineno} — cannot parse probability")

            rule = " ".join(parts[1:])
            rules[rule] = prob
    return rules


def is_nonterminal(sym: str) -> bool:
    if not sym:
        return False
    if not sym[0].isupper():
        return False
    return all(c.isupper() or c.isdigit() or c == "_" for c in sym)


def is_lexical_rule(rule: str) -> bool:
    if "-->" not in rule:
        return False
    _, rhs = rule.split("-->", 1)
    rhs = rhs.split("#", 1)[0].strip()
    if not rhs:
        return False
    tokens = rhs.split()
    if len(tokens) != 1:
        return False
    sym = tokens[0]
    if sym.startswith(("'", '"')):
        return True
    return not is_nonterminal(sym)


def main(f1, f2, out_csv):
    d1 = read_pcfg_file(f1)
    d2 = read_pcfg_file(f2)

    name1 = os.path.basename(f1)
    name2 = os.path.basename(f2)

    all_rules = sorted(set(d1) | set(d2))
    productions = [r for r in all_rules if not is_lexical_rule(r)]
    lexicals = [r for r in all_rules if is_lexical_rule(r)]

    with open(out_csv, "w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["rule", name1, name2])

        for rule in productions + lexicals:
            writer.writerow([
                rule,
                d1.get(rule, ""),
                d2.get(rule, "")
            ])


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(
            "Usage: python compare_pcfg_probs.py file1 file2 output.csv"
        )

    main(sys.argv[1], sys.argv[2], sys.argv[3])
