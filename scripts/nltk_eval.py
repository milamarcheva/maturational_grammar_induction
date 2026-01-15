#!/usr/bin/env python3
import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

try:
    from nltk.grammar import Nonterminal, ProbabilisticProduction, PCFG
    from nltk.parse import ViterbiParser
    from nltk.tree import Tree
except ImportError as exc:
    raise SystemExit(
        "NLTK is required for this script. Install with: pip install nltk"
    ) from exc


PROB_SUFFIX_RE = re.compile(r"\s*\((?:p|prob|logprob)=[^)]+\)\s*$")


def parse_weighted_rules(
    grammar_path: Path, weight_index: int
) -> List[Tuple[str, List[str], float]]:
    rules: List[Tuple[str, List[str], float]] = []
    with grammar_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            arrow_idx = None
            if "-->" in parts:
                arrow_idx = parts.index("-->")
            elif "->" in parts:
                arrow_idx = parts.index("->")
            if arrow_idx is None or arrow_idx == 0 or arrow_idx >= len(parts) - 1:
                continue
            lhs = parts[arrow_idx - 1]
            rhs = parts[arrow_idx + 1:]

            prefix = parts[: arrow_idx - 1]
            weights: List[float] = []
            for tok in prefix:
                try:
                    weights.append(float(tok))
                except ValueError:
                    weights = []
                    break
            if not weights:
                continue

            idx = weight_index if weight_index >= 0 else len(weights) + weight_index
            if idx < 0 or idx >= len(weights):
                idx = 0
            rules.append((lhs, rhs, weights[idx]))
    return rules


def normalize_rules(
    rules: Iterable[Tuple[str, List[str], float]]
) -> List[Tuple[str, List[str], float]]:
    mass: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for lhs, _, weight in rules:
        mass[lhs] += max(weight, 0.0)
        counts[lhs] += 1

    normalized: List[Tuple[str, List[str], float]] = []
    for lhs, rhs, weight in rules:
        denom = mass[lhs]
        if denom > 0:
            prob = max(weight, 0.0) / denom
        else:
            prob = 1.0 / max(counts[lhs], 1)
        normalized.append((lhs, rhs, prob))
    return normalized


def maybe_unquote(token: str) -> str:
    if len(token) >= 2:
        if token.startswith("'") and token.endswith("'"):
            return token[1:-1]
        if token.startswith('"') and token.endswith('"'):
            return token[1:-1]
    return token


def build_pcfg(
    rules: List[Tuple[str, List[str], float]],
    start_symbol: Optional[str],
) -> PCFG:
    nonterminals = {lhs for lhs, _, _ in rules}
    if not nonterminals:
        raise ValueError("No nonterminals found in grammar rules.")

    productions = []
    for lhs, rhs, prob in rules:
        rhs_symbols = [
            Nonterminal(tok) if tok in nonterminals else maybe_unquote(tok)
            for tok in rhs
        ]
        productions.append(
            ProbabilisticProduction(Nonterminal(lhs), rhs_symbols, prob=prob)
        )

    if start_symbol:
        start = Nonterminal(start_symbol)
    elif "ROOT" in nonterminals:
        start = Nonterminal("ROOT")
    else:
        start = Nonterminal(next(iter(nonterminals)))
    return PCFG(start, productions)

def build_parser(
    rules: List[Tuple[str, List[str], float]],
    start_symbol: Optional[str],
) -> ViterbiParser:
    grammar = build_pcfg(rules, start_symbol)
    return ViterbiParser(grammar)


def read_sentences(sent_path: Path) -> List[str]:
    sentences: List[str] = []
    with sent_path.open(encoding="utf-8") as f:
        for line in f:
            sentences.append(line.rstrip("\n"))
    return sentences


def collect_vocab(sentences: List[str]) -> set[str]:
    vocab: set[str] = set()
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        vocab.update(sent.split())
    return vocab


def prune_lexicon_rules(
    rules: List[Tuple[str, List[str], float]],
    vocab: set[str],
) -> Tuple[List[Tuple[str, List[str], float]], int]:
    nonterminals = {lhs for lhs, _, _ in rules}
    pruned: List[Tuple[str, List[str], float]] = []
    dropped = 0
    for lhs, rhs, weight in rules:
        if len(rhs) == 1 and rhs[0] not in nonterminals:
            word = maybe_unquote(rhs[0])
            if word not in vocab:
                dropped += 1
                continue
        pruned.append((lhs, rhs, weight))
    return pruned, dropped


def strip_prob_suffix(tree_str: str) -> str:
    return PROB_SUFFIX_RE.sub("", tree_str).strip()


def read_parse_records(parse_path: Path) -> List[Optional[Tree]]:
    records: List[Optional[Tree]] = []
    buffer: List[str] = []
    depth = 0
    with parse_path.open(encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if depth == 0 and not raw.strip():
                continue
            if depth == 0 and raw.lstrip().startswith(";"):
                records.append(None)
                continue
            buffer.append(raw)
            depth += raw.count("(") - raw.count(")")
            if depth == 0 and buffer:
                tree_str = strip_prob_suffix("\n".join(buffer))
                buffer = []
                if not tree_str:
                    records.append(None)
                    continue
                try:
                    records.append(Tree.fromstring(tree_str))
                except Exception:
                    records.append(None)
    if buffer:
        tree_str = strip_prob_suffix("\n".join(buffer))
        if tree_str:
            try:
                records.append(Tree.fromstring(tree_str))
            except Exception:
                records.append(None)
    return records


_WORKER_PARSER: Optional[ViterbiParser] = None


def _init_worker(
    rules: List[Tuple[str, List[str], float]],
    start_symbol: Optional[str],
) -> None:
    global _WORKER_PARSER
    _WORKER_PARSER = build_parser(rules, start_symbol)


def _parse_sentence_worker(
    sent: str,
) -> Tuple[Optional[Tree], str]:
    sent = sent.strip()
    if not sent:
        return None, "empty"
    words = sent.split()
    try:
        tree = next(_WORKER_PARSER.parse(words), None)
    except ValueError:
        return None, "error"
    if tree is None:
        return None, "no_parse"
    return tree, "ok"


def parse_sentences(
    parser: ViterbiParser,
    sentences: List[str],
    progress_every: int = 0,
    label: str = "",
) -> Tuple[List[Optional[Tree]], int, int]:
    parsed: List[Optional[Tree]] = []
    parse_errors = 0
    no_parse = 0
    total = len(sentences)
    for idx, sent in enumerate(sentences, start=1):
        if progress_every > 0 and (idx % progress_every == 0 or idx == total):
            prefix = f"{label} " if label else ""
            print(f"[progress] {prefix}{idx}/{total}", file=sys.stderr)
        sent = sent.strip()
        if not sent:
            parsed.append(None)
            continue
        words = sent.split()
        try:
            parse_iter = parser.parse(words)
            tree = next(parse_iter, None)
        except ValueError:
            parse_errors += 1
            parsed.append(None)
            continue
        if tree is None:
            no_parse += 1
        parsed.append(tree)
    return parsed, parse_errors, no_parse


def parse_sentences_parallel(
    rules: List[Tuple[str, List[str], float]],
    start_symbol: Optional[str],
    sentences: List[str],
    processes: int,
    progress_every: int = 0,
    label: str = "",
) -> Tuple[List[Optional[Tree]], int, int]:
    parsed: List[Optional[Tree]] = [None] * len(sentences)
    parse_errors = 0
    no_parse = 0
    total = len(sentences)
    if total == 0:
        return parsed, parse_errors, no_parse

    chunksize = max(1, total // max(processes * 4, 1))
    with ProcessPoolExecutor(
        max_workers=processes,
        initializer=_init_worker,
        initargs=(rules, start_symbol),
    ) as ex:
        for idx, (tree, status) in enumerate(
            ex.map(_parse_sentence_worker, sentences, chunksize=chunksize), start=1
        ):
            parsed[idx - 1] = tree
            if status == "error":
                parse_errors += 1
            elif status == "no_parse":
                no_parse += 1
            if progress_every > 0 and (idx % progress_every == 0 or idx == total):
                prefix = f"{label} " if label else ""
                print(f"[progress] {prefix}{idx}/{total}", file=sys.stderr)
    return parsed, parse_errors, no_parse


def tree_spans(
    tree: Tree, include_preterminals: bool, include_root: bool
) -> set[Tuple[str, int, int]]:
    spans: set[Tuple[str, int, int]] = set()
    total = len(tree.leaves())

    def walk(node: Tree, start: int, is_root: bool) -> int:
        if isinstance(node, str):
            return start + 1
        if len(node) == 1 and isinstance(node[0], str):
            end = start + 1
            if include_preterminals:
                spans.add((node.label(), start, end))
            return end
        cur = start
        for child in node:
            cur = walk(child, cur, False)
        end = cur
        if include_root or not is_root:
            spans.add((node.label(), start, end))
        return end

    walk(tree, 0, True)
    return spans


def safe_div(num: float, denom: float) -> float:
    return num / denom if denom else 0.0


def resolve_processes(requested: int) -> int:
    if requested == 0:
        return os.cpu_count() or 1
    return requested if requested > 1 else 1


def evaluate(
    sentences: List[str],
    predicted: List[Optional[Tree]],
    gold: List[Optional[Tree]],
    include_preterminals: bool,
    include_root: bool,
    progress_every: int = 0,
) -> dict[str, float]:
    total_sentences = len(sentences)
    gold_available = 0
    pred_available = 0
    evaluated = 0
    exact_match = 0
    total_pred = 0
    total_gold = 0
    total_correct = 0
    leaf_mismatch = 0

    for idx in range(total_sentences):
        if progress_every > 0:
            current = idx + 1
            if current % progress_every == 0 or current == total_sentences:
                print(f"[progress] eval {current}/{total_sentences}", file=sys.stderr)
        sent = sentences[idx].strip()
        if not sent:
            continue
        tokens = sent.split()

        gold_tree = gold[idx] if idx < len(gold) else None
        pred_tree = predicted[idx] if idx < len(predicted) else None

        if gold_tree is not None:
            gold_available += 1
        if pred_tree is not None:
            pred_available += 1
        if gold_tree is None or pred_tree is None:
            continue

        if gold_tree.leaves() != tokens or pred_tree.leaves() != tokens:
            leaf_mismatch += 1
            continue

        gold_spans = tree_spans(gold_tree, include_preterminals, include_root)
        pred_spans = tree_spans(pred_tree, include_preterminals, include_root)

        correct = len(gold_spans & pred_spans)
        total_correct += correct
        total_gold += len(gold_spans)
        total_pred += len(pred_spans)
        evaluated += 1
        if gold_spans == pred_spans:
            exact_match += 1

    precision = safe_div(total_correct, total_pred)
    recall = safe_div(total_correct, total_gold)
    f1 = safe_div(2 * precision * recall, precision + recall)
    accuracy = safe_div(exact_match, evaluated)

    return {
        "total_sentences": total_sentences,
        "gold_available": gold_available,
        "pred_available": pred_available,
        "evaluated": evaluated,
        "exact_match": exact_match,
        "leaf_mismatch": leaf_mismatch,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate induced PCFG parses against gold parses (or oracle grammar)."
        )
    )
    ap.add_argument(
        "--oracle-grammar",
        required=True,
        help="Oracle grammar file (used to build gold parses if --gold-parses is omitted).",
    )
    ap.add_argument(
        "--induced-grammar",
        required=True,
        help="Induced grammar file to evaluate.",
    )
    ap.add_argument(
        "--sentences",
        required=True,
        help="Sentence file (one tokenized sentence per line).",
    )
    ap.add_argument(
        "--gold-parses",
        help="Optional gold parse file (bracketed trees; one tree per sentence).",
    )
    ap.add_argument(
        "--start-symbol",
        help="Override grammar start symbol (default: ROOT if present).",
    )
    ap.add_argument(
        "--weight-index",
        type=int,
        default=0,
        help=(
            "Which numeric prefix to treat as rule weight "
            "(default: 0 = first number before LHS)."
        ),
    )
    ap.add_argument(
        "--include-preterminals",
        action="store_true",
        help="Include preterminal spans in precision/recall.",
    )
    ap.add_argument(
        "--include-root",
        action="store_true",
        help="Include the root span in precision/recall.",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Report progress every N sentences (default: 1000, 0 disables).",
    )
    ap.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Number of worker processes for parsing (default: 1, 0 = auto).",
    )
    ap.add_argument(
        "--prune-lexicon",
        action="store_true",
        help=(
            "Prune lexical rules to sentence vocabulary before decoding "
            "(renormalizes rule probabilities)."
        ),
    )

    args = ap.parse_args()

    sentences = read_sentences(Path(args.sentences))
    if not sentences:
        raise SystemExit("No sentences found.")

    oracle_rules = parse_weighted_rules(Path(args.oracle_grammar), args.weight_index)
    induced_rules = parse_weighted_rules(Path(args.induced_grammar), args.weight_index)
    if not oracle_rules:
        raise SystemExit("No oracle grammar rules found.")
    if not induced_rules:
        raise SystemExit("No induced grammar rules found.")

    if args.prune_lexicon:
        vocab = collect_vocab(sentences)
        oracle_rules, oracle_dropped = prune_lexicon_rules(oracle_rules, vocab)
        induced_rules, induced_dropped = prune_lexicon_rules(induced_rules, vocab)
        print(
            f"[info] pruned lexicon rules: oracle {oracle_dropped}, induced {induced_dropped}",
            file=sys.stderr,
        )

    normalized_oracle_rules = normalize_rules(oracle_rules)
    normalized_induced_rules = normalize_rules(induced_rules)
    processes = resolve_processes(args.processes)

    if processes > 1:
        predicted, pred_errors, pred_no_parse = parse_sentences_parallel(
            normalized_induced_rules,
            args.start_symbol,
            sentences,
            processes=processes,
            progress_every=args.progress_every,
            label="predicted",
        )
    else:
        induced_parser = build_parser(normalized_induced_rules, args.start_symbol)
        predicted, pred_errors, pred_no_parse = parse_sentences(
            induced_parser,
            sentences,
            progress_every=args.progress_every,
            label="predicted",
        )

    if args.gold_parses:
        gold = read_parse_records(Path(args.gold_parses))
        gold_source = "gold parse file"
        gold_errors = 0
        gold_no_parse = 0
        if len(gold) != len(sentences):
            print(
                f"[WARN] gold parse count ({len(gold)}) != sentence count ({len(sentences)})"
            )
    else:
        if processes > 1:
            gold, gold_errors, gold_no_parse = parse_sentences_parallel(
                normalized_oracle_rules,
                args.start_symbol,
                sentences,
                processes=processes,
                progress_every=args.progress_every,
                label="gold",
            )
        else:
            oracle_parser = build_parser(normalized_oracle_rules, args.start_symbol)
            gold, gold_errors, gold_no_parse = parse_sentences(
                oracle_parser,
                sentences,
                progress_every=args.progress_every,
                label="gold",
            )
        gold_source = "oracle grammar"

    stats = evaluate(
        sentences,
        predicted,
        gold,
        include_preterminals=args.include_preterminals,
        include_root=args.include_root,
        progress_every=args.progress_every,
    )

    print("Evaluation summary")
    print(f"- sentences: {stats['total_sentences']}")
    print(f"- gold source: {gold_source}")
    print(f"- gold parses available: {stats['gold_available']}")
    print(f"- predicted parses available: {stats['pred_available']}")
    print(f"- evaluated: {stats['evaluated']}")
    print(f"- exact match accuracy: {stats['accuracy']:.6f}")
    print(f"- precision: {stats['precision']:.6f}")
    print(f"- recall: {stats['recall']:.6f}")
    print(f"- f1: {stats['f1']:.6f}")
    print(f"- leaf mismatches: {stats['leaf_mismatch']}")
    print(
        f"- predicted parse errors: {pred_errors}, no-parse: {pred_no_parse}"
    )
    if args.gold_parses:
        print("- gold parse errors: 0, no-parse: 0")
    else:
        print(
            f"- gold parse errors: {gold_errors}, no-parse: {gold_no_parse}"
        )


if __name__ == "__main__":
    main()
