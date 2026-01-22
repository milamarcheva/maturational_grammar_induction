#!/usr/bin/env python3
import argparse
import math
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
    try:
        from nltk.parse.pchart import InsideChartParser
    except Exception:
        InsideChartParser = None
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


def choose_start_symbol(
    rules: List[Tuple[str, List[str], float]],
    start_symbol: Optional[str],
) -> str:
    nonterminals = {lhs for lhs, _, _ in rules}
    if not nonterminals:
        raise ValueError("No nonterminals found in grammar rules.")
    if start_symbol:
        return start_symbol
    if "ROOT" in nonterminals:
        return "ROOT"
    return next(iter(nonterminals))


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

    start_label = choose_start_symbol(rules, start_symbol)
    start = Nonterminal(start_label)
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


def predicted_cache_path(
    grammar_path: Path,
    sentences_path: Path,
    start_symbol: Optional[str],
    weight_index: int,
    pruned: bool,
) -> Path:
    parts = [sentences_path.stem, f"w{weight_index}"]
    if start_symbol:
        parts.append(f"start-{start_symbol}")
    if pruned:
        parts.append("pruned")
    name = "predicted__" + "__".join(parts) + ".txt"
    cache_dir = grammar_path.parent / grammar_path.stem
    return cache_dir / name


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


def rules_by_lhs(
    rules: List[Tuple[str, List[str], float]]
) -> dict[str, dict[Tuple[str, ...], float]]:
    by_lhs: dict[str, dict[Tuple[str, ...], float]] = defaultdict(dict)
    for lhs, rhs, prob in rules:
        rhs_key = tuple(maybe_unquote(tok) for tok in rhs)
        by_lhs[lhs][rhs_key] = by_lhs[lhs].get(rhs_key, 0.0) + prob
    return by_lhs


def find_preterminal_lhs(
    rules_a: Optional[List[Tuple[str, List[str], float]]],
    rules_b: Optional[List[Tuple[str, List[str], float]]],
) -> set[str]:
    combined: List[Tuple[str, List[str], float]] = []
    if rules_a:
        combined.extend(rules_a)
    if rules_b:
        combined.extend(rules_b)
    nonterminals = {lhs for lhs, _, _ in combined}
    if not nonterminals:
        return set()
    preterminals = set(nonterminals)
    for lhs, rhs, _ in combined:
        if len(rhs) != 1 or rhs[0] in nonterminals:
            preterminals.discard(lhs)
    return preterminals


def jsd_divergence(
    p: dict[Tuple[str, ...], float],
    q: dict[Tuple[str, ...], float],
    log_base: float,
) -> float:
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    log_denom = math.log(log_base)
    total = 0.0
    for key in keys:
        pv = p.get(key, 0.0)
        qv = q.get(key, 0.0)
        m = 0.5 * (pv + qv)
        if pv > 0:
            total += 0.5 * pv * (math.log(pv / m) / log_denom)
        if qv > 0:
            total += 0.5 * qv * (math.log(qv / m) / log_denom)
    return total


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


def write_parse_records(
    parse_path: Path,
    sentences: List[str],
    trees: List[Optional[Tree]],
) -> None:
    lines: List[str] = []
    for sent, tree in zip(sentences, trees):
        sent = sent.strip()
        if not sent:
            lines.append("; EMPTY")
        elif tree is None:
            lines.append(f"; NO_PARSE: {sent}")
        else:
            lines.append(str(tree))
    parse_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


_WORKER_PARSER: Optional[ViterbiParser] = None
_WORKER_LOGLIK_PARSER = None
_WORKER_LOGLIK_START = None
_WORKER_LOGLIK_MODE = "viterbi"


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


def tree_probability(tree: Tree) -> Optional[float]:
    if hasattr(tree, "prob") and callable(tree.prob):
        return tree.prob()
    if hasattr(tree, "probability") and callable(tree.probability):
        return tree.probability()
    return None


def edge_probability(edge) -> Optional[float]:
    if hasattr(edge, "prob") and callable(edge.prob):
        return edge.prob()
    if hasattr(edge, "probability") and callable(edge.probability):
        return edge.probability()
    return None


def sentence_loglik_viterbi(
    parser: ViterbiParser,
    sentence: str,
) -> Tuple[Optional[float], str]:
    sentence = sentence.strip()
    if not sentence:
        return None, "empty"
    words = sentence.split()
    try:
        tree = next(parser.parse(words), None)
    except ValueError:
        return None, "error"
    if tree is None:
        return None, "no_parse"
    prob = tree_probability(tree)
    if prob is None or prob <= 0:
        return None, "no_parse"
    return math.log(prob), "ok"


def sentence_loglik_inside(
    parser,
    start_symbol: str,
    sentence: str,
) -> Tuple[Optional[float], str]:
    sentence = sentence.strip()
    if not sentence:
        return None, "empty"
    words = sentence.split()
    try:
        chart = parser.chart_parse(words)
    except ValueError:
        return None, "error"
    if chart is None:
        return None, "no_parse"
    start_nt = Nonterminal(start_symbol)
    total_prob = 0.0
    for edge in chart.edges():
        if not edge.is_complete():
            continue
        if edge.lhs() != start_nt:
            continue
        if edge.span() != (0, len(words)):
            continue
        prob = edge_probability(edge)
        if prob:
            total_prob += prob
    if total_prob <= 0:
        return None, "no_parse"
    return math.log(total_prob), "ok"


def resolve_loglik_mode(mode: str) -> str:
    if mode == "inside" and InsideChartParser is None:
        print(
            "[WARN] InsideChartParser not available; falling back to viterbi log-likelihood.",
            file=sys.stderr,
        )
        return "viterbi"
    return mode


def _init_loglik_worker(
    rules: List[Tuple[str, List[str], float]],
    start_symbol: Optional[str],
    mode: str,
) -> None:
    global _WORKER_LOGLIK_PARSER
    global _WORKER_LOGLIK_START
    global _WORKER_LOGLIK_MODE
    _WORKER_LOGLIK_MODE = mode
    _WORKER_LOGLIK_START = choose_start_symbol(rules, start_symbol)
    grammar = build_pcfg(rules, start_symbol)
    if _WORKER_LOGLIK_MODE == "inside" and InsideChartParser is not None:
        _WORKER_LOGLIK_PARSER = InsideChartParser(grammar)
    else:
        _WORKER_LOGLIK_PARSER = ViterbiParser(grammar)


def _loglik_worker(sentence: str) -> Tuple[Optional[float], str]:
    if _WORKER_LOGLIK_MODE == "inside" and InsideChartParser is not None:
        return sentence_loglik_inside(
            _WORKER_LOGLIK_PARSER, _WORKER_LOGLIK_START, sentence
        )
    return sentence_loglik_viterbi(_WORKER_LOGLIK_PARSER, sentence)


def compute_logliks(
    rules: List[Tuple[str, List[str], float]],
    start_symbol: Optional[str],
    sentences: List[str],
    mode: str,
    processes: int,
    progress_every: int = 0,
) -> Tuple[List[Optional[float]], int, int, int]:
    logliks: List[Optional[float]] = []
    parse_errors = 0
    no_parse = 0
    empty = 0
    total = len(sentences)
    if total == 0:
        return logliks, parse_errors, no_parse, empty

    if processes > 1:
        chunksize = max(1, total // max(processes * 4, 1))
        with ProcessPoolExecutor(
            max_workers=processes,
            initializer=_init_loglik_worker,
            initargs=(rules, start_symbol, mode),
        ) as ex:
            for idx, (loglik, status) in enumerate(
                ex.map(_loglik_worker, sentences, chunksize=chunksize), start=1
            ):
                logliks.append(loglik)
                if status == "error":
                    parse_errors += 1
                elif status == "no_parse":
                    no_parse += 1
                elif status == "empty":
                    empty += 1
                if progress_every > 0 and (idx % progress_every == 0 or idx == total):
                    print(f"[progress] loglik {idx}/{total}", file=sys.stderr)
        return logliks, parse_errors, no_parse, empty

    start_label = choose_start_symbol(rules, start_symbol)
    if mode == "inside" and InsideChartParser is not None:
        parser = InsideChartParser(build_pcfg(rules, start_symbol))
    else:
        parser = build_parser(rules, start_symbol)
    for idx, sentence in enumerate(sentences, start=1):
        loglik, status = (
            sentence_loglik_inside(parser, start_label, sentence)
            if mode == "inside" and InsideChartParser is not None
            else sentence_loglik_viterbi(parser, sentence)
        )
        logliks.append(loglik)
        if status == "error":
            parse_errors += 1
        elif status == "no_parse":
            no_parse += 1
        elif status == "empty":
            empty += 1
        if progress_every > 0 and (idx % progress_every == 0 or idx == total):
            print(f"[progress] loglik {idx}/{total}", file=sys.stderr)
    return logliks, parse_errors, no_parse, empty


def tree_spans(
    tree: Tree, include_preterminals: bool, include_root: bool, labeled: bool
) -> set[Tuple]:
    spans: set[Tuple] = set()
    total = len(tree.leaves())

    def walk(node: Tree, start: int, is_root: bool) -> int:
        if isinstance(node, str):
            return start + 1
        if len(node) == 1 and isinstance(node[0], str):
            end = start + 1
            if include_preterminals:
                spans.add((node.label(), start, end) if labeled else (start, end))
            return end
        cur = start
        for child in node:
            cur = walk(child, cur, False)
        end = cur
        if include_root or not is_root:
            spans.add((node.label(), start, end) if labeled else (start, end))
        return end

    walk(tree, 0, True)
    return spans


def unlabeled_spans_from_labeled(spans: set[Tuple[str, int, int]]) -> set[Tuple[int, int]]:
    return {(start, end) for _, start, end in spans}


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
    labeled: bool,
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

        gold_spans = tree_spans(gold_tree, include_preterminals, include_root, labeled)
        pred_spans = tree_spans(pred_tree, include_preterminals, include_root, labeled)

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


def evaluate_both(
    sentences: List[str],
    predicted: List[Optional[Tree]],
    gold: List[Optional[Tree]],
    include_preterminals: bool,
    include_root: bool,
    progress_every: int = 0,
) -> dict[str, object]:
    total_sentences = len(sentences)
    gold_available = 0
    pred_available = 0
    evaluated = 0
    leaf_mismatch = 0

    total_pred_u = 0
    total_gold_u = 0
    total_correct_u = 0
    exact_match_u = 0

    total_pred_l = 0
    total_gold_l = 0
    total_correct_l = 0
    exact_match_l = 0

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

        gold_labeled = tree_spans(gold_tree, include_preterminals, include_root, True)
        pred_labeled = tree_spans(pred_tree, include_preterminals, include_root, True)
        gold_unlabeled = unlabeled_spans_from_labeled(gold_labeled)
        pred_unlabeled = unlabeled_spans_from_labeled(pred_labeled)

        total_correct_l += len(gold_labeled & pred_labeled)
        total_gold_l += len(gold_labeled)
        total_pred_l += len(pred_labeled)
        if gold_labeled == pred_labeled:
            exact_match_l += 1

        total_correct_u += len(gold_unlabeled & pred_unlabeled)
        total_gold_u += len(gold_unlabeled)
        total_pred_u += len(pred_unlabeled)
        if gold_unlabeled == pred_unlabeled:
            exact_match_u += 1

        evaluated += 1

    def finalize_metrics(
        total_correct: int,
        total_pred: int,
        total_gold: int,
        exact_match: int,
    ) -> dict[str, float]:
        precision = safe_div(total_correct, total_pred)
        recall = safe_div(total_correct, total_gold)
        f1 = safe_div(2 * precision * recall, precision + recall)
        accuracy = safe_div(exact_match, evaluated)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "exact_match": exact_match,
            "total_pred": total_pred,
            "total_gold": total_gold,
            "total_correct": total_correct,
        }

    return {
        "total_sentences": total_sentences,
        "gold_available": gold_available,
        "pred_available": pred_available,
        "evaluated": evaluated,
        "leaf_mismatch": leaf_mismatch,
        "unlabeled": finalize_metrics(
            total_correct_u, total_pred_u, total_gold_u, exact_match_u
        ),
        "labeled": finalize_metrics(
            total_correct_l, total_pred_l, total_gold_l, exact_match_l
        ),
    }


def print_parse_eval(
    stats: dict[str, object],
    gold_source: str,
    cache_path: Optional[Path],
    cached_predicted: bool,
    pred_errors: int,
    pred_no_parse: int,
    gold_errors: int,
    gold_no_parse: int,
    label: Optional[str] = None,
) -> None:
    header = "Parse evaluation" if not label else f"Parse evaluation ({label})"
    print(header)
    print(f"- sentences: {stats['total_sentences']}")
    print(f"- gold source: {gold_source}")
    print(f"- gold parses available: {stats['gold_available']}")
    print(f"- predicted parses available: {stats['pred_available']}")
    print(f"- evaluated: {stats['evaluated']}")
    print("Unlabeled spans")
    print(f"- exact match accuracy: {stats['unlabeled']['accuracy']:.6f}")
    print(f"- precision: {stats['unlabeled']['precision']:.6f}")
    print(f"- recall: {stats['unlabeled']['recall']:.6f}")
    print(f"- f1: {stats['unlabeled']['f1']:.6f}")
    print("Labeled spans")
    print(f"- exact match accuracy: {stats['labeled']['accuracy']:.6f}")
    print(f"- precision: {stats['labeled']['precision']:.6f}")
    print(f"- recall: {stats['labeled']['recall']:.6f}")
    print(f"- f1: {stats['labeled']['f1']:.6f}")
    print(f"- leaf mismatches: {stats['leaf_mismatch']}")
    if cache_path and cached_predicted:
        print(f"- predicted cache: {cache_path}")
        print(f"- predicted parse errors: n/a, no-parse: {pred_no_parse}")
    else:
        print(f"- predicted parse errors: {pred_errors}, no-parse: {pred_no_parse}")
    print(f"- gold parse errors: {gold_errors}, no-parse: {gold_no_parse}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate induced PCFG parses against gold parses (or oracle grammar)."
        )
    )
    ap.add_argument(
        "--oracle-grammar",
        help="Oracle grammar file (used to build gold parses if --gold-parses is omitted).",
    )
    ap.add_argument(
        "--induced-grammar",
        required=True,
        help="Induced grammar file to evaluate.",
    )
    ap.add_argument(
        "--sentences",
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
        "--labeled",
        action="store_true",
        help="Deprecated: parse metrics now report both labeled and unlabeled spans.",
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
    ap.add_argument(
        "--no-pred-cache",
        action="store_true",
        help="Disable caching for predicted parses.",
    )
    ap.add_argument(
        "--refresh-pred-cache",
        action="store_true",
        help="Recompute predicted parses even if a cache exists.",
    )
    ap.add_argument(
        "--eval-parse",
        action="store_true",
        help="Run parse evaluation (accuracy/precision/recall/F1).",
    )
    ap.add_argument(
        "--eval-jsd",
        action="store_true",
        help="Run Jensen-Shannon divergence over grammar rule distributions.",
    )
    ap.add_argument(
        "--eval-loglik",
        action="store_true",
        help="Run sentence log-likelihood evaluation.",
    )
    ap.add_argument(
        "--jsd-base",
        type=float,
        default=2.0,
        help="Log base for JSD (default: 2.0).",
    )
    ap.add_argument(
        "--jsd-per-lhs",
        action="store_true",
        help="Report JSD per nonterminal (excludes preterminals).",
    )
    ap.add_argument(
        "--loglik-grammar",
        choices=["induced", "oracle"],
        default="induced",
        help="Which grammar to use for sentence log-likelihood (default: induced).",
    )
    ap.add_argument(
        "--loglik-mode",
        choices=["inside", "viterbi"],
        default="inside",
        help="Log-likelihood mode: inside (sum over parses) or viterbi (best parse).",
    )
    ap.add_argument(
        "--loglik-out",
        help="Optional output file with per-sentence log-likelihoods.",
    )

    args = ap.parse_args()

    want_parse = args.eval_parse
    want_jsd = args.eval_jsd
    want_loglik = args.eval_loglik
    if not (want_parse or want_jsd or want_loglik):
        want_parse = True
    if want_jsd and (args.jsd_base <= 0 or args.jsd_base == 1.0):
        raise SystemExit("--jsd-base must be > 0 and not equal to 1.")

    if (want_parse or want_loglik or args.prune_lexicon) and not args.sentences:
        raise SystemExit("--sentences is required for parse and log-likelihood evaluations.")

    sentences = []
    sentences_path = Path(args.sentences) if args.sentences else None
    if sentences_path:
        sentences = read_sentences(sentences_path)
        if (want_parse or want_loglik or args.prune_lexicon) and not sentences:
            raise SystemExit("No sentences found.")

    need_oracle_for_parse = want_parse and not args.gold_parses
    need_oracle_rules = (
        want_jsd
        or (want_loglik and args.loglik_grammar == "oracle")
        or need_oracle_for_parse
        or (want_parse and args.oracle_grammar is not None)
    )

    if need_oracle_rules and not args.oracle_grammar:
        raise SystemExit("--oracle-grammar is required for this evaluation.")

    induced_path = Path(args.induced_grammar)
    induced_rules = parse_weighted_rules(induced_path, args.weight_index)
    if not induced_rules:
        raise SystemExit("No induced grammar rules found.")

    oracle_rules = []
    if need_oracle_rules:
        oracle_rules = parse_weighted_rules(Path(args.oracle_grammar), args.weight_index)
        if not oracle_rules:
            raise SystemExit("No oracle grammar rules found.")

    normalized_oracle_rules = None
    normalized_induced_rules = None
    if want_jsd:
        normalized_oracle_rules = normalize_rules(oracle_rules)
        normalized_induced_rules = normalize_rules(induced_rules)

    decoding_oracle_rules = oracle_rules
    decoding_induced_rules = induced_rules
    if args.prune_lexicon and sentences:
        vocab = collect_vocab(sentences)
        decoding_induced_rules, induced_dropped = prune_lexicon_rules(
            decoding_induced_rules, vocab
        )
        oracle_dropped = 0
        if decoding_oracle_rules:
            decoding_oracle_rules, oracle_dropped = prune_lexicon_rules(
                decoding_oracle_rules, vocab
            )
        print(
            f"[info] pruned lexicon rules: oracle {oracle_dropped}, induced {induced_dropped}",
            file=sys.stderr,
        )

    normalized_decoding_oracle = (
        normalize_rules(decoding_oracle_rules) if decoding_oracle_rules else None
    )
    normalized_decoding_induced = normalize_rules(decoding_induced_rules)
    processes = resolve_processes(args.processes)

    if want_parse:
        predicted = None
        pred_errors = 0
        pred_no_parse = 0
        cache_path = None
        cached_predicted = False
        if not args.no_pred_cache and sentences_path:
            cache_path = predicted_cache_path(
                induced_path,
                sentences_path,
                args.start_symbol,
                args.weight_index,
                args.prune_lexicon,
            )
            if cache_path.exists() and not args.refresh_pred_cache:
                predicted = read_parse_records(cache_path)
                if len(predicted) != len(sentences):
                    print(
                        f"[WARN] predicted cache count ({len(predicted)}) != sentence count ({len(sentences)}); recomputing",
                        file=sys.stderr,
                    )
                    predicted = None
                else:
                    cached_predicted = True
                    pred_no_parse = sum(
                        1
                        for sent, tree in zip(sentences, predicted)
                        if tree is None and sent.strip()
                    )

        if predicted is None:
            if processes > 1:
                predicted, pred_errors, pred_no_parse = parse_sentences_parallel(
                    normalized_decoding_induced,
                    args.start_symbol,
                    sentences,
                    processes=processes,
                    progress_every=args.progress_every,
                    label="predicted",
                )
            else:
                induced_parser = build_parser(
                    normalized_decoding_induced, args.start_symbol
                )
                predicted, pred_errors, pred_no_parse = parse_sentences(
                    induced_parser,
                    sentences,
                    progress_every=args.progress_every,
                    label="predicted",
                )
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                write_parse_records(cache_path, sentences, predicted)

        eval_jobs = []
        if args.oracle_grammar:
            if not normalized_decoding_oracle:
                raise SystemExit("Oracle grammar required to generate gold parses.")
            if processes > 1:
                gold_oracle, gold_errors, gold_no_parse = parse_sentences_parallel(
                    normalized_decoding_oracle,
                    args.start_symbol,
                    sentences,
                    processes=processes,
                    progress_every=args.progress_every,
                    label="gold",
                )
            else:
                oracle_parser = build_parser(normalized_decoding_oracle, args.start_symbol)
                gold_oracle, gold_errors, gold_no_parse = parse_sentences(
                    oracle_parser,
                    sentences,
                    progress_every=args.progress_every,
                    label="gold",
                )
            eval_jobs.append(
                {
                    "label": "oracle grammar",
                    "gold": gold_oracle,
                    "gold_source": "oracle grammar",
                    "gold_errors": gold_errors,
                    "gold_no_parse": gold_no_parse,
                }
            )

        if args.gold_parses:
            gold_file = read_parse_records(Path(args.gold_parses))
            gold_errors = 0
            gold_no_parse = sum(
                1
                for sent, tree in zip(sentences, gold_file)
                if tree is None and sent.strip()
            )
            if len(gold_file) != len(sentences):
                print(
                    f"[WARN] gold parse count ({len(gold_file)}) != sentence count ({len(sentences)})"
                )
            eval_jobs.append(
                {
                    "label": "gold parses",
                    "gold": gold_file,
                    "gold_source": "gold parse file",
                    "gold_errors": gold_errors,
                    "gold_no_parse": gold_no_parse,
                }
            )

        for job in eval_jobs:
            stats = evaluate_both(
                sentences,
                predicted,
                job["gold"],
                include_preterminals=args.include_preterminals,
                include_root=args.include_root,
                progress_every=args.progress_every,
            )
            print_parse_eval(
                stats,
                job["gold_source"],
                cache_path,
                cached_predicted,
                pred_errors,
                pred_no_parse,
                job["gold_errors"],
                job["gold_no_parse"],
                label=job["label"],
            )

    if want_jsd:
        oracle_by = rules_by_lhs(normalized_oracle_rules)
        induced_by = rules_by_lhs(normalized_induced_rules)
        shared_lhs = sorted(set(oracle_by) & set(induced_by))
        missing_oracle = sorted(set(induced_by) - set(oracle_by))
        missing_induced = sorted(set(oracle_by) - set(induced_by))
        if not shared_lhs:
            raise SystemExit("No shared nonterminals found for JSD.")
        jsd_vals = []
        jsd_by_lhs = {}
        for lhs in shared_lhs:
            val = jsd_divergence(oracle_by[lhs], induced_by[lhs], args.jsd_base)
            jsd_vals.append(val)
            jsd_by_lhs[lhs] = val
        mean_jsd = sum(jsd_vals) / len(jsd_vals)
        print("JSD evaluation")
        print(f"- shared nonterminals: {len(shared_lhs)}")
        print(f"- only in oracle: {len(missing_induced)}")
        print(f"- only in induced: {len(missing_oracle)}")
        print(f"- mean JSD (base {args.jsd_base}): {mean_jsd:.6f}")
        if args.jsd_per_lhs:
            print("JSD per nonterminal")
            preterminals = find_preterminal_lhs(
                normalized_oracle_rules, normalized_induced_rules
            )
            for lhs in shared_lhs:
                if lhs in preterminals:
                    continue
                print(f"{lhs}\t{jsd_by_lhs[lhs]:.6f}")

    if want_loglik:
        loglik_rules = (
            normalized_decoding_induced
            if args.loglik_grammar == "induced"
            else normalized_decoding_oracle
        )
        if not loglik_rules:
            raise SystemExit("Requested log-likelihood grammar is unavailable.")
        loglik_mode = resolve_loglik_mode(args.loglik_mode)
        logliks, ll_errors, ll_no_parse, ll_empty = compute_logliks(
            loglik_rules,
            args.start_symbol,
            sentences,
            mode=loglik_mode,
            processes=processes,
            progress_every=args.progress_every,
        )
        valid = [v for v in logliks if v is not None]
        total = sum(valid)
        mean = total / len(valid) if valid else float("nan")
        print("Sentence log-likelihood")
        print(f"- grammar: {args.loglik_grammar}")
        print(f"- mode: {loglik_mode}")
        print(f"- sentences: {len(sentences)}")
        print(f"- evaluated: {len(valid)}")
        print(f"- total log-likelihood: {total:.6f}")
        print(f"- mean log-likelihood: {mean:.6f}")
        print(f"- no-parse: {ll_no_parse}, parse errors: {ll_errors}, empty: {ll_empty}")
        if args.loglik_out:
            out_lines = [
                "" if v is None else f"{v:.6f}"
                for v in logliks
            ]
            Path(args.loglik_out).write_text(
                "\n".join(out_lines) + "\n", encoding="utf-8"
            )


if __name__ == "__main__":
    main()
