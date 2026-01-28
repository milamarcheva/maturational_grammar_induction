#!/usr/bin/env python3
import argparse
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

Symbol = Tuple[str, bool]


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
            rhs = parts[arrow_idx + 1 :]

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


def read_sentences(sent_path: Path) -> List[str]:
    return sent_path.read_text(encoding="utf-8").splitlines()


def invert_matrix(matrix: List[List[float]]) -> List[List[float]]:
    n = len(matrix)
    aug = [
        row[:] + [1.0 if i == j else 0.0 for j in range(n)]
        for i, row in enumerate(matrix)
    ]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("Unary rule matrix is singular; grammar may be inconsistent.")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_val = aug[col][col]
        for j in range(2 * n):
            aug[col][j] /= pivot_val
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor == 0.0:
                continue
            for j in range(2 * n):
                aug[r][j] -= factor * aug[col][j]
    return [row[n:] for row in aug]


def unary_closure(
    nonterminals: List[str],
    unary_rules: List[Tuple[str, str, float]],
) -> List[List[float]]:
    n = len(nonterminals)
    index = {nt: i for i, nt in enumerate(nonterminals)}
    mat = [[0.0 for _ in range(n)] for _ in range(n)]
    for lhs, rhs, prob in unary_rules:
        mat[index[lhs]][index[rhs]] += prob
    system = [
        [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)
    ]
    for i in range(n):
        for j in range(n):
            system[i][j] -= mat[i][j]
    return invert_matrix(system)


def eliminate_unary_rules(
    rules: List[Tuple[str, List[Symbol], float]],
    nonterminals: set[str],
) -> List[Tuple[str, Tuple[Symbol, ...], float]]:
    lexical: List[Tuple[str, Tuple[Symbol, ...], float]] = []
    unary: List[Tuple[str, str, float]] = []
    other: List[Tuple[str, Tuple[Symbol, ...], float]] = []
    for lhs, rhs, prob in rules:
        if len(rhs) == 1 and rhs[0][1]:
            unary.append((lhs, rhs[0][0], prob))
        elif len(rhs) == 1:
            lexical.append((lhs, (rhs[0],), prob))
        else:
            other.append((lhs, tuple(rhs), prob))

    if not unary:
        return lexical + other

    nts = sorted(nonterminals)
    closure = unary_closure(nts, unary)
    index = {nt: i for i, nt in enumerate(nts)}

    base_by_lhs: dict[str, List[Tuple[Tuple[Symbol, ...], float]]] = defaultdict(list)
    for lhs, rhs, prob in lexical + other:
        base_by_lhs[lhs].append((rhs, prob))

    expanded: dict[Tuple[str, Tuple[Symbol, ...]], float] = defaultdict(float)
    for lhs in nts:
        i = index[lhs]
        for rhs_nt in nts:
            coeff = closure[i][index[rhs_nt]]
            if coeff == 0.0:
                continue
            for rhs, prob in base_by_lhs.get(rhs_nt, []):
                expanded[(lhs, rhs)] += coeff * prob

    return [(lhs, rhs, prob) for (lhs, rhs), prob in expanded.items()]


def lift_terminals(
    rules: List[Tuple[str, Tuple[Symbol, ...], float]],
    nonterminals: set[str],
) -> Tuple[List[Tuple[str, Tuple[Symbol, ...], float]], set[str]]:
    terminal_map: dict[str, str] = {}
    lifted: List[Tuple[str, Tuple[Symbol, ...], float]] = []
    extra: List[Tuple[str, Tuple[Symbol, ...], float]] = []
    counter = 0
    for lhs, rhs, prob in rules:
        if len(rhs) < 2:
            lifted.append((lhs, rhs, prob))
            continue
        new_rhs: List[Symbol] = []
        for name, is_nt in rhs:
            if is_nt:
                new_rhs.append((name, True))
                continue
            nt = terminal_map.get(name)
            if nt is None:
                nt = f"__TERM_{counter}"
                counter += 1
                while nt in nonterminals:
                    nt = f"__TERM_{counter}"
                    counter += 1
                terminal_map[name] = nt
                nonterminals.add(nt)
                extra.append((nt, ((name, False),), 1.0))
            new_rhs.append((nt, True))
        lifted.append((lhs, tuple(new_rhs), prob))
    lifted.extend(extra)
    return lifted, nonterminals


def binarize_rules(
    rules: List[Tuple[str, Tuple[Symbol, ...], float]],
    nonterminals: set[str],
) -> Tuple[List[Tuple[str, Tuple[Symbol, ...], float]], set[str]]:
    binarized: List[Tuple[str, Tuple[Symbol, ...], float]] = []
    counter = 0
    for lhs, rhs, prob in rules:
        if len(rhs) <= 2:
            binarized.append((lhs, rhs, prob))
            continue
        current = lhs
        for idx in range(len(rhs) - 2):
            new_nt = f"__BIN_{counter}"
            counter += 1
            while new_nt in nonterminals:
                new_nt = f"__BIN_{counter}"
                counter += 1
            nonterminals.add(new_nt)
            first_prob = prob if idx == 0 else 1.0
            binarized.append((current, (rhs[idx], (new_nt, True)), first_prob))
            current = new_nt
        binarized.append((current, (rhs[-2], rhs[-1]), 1.0))
    return binarized, nonterminals


def build_indexed_grammar(
    rules: List[Tuple[str, Tuple[Symbol, ...], float]],
    nonterminals: set[str],
) -> Tuple[dict[str, List[Tuple[str, float]]], dict[str, List[Tuple[str, str, float]]]]:
    lex_rules: dict[str, List[Tuple[str, float]]] = defaultdict(list)
    binary_by_left: dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
    for lhs, rhs, prob in rules:
        if prob <= 0:
            continue
        if len(rhs) == 1 and not rhs[0][1]:
            lex_rules[rhs[0][0]].append((lhs, math.log(prob)))
        elif len(rhs) == 2 and rhs[0][1] and rhs[1][1]:
            binary_by_left[rhs[0][0]].append((rhs[1][0], lhs, math.log(prob)))
        else:
            rhs_str = " ".join(sym[0] for sym in rhs)
            raise ValueError(f"Unexpected rule arity for {lhs} -> {rhs_str}")
    return lex_rules, binary_by_left


def logsumexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def inside_logprob(
    tokens: List[str],
    lex_rules: dict[str, List[Tuple[str, float]]],
    binary_by_left: dict[str, List[Tuple[str, str, float]]],
    start_symbol: str,
) -> Tuple[Optional[float], str]:
    if not tokens:
        return None, "empty"
    n = len(tokens)
    chart: List[List[dict[str, float]]] = [
        [defaultdict(lambda: -math.inf) for _ in range(n + 1)]
        for _ in range(n)
    ]

    for i, word in enumerate(tokens):
        for lhs, logp in lex_rules.get(word, []):
            chart[i][i + 1][lhs] = logsumexp(chart[i][i + 1][lhs], logp)

    for span in range(2, n + 1):
        for i in range(0, n - span + 1):
            j = i + span
            cell = chart[i][j]
            for k in range(i + 1, j):
                left = chart[i][k]
                right = chart[k][j]
                if not left or not right:
                    continue
                for b_sym, logp_b in left.items():
                    rules = binary_by_left.get(b_sym)
                    if not rules:
                        continue
                    for c_sym, lhs, logp_rule in rules:
                        logp_c = right.get(c_sym)
                        if logp_c is None:
                            continue
                        val = logp_rule + logp_b + logp_c
                        cell[lhs] = logsumexp(cell[lhs], val)

    total = chart[0][n].get(start_symbol, -math.inf)
    if total == -math.inf:
        return None, "no_parse"
    return total, "ok"


def prepare_rules(
    raw_rules: List[Tuple[str, List[str], float]]
) -> Tuple[
    dict[str, List[Tuple[str, float]]],
    dict[str, List[Tuple[str, str, float]]],
    set[str],
]:
    nonterminals = {lhs for lhs, _, _ in raw_rules}
    typed_rules: List[Tuple[str, List[Symbol], float]] = []
    for lhs, rhs, prob in raw_rules:
        rhs_symbols: List[Symbol] = []
        for tok in rhs:
            if tok in nonterminals:
                rhs_symbols.append((tok, True))
            else:
                rhs_symbols.append((maybe_unquote(tok), False))
        typed_rules.append((lhs, rhs_symbols, prob))

    no_unary = eliminate_unary_rules(typed_rules, nonterminals)
    lifted, nonterminals = lift_terminals(no_unary, nonterminals)
    binarized, nonterminals = binarize_rules(lifted, nonterminals)
    lex_rules, binary_by_left = build_indexed_grammar(binarized, nonterminals)
    return lex_rules, binary_by_left, nonterminals


def resolve_processes(requested: int) -> int:
    if requested == 0:
        return os.cpu_count() or 1
    return requested if requested > 1 else 1


_WORKER_LEX = None
_WORKER_BINARY = None
_WORKER_START = None


def _init_worker(lex_rules, binary_by_left, start_symbol: str) -> None:
    global _WORKER_LEX
    global _WORKER_BINARY
    global _WORKER_START
    _WORKER_LEX = lex_rules
    _WORKER_BINARY = binary_by_left
    _WORKER_START = start_symbol


def _loglik_worker(sentence: str) -> Tuple[Optional[float], str]:
    tokens = sentence.split()
    return inside_logprob(tokens, _WORKER_LEX, _WORKER_BINARY, _WORKER_START)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute marginal (inside) log-likelihood for tokenized sentences."
    )
    ap.add_argument(
        "--grammar",
        required=True,
        help="Grammar file with weighted rules.",
    )
    ap.add_argument(
        "--sentences",
        required=True,
        help="Sentence file (one tokenized sentence per line).",
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
        "--start-symbol",
        default="ROOT",
        help="Start symbol (default: ROOT).",
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
        help="Number of worker processes (default: 1, 0 = auto).",
    )
    ap.add_argument(
        "--loglik-out",
        help="Optional output file with per-sentence log-likelihoods.",
    )
    ap.add_argument(
        "--timing",
        action="store_true",
        help="Report total elapsed time.",
    )

    args = ap.parse_args()

    overall_start = time.perf_counter() if args.timing else None

    grammar_path = Path(args.grammar)
    rules = parse_weighted_rules(grammar_path, args.weight_index)
    if not rules:
        raise SystemExit("No grammar rules found.")
    rules = normalize_rules(rules)

    lex_rules, binary_by_left, nonterminals = prepare_rules(rules)
    if args.start_symbol not in nonterminals:
        raise SystemExit(f"Start symbol {args.start_symbol} not found in grammar.")

    sentences = read_sentences(Path(args.sentences))
    if not sentences:
        raise SystemExit("No sentences found.")

    processes = resolve_processes(args.processes)

    logliks: List[Optional[float]] = []
    no_parse = 0
    empty = 0
    total = len(sentences)
    if processes > 1:
        chunksize = max(1, total // max(processes * 4, 1))
        with ProcessPoolExecutor(
            max_workers=processes,
            initializer=_init_worker,
            initargs=(lex_rules, binary_by_left, args.start_symbol),
        ) as ex:
            for idx, (loglik, status) in enumerate(
                ex.map(_loglik_worker, sentences, chunksize=chunksize), start=1
            ):
                if args.progress_every > 0 and (
                    idx % args.progress_every == 0 or idx == total
                ):
                    print(f"[progress] loglik {idx}/{total}", file=sys.stderr)
                logliks.append(loglik)
                if status == "no_parse":
                    no_parse += 1
                elif status == "empty":
                    empty += 1
    else:
        for idx, sent in enumerate(sentences, start=1):
            if args.progress_every > 0 and (
                idx % args.progress_every == 0 or idx == total
            ):
                print(f"[progress] loglik {idx}/{total}", file=sys.stderr)
            loglik, status = inside_logprob(
                sent.split(), lex_rules, binary_by_left, args.start_symbol
            )
            logliks.append(loglik)
            if status == "no_parse":
                no_parse += 1
            elif status == "empty":
                empty += 1

    valid = [v for v in logliks if v is not None]
    total_ll = sum(valid)
    mean_ll = total_ll / len(valid) if valid else float("nan")
    normalized_vals = []
    total_tokens = 0
    for sent, loglik in zip(sentences, logliks):
        if loglik is None:
            continue
        tokens = sent.split()
        if not tokens:
            continue
        normalized_vals.append(loglik / len(tokens))
        total_tokens += len(tokens)
    mean_norm = (
        sum(normalized_vals) / len(normalized_vals) if normalized_vals else float("nan")
    )
    corpus_norm = total_ll / total_tokens if total_tokens else float("nan")

    print("Marginal log-likelihood")
    print(f"- sentences: {len(sentences)}")
    print(f"- evaluated: {len(valid)}")
    print(f"- total log-likelihood: {total_ll:.6f}")
    print(f"- mean log-likelihood: {mean_ll:.6f}")
    print(f"- mean normalized log-likelihood (per token, sentence avg): {mean_norm:.6f}")
    print(f"- normalized log-likelihood (per token, corpus): {corpus_norm:.6f}")
    print(f"- total tokens (evaluated): {total_tokens}")
    print(f"- no-parse: {no_parse}, empty: {empty}")

    if args.loglik_out:
        out_lines = ["" if v is None else f"{v:.6f}" for v in logliks]
        Path(args.loglik_out).write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    if args.timing:
        elapsed = time.perf_counter() - overall_start
        print(f"[timer] total: {elapsed:.2f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
