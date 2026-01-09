#!/usr/bin/env python3
import sys
import math
import argparse
from collections import defaultdict

NEG_INF = -1e9  # stand-in for log(0)


def log(x):
    if x <= 0.0:
        return NEG_INF
    return math.log(x)


def logsumexp(values):
    """Stable log-sum-exp over an iterable of log-values."""
    vals = [v for v in values if v > NEG_INF]
    if not vals:
        return NEG_INF
    m = max(vals)
    return m + math.log(sum(math.exp(v - m) for v in vals))


def parse_grammar(grammar_path):
    """
    Extract HMM-style parameters from an Inside-Outside grammar:
      S      --> H_TAG          (start distribution)
      H_X    --> H_Y L_X        (transition X -> Y)
      H_X    --> L_X            (stop probability for X)
      L_X    --> word           (emission for X)
    """
    start = {}
    trans = defaultdict(dict)
    stop = {}
    emit = defaultdict(dict)

    with open(grammar_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()

            # find '-->'
            try:
                arrow = parts.index("-->")
            except ValueError:
                continue

            # first number is weight; ignore any alpha/pseudocount after
            try:
                weight = float(parts[0])
            except ValueError:
                continue

            parent = parts[arrow - 1]
            rhs = parts[arrow + 1:]

            # S --> H_TAG  : start prob (unnormalized)
            if parent == "S" and len(rhs) == 1 and rhs[0].startswith("H_"):
                state = rhs[0][2:]
                start[state] = start.get(state, 0.0) + weight
                continue

            # H_X --> H_Y L_X  : transition X->Y
            if parent.startswith("H_") and len(rhs) == 2 \
               and rhs[0].startswith("H_") and rhs[1].startswith("L_"):
                src = parent[2:]
                dst = rhs[0][2:]
                trans[src][dst] = trans[src].get(dst, 0.0) + weight
                continue

            # H_X --> L_X  : stop probability from X
            if parent.startswith("H_") and len(rhs) == 1 and rhs[0].startswith("L_"):
                state = parent[2:]
                stop[state] = stop.get(state, 0.0) + weight
                continue

            # L_X --> word  : emission
            if parent.startswith("L_") and len(rhs) == 1:
                state = parent[2:]
                word = rhs[0]
                emit[state][word] = emit[state].get(word, 0.0) + weight
                continue

    # Normalize distributions
    def normalize(d):
        s = sum(d.values())
        if s > 0:
            for k in list(d.keys()):
                d[k] /= s

    if start:
        normalize(start)
    for src in trans:
        normalize(trans[src])
    for st in emit:
        normalize(emit[st])
    # stop probs we leave as-is; they are already normalized within each H_X rule group in IO’s output.

    return start, trans, stop, emit


def build_log_params(states, start, trans, stop, emit):
    """Precompute log-prob tables for decoding."""
    log_start = {s: log(start.get(s, 0.0)) for s in states}
    log_stop = {s: log(stop.get(s, 0.0)) for s in states}
    log_trans = {
        s: {t2: log(trans[s].get(t2, 0.0)) for t2 in states}
        for s in states
    }
    log_emit = {
        s: defaultdict(
            lambda: NEG_INF,
            {w: log(p) for w, p in emit.get(s, {}).items()}
        )
        for s in states
    }
    return log_start, log_stop, log_trans, log_emit


def viterbi_tag(words, states, log_start, log_stop, log_trans, log_emit):
    """Standard Viterbi (MAP) decoding."""
    T = len(words)
    if T == 0:
        return []

    dp = [{s: NEG_INF for s in states} for _ in range(T)]
    bp = [{s: None for s in states} for _ in range(T)]

    # t = 0
    w0 = words[0]
    for s in states:
        dp[0][s] = log_start[s] + log_emit[s][w0]

    # recursion
    for t in range(1, T):
        w = words[t]
        for s in states:
            best = NEG_INF
            best_prev = None
            for p in states:
                score = dp[t-1][p] + log_trans[p][s] + log_emit[s][w]
                if score > best:
                    best = score
                    best_prev = p
            dp[t][s] = best
            bp[t][s] = best_prev

    # termination
    best_final = NEG_INF
    last_state = None
    for s in states:
        score = dp[T-1][s] + log_stop[s]
        if score > best_final:
            best_final = score
            last_state = s

    if last_state is None:  # fallback
        last_state = max(states, key=lambda s: dp[T-1][s])

    tags = [None] * T
    tags[T-1] = last_state
    for t in range(T-1, 0, -1):
        prev = bp[t][tags[t]]
        if prev is None:
            prev = max(states, key=lambda s: dp[t-1][s])
        tags[t-1] = prev
    return tags


def posterior_tag(words, states, log_start, log_stop, log_trans, log_emit):
    """
    Posterior decoding using forward–backward:

      p_t(s) ∝ alpha_t(s) + beta_t(s) - logZ
      tag_t = argmax_s p_t(s)

    where alpha, beta are in log-space.
    """
    T = len(words)
    if T == 0:
        return []

    # Forward: alpha[t][s]
    alpha = [{s: NEG_INF for s in states} for _ in range(T)]
    # t=0
    w0 = words[0]
    for s in states:
        alpha[0][s] = log_start[s] + log_emit[s][w0]

    # recursion
    for t in range(1, T):
        w = words[t]
        for s in states:
            terms = []
            for p in states:
                if alpha[t-1][p] <= NEG_INF or log_trans[p][s] <= NEG_INF:
                    continue
                terms.append(alpha[t-1][p] + log_trans[p][s])
            alpha[t][s] = (log_emit[s][w] + logsumexp(terms)) if terms else NEG_INF

    # logZ (sentence probability)
    terms = []
    for s in states:
        if alpha[T-1][s] <= NEG_INF or log_stop[s] <= NEG_INF:
            continue
        terms.append(alpha[T-1][s] + log_stop[s])
    logZ = logsumexp(terms)

    # Backward: beta[t][s]
    beta = [{s: NEG_INF for s in states} for _ in range(T)]
    # t = T-1
    for s in states:
        beta[T-1][s] = log_stop[s]

    # recursion backwards
    for t in range(T-2, -1, -1):
        w_next = words[t+1]
        for s in states:
            terms = []
            for nxt in states:
                if log_trans[s][nxt] <= NEG_INF:
                    continue
                val = log_trans[s][nxt] + log_emit[nxt][w_next] + beta[t+1][nxt]
                terms.append(val)
            beta[t][s] = logsumexp(terms)

    # Posterior: argmax_s alpha[t][s] + beta[t][s] - logZ
    tags = []
    for t in range(T):
        best_s = None
        best_val = NEG_INF
        for s in states:
            val = alpha[t][s] + beta[t][s] - logZ
            if val > best_val:
                best_val = val
                best_s = s
        if best_s is None:
            best_s = max(states, key=lambda s: alpha[t][s])
        tags.append(best_s)
    return tags


def main():
    parser = argparse.ArgumentParser(
        description="Decode tags from IO-trained grammar (HMM-style)."
    )
    parser.add_argument("-g", "--grammar", required=True,
                        help="Path to grammar file (io output)")
    parser.add_argument("-i", "--input", required=True,
                        help="Test sentences file (one tokenized sentence per line)")
    parser.add_argument("-o", "--output", required=True,
                        help="Where to save predicted tags (one line per sentence)")
    parser.add_argument("-d", "--decode", choices=["viterbi", "posterior"],
                        default="viterbi",
                        help="Decoding method: 'viterbi' (MAP path) or 'posterior' (per-position marginals)")

    args = parser.parse_args()

    start, trans, stop, emit = parse_grammar(args.grammar)

    # States = all H_*/L_* suffixes that appear in start/trans/stop/emit
    states = sorted(
        set(start.keys()) |
        set(trans.keys()) |
        set(stop.keys()) |
        set(emit.keys())
    )
    if not states:
        raise ValueError("No states found in grammar. Check that H_*, L_* rules are present.")

    log_start, log_stop, log_trans, log_emit = build_log_params(
        states, start, trans, stop, emit
    )

    decode_fn = viterbi_tag if args.decode == "viterbi" else posterior_tag

    with open(args.input, encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                fout.write("\n")
                continue
            words = line.split()
            tags = decode_fn(words, states, log_start, log_stop, log_trans, log_emit)
            fout.write(" ".join(tags) + "\n")


if __name__ == "__main__":
    main()
