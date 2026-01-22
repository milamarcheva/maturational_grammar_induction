#!/usr/bin/env python3
import argparse
import math
import subprocess
import signal
import contextlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


Rule = Tuple[str, List[str], float]
LexRule = Tuple[str, List[str], float, float]


def parse_grammar_weights(path: Path) -> List[Rule]:
    rules: List[Rule] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if "-->" in parts:
            arrow = parts.index("-->")
        elif "->" in parts:
            arrow = parts.index("->")
        else:
            continue
        if arrow < 2 or arrow >= len(parts) - 1:
            continue
        try:
            weight = float(parts[0])
        except ValueError:
            continue
        lhs = parts[1]
        rhs = parts[arrow + 1 :]
        rules.append((lhs, rhs, weight))
    return rules


def parse_lexicon(path: Path) -> List[LexRule]:
    rules: List[LexRule] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if "-->" in parts:
            arrow = parts.index("-->")
        elif "->" in parts:
            arrow = parts.index("->")
        else:
            continue
        if arrow < 3 or arrow >= len(parts) - 1:
            continue
        try:
            weight = float(parts[0])
            bias = float(parts[1])
        except ValueError:
            continue
        lhs = parts[2]
        rhs = parts[arrow + 1 :]
        rules.append((lhs, rhs, weight, bias))
    return rules


def merge_missing_lexicon(rules: List[Rule], lexicon: List[LexRule]) -> List[Rule]:
    rule_ids = {(lhs, tuple(rhs)) for lhs, rhs, _ in rules}
    merged = list(rules)
    for lhs, rhs, weight, _ in lexicon:
        rid = (lhs, tuple(rhs))
        if rid not in rule_ids:
            merged.append((lhs, rhs, weight))
    return merged


def format_float(value: float) -> str:
    return f"{value:.12g}"


def format_rule_line(lhs: str, rhs: List[str], weight: float, bias: float) -> str:
    return f"{format_float(weight)} {format_float(bias)} {lhs} --> {' '.join(rhs)}"


def normalize_weights_by_lhs(rules: List[Rule]) -> List[Rule]:
    totals: Dict[str, float] = defaultdict(float)
    for lhs, _, weight in rules:
        totals[lhs] += weight
    normalized: List[Rule] = []
    for lhs, rhs, weight in rules:
        total = totals[lhs]
        weight_out = weight / total if total else 0.0
        normalized.append((lhs, rhs, weight_out))
    return normalized


def cap_unary_mass(
    rules: List[Rule], nonterminals: set[str], max_unary_mass: float
) -> List[Rule]:
    totals: Dict[str, float] = defaultdict(float)
    unary_totals: Dict[str, float] = defaultdict(float)
    for lhs, rhs, weight in rules:
        totals[lhs] += weight
        if len(rhs) == 1 and rhs[0] in nonterminals:
            unary_totals[lhs] += weight

    scale: Dict[str, float] = {}
    for lhs, total in totals.items():
        unary = unary_totals.get(lhs, 0.0)
        if total <= 0.0 or unary <= 0.0:
            continue
        if unary / total > max_unary_mass:
            if total == unary:
                continue
            scale[lhs] = (max_unary_mass * (total - unary)) / (
                unary * (1.0 - max_unary_mass)
            )

    if not scale:
        return rules

    adjusted: List[Rule] = []
    for lhs, rhs, weight in rules:
        if lhs in scale and len(rhs) == 1 and rhs[0] in nonterminals:
            weight = weight * scale[lhs]
        adjusted.append((lhs, rhs, weight))
    return adjusted


def sanitize_rules(
    rules: List[Rule],
    weight_floor: float,
    normalize: bool,
    max_unary_mass: Optional[float],
) -> List[Rule]:
    cleaned: List[Rule] = []
    for lhs, rhs, weight in rules:
        if not math.isfinite(weight) or weight < 0.0:
            weight = 0.0
        if weight_floor and weight < weight_floor:
            weight = weight_floor
        cleaned.append((lhs, rhs, weight))

    nonterminals = {lhs for lhs, _, _ in cleaned}
    if max_unary_mass is not None:
        cleaned = cap_unary_mass(cleaned, nonterminals, max_unary_mass)

    if normalize:
        cleaned = normalize_weights_by_lhs(cleaned)

    return cleaned


def add_biases(
    rules: List[Rule],
    alpha_total: float,
    alpha_lex_total: Optional[float],
    bias_floor: float,
) -> List[Tuple[str, List[str], float, float]]:
    nonterminals = {lhs for lhs, _, _ in rules}
    totals: Dict[str, float] = defaultdict(float)
    for lhs, _, weight in rules:
        totals[lhs] += weight

    updated: List[Tuple[str, List[str], float, float]] = []
    for lhs, rhs, weight in rules:
        total = totals[lhs]
        alpha = alpha_total
        if len(rhs) == 1 and rhs[0] not in nonterminals and alpha_lex_total is not None:
            alpha = alpha_lex_total
        bias = alpha * (weight / total) if total > 0.0 else 0.0
        if bias_floor and bias < bias_floor:
            bias = bias_floor
        updated.append((lhs, rhs, weight, bias))
    return updated


def run_io_once(
    io_path: Path,
    grammar_path: Path,
    yields_path: Path,
    out_path: Path,
    log_path: Optional[Path],
    args: argparse.Namespace,
) -> None:
    cmd = [str(io_path)]
    if args.variational_bayes:
        cmd.append("-V")
    cmd += [
        "-d",
        str(args.debug),
        "-n",
        str(args.maxits),
        "-g",
        str(grammar_path),
        "-p",
        str(args.minruleprob),
        "-W",
        str(args.wordscale),
        str(yields_path),
    ]
    with out_path.open("w", encoding="utf-8") as fout:
        if args.interrupt_after is None:
            if log_path:
                with log_path.open("w", encoding="utf-8") as flog:
                    subprocess.run(cmd, check=True, stdout=fout, stderr=flog)
            else:
                subprocess.run(cmd, check=True, stdout=fout)
            return

        with (log_path.open("w", encoding="utf-8") if log_path else contextlib.nullcontext()) as flog:
            proc = subprocess.Popen(
                cmd,
                stdout=fout,
                stderr=subprocess.PIPE,
                text=True,
            )
            interrupted = False
            assert proc.stderr is not None
            for line in proc.stderr:
                if log_path and flog is not None:
                    flog.write(line)
                stripped = line.lstrip()
                if stripped and stripped[0].isdigit():
                    parts = stripped.split()
                    try:
                        iteration = int(parts[0])
                    except ValueError:
                        iteration = -1
                    if iteration >= args.interrupt_after:
                        proc.send_signal(signal.SIGINT)
                        interrupted = True
                        break
            proc.wait()
            if proc.returncode not in (0, -signal.SIGINT) and not interrupted:
                raise subprocess.CalledProcessError(proc.returncode, cmd)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run io -n 1 repeatedly and sanitize the grammar between iterations."
    )
    ap.add_argument("--grammar", required=True, help="Initial grammar file.")
    ap.add_argument("--yields", required=True, help="Yields file.")
    ap.add_argument("--iterations", type=int, required=True, help="Total iterations.")
    ap.add_argument(
        "--out",
        required=True,
        help="Final sanitized grammar output path.",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Directory for intermediate grammars/logs (default: alongside --out).",
    )
    ap.add_argument(
        "--io",
        default=None,
        help="Path to inside-outside/io (default: ../inside-outside/io).",
    )
    ap.add_argument("--debug", type=int, default=1000, help="io debuglevel.")
    ap.add_argument("--wordscale", type=float, default=1.0, help="io wordscale (-W).")
    ap.add_argument(
        "--minruleprob",
        type=float,
        default=0.0,
        help="io prune threshold (-p).",
    )
    ap.add_argument(
        "--maxits",
        type=int,
        default=1,
        help="io max iterations (-n) per run (default: 1).",
    )
    ap.add_argument(
        "--no-vb",
        dest="variational_bayes",
        action="store_false",
        help="Disable Variational Bayes (-V).",
    )
    ap.set_defaults(variational_bayes=True)
    ap.add_argument(
        "--alpha-total",
        type=float,
        default=1.0,
        help="Total alpha per LHS for bias conversion.",
    )
    ap.add_argument(
        "--alpha-lex-total",
        type=float,
        default=None,
        help="Total alpha per preterminal for lexical rules (default: alpha-total).",
    )
    ap.add_argument(
        "--bias-floor",
        type=float,
        default=1e-6,
        help="Minimum bias assigned to any rule.",
    )
    ap.add_argument(
        "--weight-floor",
        type=float,
        default=0.0,
        help="Minimum weight assigned to any rule before normalization.",
    )
    ap.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize weights per LHS after sanitization.",
    )
    ap.add_argument(
        "--max-unary-mass",
        type=float,
        default=None,
        help="Cap unary (nonterminal) mass per LHS by scaling unary weights.",
    )
    ap.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep per-iteration intermediate grammars.",
    )
    ap.add_argument(
        "--log-dir",
        default=None,
        help="Directory for io stderr logs (default: alongside --out).",
    )
    ap.add_argument(
        "--interrupt-after",
        type=int,
        default=1,
        help=(
            "Send SIGINT after this iteration line is printed to dump the grammar "
            "(default: 1). Use -1 to disable."
        ),
    )
    ap.add_argument(
        "--lexicon",
        default=None,
        help="Lexicon file to restore missing lexical rules after each iteration.",
    )
    args = ap.parse_args()

    if args.debug < 1:
        raise SystemExit("debug must be >= 1 so io writes the grammar to stdout.")
    if args.interrupt_after is not None and args.interrupt_after < 0:
        args.interrupt_after = None

    grammar_path = Path(args.grammar).resolve()
    yields_path = Path(args.yields).resolve()
    out_path = Path(args.out).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else out_path.parent
    log_dir = Path(args.log_dir).resolve() if args.log_dir else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    io_path = (
        Path(args.io).resolve()
        if args.io
        else (Path(__file__).resolve().parents[1] / "inside-outside" / "io")
    )
    lexicon_rules = (
        parse_lexicon(Path(args.lexicon).resolve()) if args.lexicon else None
    )

    current_grammar = grammar_path
    for idx in range(1, args.iterations + 1):
        raw_path = out_dir / f"iter_{idx:02d}_raw.lt"
        log_path = log_dir / f"iter_{idx:02d}.log"
        run_io_once(io_path, current_grammar, yields_path, raw_path, log_path, args)

        rules = parse_grammar_weights(raw_path)
        if lexicon_rules:
            rules = merge_missing_lexicon(rules, lexicon_rules)
        rules = sanitize_rules(
            rules,
            weight_floor=args.weight_floor,
            normalize=args.normalize,
            max_unary_mass=args.max_unary_mass,
        )
        rules_with_bias = add_biases(
            rules,
            alpha_total=args.alpha_total,
            alpha_lex_total=args.alpha_lex_total,
            bias_floor=args.bias_floor,
        )

        sanitized_path = out_dir / f"iter_{idx:02d}.lt"
        sanitized_path.write_text(
            "\n".join(
                format_rule_line(lhs, rhs, weight, bias)
                for lhs, rhs, weight, bias in rules_with_bias
            )
            + "\n",
            encoding="utf-8",
        )

        current_grammar = sanitized_path
        if not args.keep_intermediate:
            raw_path.unlink(missing_ok=True)

    if current_grammar != out_path:
        out_path.write_text(
            current_grammar.read_text(encoding="utf-8"), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
