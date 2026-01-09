#!/usr/bin/env python3
"""
Build grammars with different production/lexicon biases (alphas) and train them.

Example:
  python milas_scripts/train_nva_alphas.py \
    --prod-biases 0.3 1 \
    --lex-biases 0.3 1 \
    --productions milas_grammars/grammar_NVA_VB_len2plus_27_reduced.lt \
    --lexicon milas_grammars/lexicon_len2plus_NVA_bitpar.lt \
    --prime-file milas_grammars/primed_words.txt \
    --yields milas_grammars/brown_sentences_len2plus.txt
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Dict, Set, Tuple
from datetime import datetime
from collections import defaultdict

def alpha_suffix(alpha: float) -> str:
    """Turn 0.1 -> '0p1' etc. for filenames."""
    s = str(alpha)
    return s.replace("-", "m").replace(".", "p")


def build_io_cmd(io_bin: Path, grammar: Path, yields: Path) -> List[str]:
    return [
        str(io_bin),
        "-d", "1000",
        "-m", "10",
        "-n", "50",
        "-W", "10",
        "-V",
        "-g", str(grammar),
        str(yields),
    ]


def run_cmd(cmd: List[str], cwd: Path, stdout=None) -> None:
    subprocess.run(cmd, check=True, cwd=cwd, stdout=stdout)

def log_print(msg: str, logf):
    print(msg)
    if logf:
        logf.write(msg + "\n")
        logf.flush()


def load_word_list(path: Path) -> Set[str]:
    """Load a newline-separated list of words; ignore blanks and comment lines."""
    words: Set[str] = set()
    if not path or not path.exists():
        return words
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            words.add(line)
    return words


def parse_results_for_lexicals(results_path: Path) -> Dict[str, List[Tuple[float, str, str]]]:
    """
    Return lexical rules grouped by nonterminal.
    Each entry is (prob, rhs_word, original_line).
    """
    entries: List[Tuple[float, str, List[str], str]] = []
    nonterminals: Set[str] = set()

    with results_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            stripped = line.strip()
            if not stripped or "-->" not in stripped:
                continue
            parts = stripped.split()
            try:
                prob = float(parts[0])
            except (ValueError, IndexError):
                continue
            try:
                arrow_idx = parts.index("-->")
            except ValueError:
                continue
            if arrow_idx < 1 or arrow_idx + 1 >= len(parts):
                continue
            lhs = parts[arrow_idx - 1]
            rhs = parts[arrow_idx + 1 :]
            nonterminals.add(lhs)
            entries.append((prob, lhs, rhs, stripped))

    lex_by_nt: Dict[str, List[Tuple[float, str, str]]] = defaultdict(list)
    for prob, lhs, rhs, stripped in entries:
        if len(rhs) == 1 and rhs[0] not in nonterminals:
            lex_by_nt[lhs].append((prob, rhs[0], stripped))

    for nt in lex_by_nt:
        lex_by_nt[nt].sort(key=lambda t: t[0], reverse=True)
    return lex_by_nt


def write_lines(path: Path, lines: List[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines))
            f.write("\n")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build grammars with per-part biases and train for multiple alpha settings."
    )
    parser.add_argument(
        "--prod-biases",
        type=float,
        nargs="+",
        required=True,
        help="Bias values to apply to production rules (multiple values → multiple grammars).",
    )
    parser.add_argument(
        "--lex-biases",
        type=float,
        nargs="+",
        required=True,
        help="Bias values to apply to lexical rules (multiple values → multiple grammars).",
    )
    parser.add_argument(
        "--productions",
        default="milas_grammars/grammar_NVA_VB_len2plus_27_reduced.lt",
        help="Productions (non-lexical) file.",
    )
    parser.add_argument(
        "--lexicon",
        default="milas_grammars/lexicon_len2plus_NVA_bitpar.lt",
        help="Lexicon file.",
    )
    parser.add_argument(
        "--prime-file",
        help="Optional file with word TAG bias [weight] to prime specific lexical entries.",
    )
    parser.add_argument(
        "--lexicalised",
        action="store_true",
        help="Pass to create_vocab_and_grammar to treat tags as lexicalised (L_TAG) when priming.",
    )
    parser.add_argument(
        "--yields",
        default="milas_grammars/brown_sentences_len2plus.txt",
        help="Yield (training sentences) file.",
    )
    parser.add_argument(
        "--grammar-dir",
        default="milas_grammars",
        help="Where to write generated grammars.",
    )
    parser.add_argument(
        "--output-dir",
        default="milas_grammars/results",
        help="Where to write training result files.",
    )
    parser.add_argument(
        "--io-bin",
        default="io",
        help="Path to the io binary (default: ./io when run from repo root).",
    )
    args = parser.parse_args()

    io_bin = Path(args.io_bin).resolve()
    productions = Path(args.productions).resolve()
    lexicon = Path(args.lexicon).resolve()
    yields = Path(args.yields).resolve()
    grammar_dir = Path(args.grammar_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    repo_root = Path(__file__).resolve().parent.parent

    grammar_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not io_bin.exists():
        raise FileNotFoundError(f"io binary not found at {io_bin}")
    if not productions.exists():
        raise FileNotFoundError(f"Productions file not found: {productions}")
    if not lexicon.exists():
        raise FileNotFoundError(f"Lexicon file not found: {lexicon}")
    if not yields.exists():
        raise FileNotFoundError(f"Yield file not found: {yields}")

    prime_flag: List[str] = []
    if args.prime_file:
        prime_flag += ["--prime-file", str(Path(args.prime_file).resolve())]
    if args.lexicalised:
        prime_flag.append("--lexicalised")

    primed_words_map: Dict[str, Set[str]] = defaultdict(set)
    if args.prime_file:
        prime_file_path = Path(args.prime_file).resolve()
        with prime_file_path.open("r", encoding="utf-8") as pf:
            for line in pf:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                # expected: WORD TAG [BIAS], but keep robust
                if len(parts) >= 1:
                    word = parts[0]
                    tag = parts[1] if len(parts) >= 2 else ""
                    primed_words_map[word].add(tag)
    primed_words = set(primed_words_map.keys())

    top_nouns = load_word_list(repo_root / "milas_grammars/topnouns.txt")
    # also accept alternate filename if present
    top_nouns |= load_word_list(repo_root / "milas_grammars/top_nouns.txt")
    top_verbs = load_word_list(repo_root / "milas_grammars/topverbs.txt")

    # build and train for every combination
    for prod_bias in args.prod_biases:
        for lex_bias in args.lex_biases:
            suffix = f"pb{alpha_suffix(prod_bias)}_lb{alpha_suffix(lex_bias)}"
            grammar_path = grammar_dir / f"{productions.stem}_{suffix}.lt"
            if args.prime_file:
                grammar_path =  grammar_dir / f"{productions.stem}_{suffix}_primed.lt"

            # build grammar via create_vocab_and_grammar.py combine
            combine_cmd: List[str] = [
                sys.executable,
                "milas_scripts/create_vocab_and_grammar.py",
                "combine",
                "-p",
                str(productions),
                "-l",
                str(lexicon),
                "-o",
                str(grammar_path),
                "--prod-bias",
                str(prod_bias),
                "--lex-bias",
                str(lex_bias),
            ] + prime_flag
            run_cmd(combine_cmd, cwd=repo_root)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            run_dir = output_dir / f"{grammar_path.stem}_{timestamp}"
            run_dir.mkdir(parents=True, exist_ok=True)
            outfile = run_dir / "results.txt"
            logfile = run_dir / "train.log"

            io_cmd = build_io_cmd(io_bin, grammar_path, yields)
            print(f"[prod_bias={prod_bias} lex_bias={lex_bias}] -> {outfile}")

            with outfile.open("w", encoding="utf-8") as fout, logfile.open("w", encoding="utf-8") as flog:
                log_print(f"[prod_bias={prod_bias} lex_bias={lex_bias}] -> {outfile}", flog)
                io_cmd = build_io_cmd(io_bin, grammar_path, yields)
                subprocess.run(io_cmd, check=True, cwd=repo_root, stdout=fout, stderr=flog)

            # with outfile.open("w", encoding="utf-8") as fout:
            #     run_cmd(io_cmd, cwd=repo_root, stdout=fout)

            # Collect lexicalisations and write summary files
            lex_by_nt = parse_results_for_lexicals(outfile)

            # Primed words
            if primed_words:
                primed_rules_path = run_dir / "primed_words_lexicalisations.txt"
                primed_lines: List[str] = []
                for nt, nt_rules in lex_by_nt.items():
                    for _, word, line in nt_rules:
                        if word in primed_words:
                            primed_lines.append(line)
                write_lines(primed_rules_path, primed_lines)
                log_print(f"Primed lexicalisations written to {primed_rules_path}", None)

            # Top lexicalisations per nonterminal (top 50 by prob)
            top_lex_lines: List[str] = []
            for nt in sorted(lex_by_nt.keys()):
                for prob, word, line in lex_by_nt[nt][:50]:
                    top_lex_lines.append(line)
            write_lines(run_dir / "top_lexicalisations.txt", top_lex_lines)

            def is_primed(word: str, nt: str) -> bool:
                if word not in primed_words_map:
                    return False
                tags = primed_words_map[word]
                return (not tags) or (nt in tags)

            # Top nouns (mark primed nouns) — include all lexicalisations for those words, grouped by word and sorted by prob
            noun_by_word: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
            for nt, nt_rules in lex_by_nt.items():
                for prob, word, line in nt_rules:
                    if word in top_nouns:
                        suffix = " #primed" if is_primed(word, nt) else ""
                        noun_by_word[word].append((prob, f"{line}{suffix}"))
            noun_lines: List[str] = []
            for word in sorted(noun_by_word.keys()):
                for _, lined in sorted(noun_by_word[word], key=lambda t: t[0], reverse=True):
                    noun_lines.append(lined)
            write_lines(run_dir / "top_nouns_lexicalisations.txt", noun_lines)

            # Top verbs — include all lexicalisations for those words, grouped by word and sorted by prob
            verb_by_word: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
            for nt, nt_rules in lex_by_nt.items():
                for prob, word, line in nt_rules:
                    if word in top_verbs:
                        suffix = " #primed" if is_primed(word, nt) else ""
                        verb_by_word[word].append((prob, f"{line}{suffix}"))
            verb_lines: List[str] = []
            for word in sorted(verb_by_word.keys()):
                for _, lined in sorted(verb_by_word[word], key=lambda t: t[0], reverse=True):
                    verb_lines.append(lined)
            write_lines(run_dir / "top_verbs_lexicalisations.txt", verb_lines)


if __name__ == "__main__":
    main()
