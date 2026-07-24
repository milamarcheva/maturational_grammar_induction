#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path
from typing import Dict

EVAL_DIRS = [
    "bg_eval_outputs_allstages",
    "bg_eval_outputs_finalstage",
]

MLL_DIRS = [
    "bg_mll_outputs_allstages",
    "bg_mll_outputs_finalstage",
]

MORPHTOK_DIRS = [
    "morphtok_bg_eval_outputs_allstages",
    "morphtok_bg_eval_outputs_finalstage",
]

OUT_HEADERS = [
    "base",
    "s_p",
    "s_l",
    "eta",
    "gold_F1_u",
    "mll",
    "JSD",
    "stage",
    "dir",
    "file_eval",
    "file_mll",
]

MORPHTOK_HEADERS = [
    "grammar",
    "base",
    "stage",
    "s_p",
    "s_l",
    "eta",
    "f1",
    "JSD",
    "mll",
    "dir",
    "file",
]

PARAM_RE = re.compile(
    r"__ps-(?P<ps>[^_]+)"
    r".*?__nbe-(?P<nbe>[^_]+)"
    r".*?__ls-(?P<ls>[^_]+)"
)

STAGE_RE = re.compile(r"__(\d{2})_[A-Za-z0-9]+$")
MORPHTOK_STAGE_BASE_RE = re.compile(
    r"__(?P<stage>\d{2})_(?P<base>[A-Za-z]+)(?P<stage_num>\d+)$"
)


def extract_gold_f1(text: str) -> str:
    """
    Extract:
    Parse evaluation (gold parses)
    -> Unlabeled spans
    -> f1
    """
    pattern = re.compile(
        r"Parse evaluation \(gold parses\).*?"
        r"Unlabeled spans.*?"
        r"- f1:\s*([-+]?\d+\.\d+)",
        re.S,
    )
    m = pattern.search(text)
    return m.group(1) if m else ""


def extract_jsd(text: str) -> str:
    """
    Extract from eval.out:
    JSD evaluation
    - mean JSD (base 2.0): X
    """
    m = re.search(r"- mean JSD \(base [0-9.]+\):\s*([-+]?\d+\.\d+)", text)
    return m.group(1) if m else ""


def extract_mll(text: str) -> str:
    """
    Extract from mll.out:
    Marginal log-likelihood
    - mean log-likelihood: X
    """
    m = re.search(r"- mean log-likelihood:\s*([-+]?\d+\.\d+)", text)
    return m.group(1) if m else ""


def extract_mll_normalized(text: str) -> str:
    """
    Extract from mll.out:
    - mean normalized log-likelihood (per token, sentence avg): X
    """
    m = re.search(
        r"- mean normalized log-likelihood "
        r"\(per token, sentence avg\):\s*([-+]?\d+\.\d+)",
        text,
    )
    return m.group(1) if m else ""


def strip_suffix(name: str) -> str:
    """
    Remove known output suffixes.
    Order matters: longest first.
    """
    suffixes = [
        ".eval.out",
        ".eval.err",
        ".mll.out",
        ".mll.err",
        ".out",
        ".err",
    ]
    for suf in suffixes:
        if name.endswith(suf):
            return name[: -len(suf)]
    return Path(name).stem


def decode(x: str) -> str:
    """
    Convert e.g. 0p001 -> 0.001
    """
    return x.replace("p", ".") if "p" in x and "." not in x else x


def extract_stage(stem: str) -> str:
    """
    Extract stage from trailing __04_BGG4 -> 4
    """
    m = STAGE_RE.search(stem)
    return str(int(m.group(1))) if m else ""


def derive_dir(stem: str) -> str:
    """
    Remove trailing stage marker, e.g.
    ...__04_BGG4 -> ...
    """
    return STAGE_RE.sub("", stem)


def extract_morphtok_stage_and_base(stem: str):
    """
    Extract stage/base from trailing __05_BGMTMMM5 -> ("5", "BGMTMMM")
    """
    m = MORPHTOK_STAGE_BASE_RE.search(stem)
    if not m:
        return "", ""
    return str(int(m.group("stage"))), m.group("base")


def extract_params(dir_name: str):
    """
    Extract:
      s_p from __ps-...
      eta from __nbe-...
      s_l from __ls-...
    """
    m = PARAM_RE.search(dir_name)
    if not m:
        return "", "", ""
    return (
        decode(m.group("ps")),   # s_p
        decode(m.group("ls")),   # s_l
        decode(m.group("nbe")),  # eta
    )


def extract_base(dir_name: str) -> str:
    """
    Base = prefix before __ps-..., with leading 'stages_' removed.
    """
    prefix = dir_name.split("__ps-")[0]
    if prefix.startswith("stages_"):
        prefix = prefix[len("stages_"):]
    parts = prefix.split("_")
    for p in parts:
        if p.startswith("base"):
            return p
    return prefix


def extract_morphtok_grammar(dir_name: str) -> str:
    """
    Grammar is the segment after stages_bg_stages_morphtok_, e.g. mv or
    merged_min5. Some runs are named with minN directly, so normalize those
    to merged_minN in the output.
    """
    prefix = dir_name.split("__ps-")[0]
    needle = "stages_bg_stages_morphtok_"
    if not prefix.startswith(needle):
        return ""

    rest = prefix[len(needle):]
    if rest.startswith("mv_"):
        return "mv"
    m = re.match(r"(?:(merged_min\d+)|(min\d+))_", rest)
    if m:
        return m.group(1) or f"merged_{m.group(2)}"
    return ""


def iter_candidate_dirs(root: Path, dir_names):
    """
    Yield root itself if it directly contains outputs; otherwise yield matching child dirs.
    """
    if any(root.glob("*.eval.out")) or any(root.glob("*.mll.out")):
        yield root
        return

    for dname in dir_names:
        d = root / dname
        if d.exists():
            yield d


def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def collect_eval(root: Path) -> Dict[str, Dict]:
    """
    Keyed by eval stem (includes stage suffix if present).
    gold_F1_u and JSD both come from eval.out.
    """
    rows = {}

    for dname in EVAL_DIRS:
        d = root / dname
        if not d.exists():
            continue

        for f in d.glob("*.eval.out"):
            stem = strip_suffix(f.name)
            dir_name = derive_dir(stem)
            text = read_file(f)
            s_p, s_l, eta = extract_params(dir_name)

            rows[stem] = {
                "base": extract_base(dir_name),
                "s_p": s_p,
                "s_l": s_l,
                "eta": eta,
                "gold_F1_u": extract_gold_f1(text),
                "mll": "",
                "JSD": extract_jsd(text),
                "stage": extract_stage(stem),
                "dir": dir_name,
                "file_eval": f.name,
                "file_mll": "",
            }

    return rows


def collect_mll(root: Path) -> Dict[str, Dict]:
    """
    Keyed by exact dir name, not stem.
    Only MLL comes from mll.out.
    """
    rows = {}

    for dname in MLL_DIRS:
        d = root / dname
        if not d.exists():
            continue

        for f in d.glob("*.out"):
            stem = strip_suffix(f.name)
            dir_name = derive_dir(stem)
            text = read_file(f)

            rows[dir_name] = {
                "mll": extract_mll(text),
                "file_mll": f.name,
            }

    return rows


def collect_morphtok(root: Path) -> Dict[str, Dict]:
    """
    Morphtok outputs keep eval and MLL files in the same directory.
    Key rows by the shared stem.
    """
    rows: Dict[str, Dict] = {}

    for d in iter_candidate_dirs(root, MORPHTOK_DIRS):
        for f in d.glob("*.eval.out"):
            stem = strip_suffix(f.name)
            dir_name = derive_dir(stem)
            stage, base = extract_morphtok_stage_and_base(stem)
            text = read_file(f)
            s_p, s_l, eta = extract_params(dir_name)

            rows.setdefault(stem, {})
            rows[stem].update(
                {
                    "grammar": extract_morphtok_grammar(dir_name),
                    "base": base,
                    "stage": stage,
                    "s_p": s_p,
                    "s_l": s_l,
                    "eta": eta,
                    "f1": extract_gold_f1(text),
                    "JSD": extract_jsd(text),
                    "mll": rows[stem].get("mll", ""),
                    "dir": dir_name,
                    "file": stem,
                }
            )

        for f in d.glob("*.mll.out"):
            stem = strip_suffix(f.name)
            dir_name = derive_dir(stem)
            stage, base = extract_morphtok_stage_and_base(stem)
            text = read_file(f)
            s_p, s_l, eta = extract_params(dir_name)

            rows.setdefault(stem, {})
            rows[stem].update(
                {
                    "grammar": rows[stem].get("grammar") or extract_morphtok_grammar(dir_name),
                    "base": rows[stem].get("base") or base,
                    "stage": rows[stem].get("stage") or stage,
                    "s_p": rows[stem].get("s_p") or s_p,
                    "s_l": rows[stem].get("s_l") or s_l,
                    "eta": rows[stem].get("eta") or eta,
                    "f1": rows[stem].get("f1", ""),
                    "JSD": rows[stem].get("JSD", ""),
                    "mll": extract_mll_normalized(text),
                    "dir": rows[stem].get("dir") or dir_name,
                    "file": stem,
                }
            )

    return rows


def merge(eval_rows, mll_rows):
    merged = []

    for _, e in eval_rows.items():
        m = mll_rows.get(e["dir"], {})

        if not m:
            print("⚠️ Missing MLL for:", e["dir"])

        row = dict(e)
        row["mll"] = m.get("mll", "")
        row["file_mll"] = m.get("file_mll", "")
        merged.append(row)

    return sorted(
        merged,
        key=lambda x: (x["dir"], int(x["stage"]) if x["stage"] else 0)
    )


def sort_morphtok_rows(rows):
    return sorted(
        rows,
        key=lambda x: (
            x["grammar"],
            x["base"],
            int(x["stage"]) if x["stage"] else 0,
            x["dir"],
            x["file"],
        ),
    )


def write_table(rows, path: Path, headers):
    delimiter = "," if path.suffix.lower() == ".csv" else "\t"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=".")
    parser.add_argument("--out", type=Path, default="bg_results.tsv")
    parser.add_argument("--morphtok", "--moprhtok", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()

    if args.morphtok:
        merged = sort_morphtok_rows(list(collect_morphtok(root).values()))
        write_table(merged, args.out, MORPHTOK_HEADERS)
    else:
        eval_rows = collect_eval(root)
        mll_rows = collect_mll(root)
        merged = merge(eval_rows, mll_rows)
        write_table(merged, args.out, OUT_HEADERS)

    print(f"✅ Wrote {len(merged)} rows → {args.out}")


if __name__ == "__main__":
    main()
