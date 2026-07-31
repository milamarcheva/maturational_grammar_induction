#!/usr/bin/env python3

import os
import re
import pandas as pd

INPUT_DIR = "."
OUTFILE = "staged_jsd_per_lhs_results_mmm.tsv"

rows = []

for fname in os.listdir(INPUT_DIR):

    if not fname.endswith(".jsd_per_lhs.out"):
        continue

    path = os.path.join(INPUT_DIR, fname)

    # ---------- stage ----------
    m = re.search(r"__([0-9]{2})_", fname)
    if not m:
        continue
    stage = m.group(1)

    # ---------- dir (everything before __<stage>_) ----------
    dir_name = fname.split(f"__{stage}_")[0]

    # ---------- parameters ----------
    sp_match = re.search(r"ps-([0-9p]+)", fname)
    sp = sp_match.group(1).replace("p", ".") if sp_match else ""

    sl_match = re.search(r"ls-([0-9p]+)", fname)
    sl = sl_match.group(1).replace("p", ".") if sl_match else ""

    eta_match = re.search(r"nbe-([0-9p]+)", fname)
    eta = eta_match.group(1).replace("p", ".") if eta_match else ""

    base = "MMM1and2"

    # ---------- read NT JSD values ----------
    jsd_vals = {}

    with open(path) as f:

        reading = False

        for line in f:

            line = line.strip()

            if line.startswith("JSD per nonterminal"):
                reading = True
                continue

            if reading and line:

                parts = line.split()

                if len(parts) == 2:
                    nt, val = parts
                    jsd_vals[nt] = float(val)

    row = {
        "dir": dir_name,
        "base": base,
        "stage": stage,
        "s_p": sp,
        "s_l": sl,
        "eta": eta
    }

    row.update(jsd_vals)

    rows.append(row)

df = pd.DataFrame(rows)

df = df.sort_values(["dir", "stage"])

df.to_csv(OUTFILE, sep="\t", index=False)

print("Wrote", OUTFILE)
