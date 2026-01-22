#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_nltk_eval.sh INDUCED_GRAMMAR ORACLE_GRAMMAR SENTENCES [GOLD_PARSES] [-- EXTRA_ARGS...]

Runs nltk_eval.py with parse evaluation (labeled + unlabeled) and JSD enabled.
EXTRA_ARGS are passed through to nltk_eval.py (e.g., --weight-index -1 --prune-lexicon).
EOF
}

if [[ $# -lt 3 ]]; then
  usage
  exit 1
fi

induced_grammar=$1
oracle_grammar=$2
sentences=$3
gold_parses=""
if [[ $# -ge 4 && "${4}" != -* ]]; then
  gold_parses=$4
  shift 4
else
  shift 3
fi
if [[ ${1:-} == "--" ]]; then
  shift 1
fi
extra_args=("$@")

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
eval_script="${script_dir}/nltk_eval.py"

cmd=(
  python3 "$eval_script"
  --induced-grammar "$induced_grammar"
  --oracle-grammar "$oracle_grammar"
  --sentences "$sentences"
  --eval-parse
  --eval-jsd
)

if [[ -n "$gold_parses" ]]; then
  cmd+=(--gold-parses "$gold_parses")
fi

if [[ ${#extra_args[@]} -gt 0 ]]; then
  cmd+=("${extra_args[@]}")
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
