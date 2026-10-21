#!/usr/bin/env bash
# Experiment 2, reproduction from scratch: generate and judge the paraphrases,
# then score. N=100 uses 100 conversations instead of the paper's 60.
# Run exp1_identification/run_full.sh first: it writes the explanations,
# corruptions, activations and rivals this reads. About 40 minutes.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && . .venv/bin/activate
export PYTHONPATH=nla
if [ -e artefacts/paraphrase_texts.jsonl ]; then
  echo "artefacts/ already holds paraphrases: delete them to regenerate" >&2
  exit 1
fi
mkdir -p results/figures
n() { echo "${N:-$1}"; }

python -u exp2_paraphrase/paraphrase.py --n "$(n 60)"
python exp2_paraphrase/analyse_paraphrase.py
python -u exp2_paraphrase/rival_rows.py
python exp2_paraphrase/analyse_paraphrase.py \
    --data results/paraphrase_plus_rivals.json \
    --out results/paraphrase_plus_rivals_analysis.json \
    --fig results/figures/rivals_on_paraphrase_axes.png
