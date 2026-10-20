#!/usr/bin/env bash
# Experiment 2 (the failure is not a private dialect), reproduction from
# artefacts: rescore the released paraphrases, and put the rivals on the same damage axes (the error
# increases quoted in Experiment 1). About 10 minutes on a GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && . .venv/bin/activate
export PYTHONPATH=nla
mkdir -p results/figures

python -u exp2_paraphrase/paraphrase.py --n 60 --rescore      # results/paraphrase_n60.json
python exp2_paraphrase/analyse_paraphrase.py                  # Figure: paraphrase invariance
python -u exp2_paraphrase/rival_rows.py                       # results/paraphrase_plus_rivals.json
python exp2_paraphrase/analyse_paraphrase.py \
    --data results/paraphrase_plus_rivals.json \
    --out results/paraphrase_plus_rivals_analysis.json \
    --fig results/figures/rivals_on_paraphrase_axes.png
