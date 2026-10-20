#!/usr/bin/env bash
# Experiment 1 (identification without discrimination), reproduction from
# artefacts: rescore the released rivals with the AR and redraw the figures. About 5 minutes on a GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && . .venv/bin/activate
export PYTHONPATH=nla
mkdir -p results/figures

python -u exp1_identification/rivals.py --rescore            # results/rivals.json
python exp1_identification/plot_forest.py                     # Figure: forest plot
python exp1_identification/change_cloud.py --jsonl artefacts/rival_texts.jsonl \
    --orig orig --mod 'rivals.*' --out results/figures/rival_cloud.png \
    --title "What the rivals change"                          # Figure: word cloud
python exp1_identification/angular_span.py                    # Figure: angular span (CPU)
python exp1_identification/mechanical_table.py                # Table: mechanical baselines (CPU)
