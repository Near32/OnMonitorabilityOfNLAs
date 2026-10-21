#!/usr/bin/env bash
# Experiment 1, reproduction from scratch: regenerate every artefact from
# WildChat, then score. Needs ./setup.sh --with-av and WildChat-1M shard 0.
# Each script holds one 7B model at a time (~16 GB of GPU memory).
# Writes a fresh artefacts/; move a downloaded one aside first.
#
#   N=100 bash exp1_identification/run_full.sh   every stage on 100 conversations
#
# Without N, each stage uses the paper's size (60, 100, 40 conversations).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && . .venv/bin/activate
export PYTHONPATH=nla
if [ -e artefacts/rival_texts.jsonl ]; then
  echo "artefacts/ already holds rivals: move it aside (mv artefacts artefacts.released)" >&2
  exit 1
fi
n() { echo "${N:-$1}"; }
mkdir -p artefacts/traces artefacts/hardneg artefacts/cache results/figures
A=artefacts; H="python -u exp1_identification/hardneg.py --n-distractors 15 --position-filter content"

# The AV's explanations with POS-matched donor corruptions: the true accounts
# used by every later stage. Runs that need the AV's scores come first; the
# donor runs after them reuse the cached explanations and load no AV.
$H --n-samples "$(n 60)" --doses 1,2,4,8,16 --candidate-mode donor_pos \
   --out $A/hardneg/hardneg_donor_pos.csv --summary $A/hardneg/hardneg_donor_pos.json \
   --trace $A/traces/hardneg_donor_pos_traces.jsonl
# The AV's own top-k substitutions. The paraphrase control uses the trace of
# its first 30 conversations at doses 1, 4, 16: each (conversation, dose) has
# its own seed, so that subset is what a separate 30-conversation run writes.
$H --n-samples "$(n 100)" --doses 1,2,4,8,16 --candidate-mode topk \
   --out $A/hardneg/hardneg_topk.csv --summary $A/hardneg/hardneg_topk.json \
   --trace $A/cache/hardneg_topk_traces.jsonl
python - "$(n 30)" <<'EOF2'
import json, sys
n = int(sys.argv[1])
with open("artefacts/cache/hardneg_topk_traces.jsonl") as f, \
     open("artefacts/traces/hardneg_traces.jsonl", "w") as out:
    for line in f:
        r = json.loads(line)
        if r["sample"] < n and r["dose"] in (1, 4, 16):
            out.write(line)
EOF2
# The other mechanical baselines of the table.
$H --n-samples "$(n 100)" --doses 1,2,4,8,16 --candidate-mode topp --top-p 0.9 \
   --out $A/hardneg/hardneg_topp09.csv --summary $A/hardneg/hardneg_topp09.json
$H --n-samples "$(n 60)" --doses 1,2,4,8,16 --candidate-mode donor \
   --out $A/hardneg/hardneg_donor.csv --summary $A/hardneg/hardneg_donor.json
$H --n-samples "$(n 60)" --doses 1,2,4,8,16 --candidate-mode donor_pos --donor-absent-only \
   --out $A/hardneg/hardneg_donor_pos_absent.csv --summary $A/hardneg/hardneg_donor_pos_absent.json
$H --n-samples "$(n 60)" --doses 1,2,4,8,16 --candidate-mode donor_pos --donor-absent-only \
   --min-donor-distance 0.13 \
   --out $A/hardneg/hardneg_donor_pos_far.csv --summary $A/hardneg/hardneg_donor_pos_far.json

# Gold activations and AR reconstructions (also trains a probe the paper does
# not use); the gold activations are then cut out of the probe cache.
python -u exp1_identification/probe_ar.py
python - <<'EOF2'
import numpy as np
z = np.load("artefacts/probe_cache.npz")
np.savez_compressed("artefacts/gold_activations.npz", conv=z["conv"], H=z["H"])
EOF2

# Rivals, judge (validated first), scoring.
python -u exp1_identification/rivals.py --n "$(n 40)" --claims 1,2,4 --k 8

python exp1_identification/plot_forest.py
python exp1_identification/change_cloud.py --jsonl artefacts/rival_texts.jsonl \
    --orig orig --mod 'rivals.*' --out results/figures/rival_cloud.png \
    --title "What the rivals change"
python exp1_identification/angular_span.py
python exp1_identification/mechanical_table.py
