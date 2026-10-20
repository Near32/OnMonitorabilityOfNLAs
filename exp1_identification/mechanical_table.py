"""Table of mechanical baselines: fraction of distractors the true account beats
at one changed word, from the released hard-negative CSVs (one row per
conversation and dose, written by hardneg.py).

    python exp1_identification/mechanical_table.py
"""
import argparse, csv, json, statistics

ROWS = [("topp0.9", "hardneg_topp09.csv"),
        ("donor", "hardneg_donor.csv"),
        ("donor_pos_absent", "hardneg_donor_pos_absent.csv"),
        ("donor_pos_absent_distance", "hardneg_donor_pos_far.csv"),
        ("donor_pos", "hardneg_donor_pos.csv"),
        ("topk", "hardneg_topk.csv")]

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="artefacts/hardneg")
ap.add_argument("--out", default="results/mechanical_table.json")
a = ap.parse_args()

out = {}
for name, f in ROWS:
    # rank 1 = the true account beats all K distractors, so 1 - (rank-1)/K is
    # the fraction it beats.
    y = [1 - (int(r["rank"]) - 1) / int(r["n_distractors"])
         for r in csv.DictReader(open(f"{a.dir}/{f}")) if r["dose"] == "1"]
    out[name] = dict(mean=statistics.mean(y),
                     se=statistics.stdev(y) / len(y) ** 0.5, n=len(y))
    print(f"{name:<28}{out[name]['mean']:.3f}  SE {out[name]['se']:.3f}  n={len(y)}")
json.dump(out, open(a.out, "w"), indent=1)
print("wrote", a.out)
