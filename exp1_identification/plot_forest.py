"""Forest plot of result 1: how often the true account outscores its distractors.

One row per lineup type. All rows use the SAME 40 conversations, the same
listener and K=8 distractors, so the only thing that varies down the axis is
what the distractors are accounts of. Higher is better; 0.5 is chance.
"""
import argparse, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, MUT, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e2e1dd"
RIVAL, DONOR, FLOOR = "#7a5af5", "#52514e", "#1baf7a"   # dataviz slots

# (key, label, colour, marker). Bottom-to-top, so the floor ends up on top.
ROWS = [
    ("donor_matched_n4",  "mechanical, 4 words",              DONOR, "X"),
    ("rival_n4_judged",   "rival, 4 claims, judge-confirmed", RIVAL, "o"),
    ("rival_n4",          "rival, 4 claims",                  RIVAL, "o"),
    ("donor_matched_n2",  "mechanical, 2 words",              DONOR, "X"),
    ("rival_n2_judged",   "rival, 2 claims, judge-confirmed", RIVAL, "o"),
    ("rival_n2",          "rival, 2 claims",                  RIVAL, "o"),
    ("donor_matched_n1",  "mechanical, 1 word",               DONOR, "X"),
    ("rival_n1_judged",   "rival, 1 claim, judge-confirmed",  RIVAL, "o"),
    ("rival_n1",          "rival, 1 claim",                   RIVAL, "o"),
    ("other_conversation","accounts of other activations",    FLOOR, "s"),
]

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="results/rivals.json")
ap.add_argument("--out", default="results/figures/rival_forest.png")
a = ap.parse_args()

cond = json.load(open(a.data))["conditions"]
rng = np.random.default_rng(0)

fig, ax = plt.subplots(figsize=(8.6, 4.9), dpi=160)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
# Faint bands group each claim count with its own matched mechanical control,
# because the rival-vs-mechanical comparison is the one the eye should make.
# Rows 0-2 are the 4-claim group, 3-5 the 2-claim, 6-8 the 1-claim; shade
# alternate groups so each rival pair reads against its own mechanical control.
for y0 in (-0.5, 5.5):
    ax.axhspan(y0, y0 + 3, color=GRID, alpha=0.35, zorder=0)
ax.axvline(0.5, color=MUT, lw=1.2, ls=":", zorder=1)
ax.axvline(1.0, color=MUT, lw=1.0, ls="-", alpha=0.35, zorder=1)

labels = []
for y, (key, label, colour, marker) in enumerate(ROWS):
    v = np.asarray(cond[key]["per_conv_dir"], float)
    v = v[np.isfinite(v)]
    m = v.mean()
    # CI by resampling CONVERSATIONS, the independent unit. Degenerate when the
    # true account beat every distractor everywhere, which is the point of the
    # floor row, so it is drawn as a bare marker.
    bs = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(10000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    if hi > lo:
        ax.plot([lo, hi], [y, y], color=colour, lw=2.0, alpha=0.85, zorder=2,
                solid_capstyle="butt")
    ax.plot([m], [y], marker=marker, ms=7, color=colour, zorder=3,
            markeredgecolor=SURFACE, markeredgewidth=0.8)
    ax.text(1.045, y, f"{m:.3f}", va="center", ha="left", fontsize=9,
            color=INK, family="monospace")
    labels.append(f"{label}  (n={len(v)})")

ax.set_yticks(range(len(ROWS)), labels, fontsize=9.5, color=INK)
ax.set_ylim(-0.7, len(ROWS) - 0.3)
ax.set_xlim(0.28, 1.14)
ax.set_xticks(np.arange(0.3, 1.01, 0.1))
ax.set_xlabel("fraction of the 8 distractors that the true account outscores\n"
              "(mean over conversations; bars are 95% CI over conversations)",
              fontsize=9.5, color=MUT)
ax.set_title("Can the reconstructor tell the verbaliser's own account\n"
             "from a fluent, verifiably false one?",
             fontsize=12, color=INK, loc="left", pad=12)
ax.text(0.5, -0.62, "chance", fontsize=9, color=MUT, va="bottom", ha="center")
ax.text(1.0, -0.62, "perfect", fontsize=9, color=MUT, va="bottom", ha="center")
ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=MUT, length=0)

fig.tight_layout()
fig.savefig(a.out, facecolor=SURFACE)
print("wrote", a.out)
