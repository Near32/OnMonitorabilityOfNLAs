"""How large an angle does a confabulation actually move the reconstruction?

The critic L2-normalises both vectors before scoring, so its MSE is exactly
2(1 - cos) and every quantity it can see is an angle. That makes one question
answerable in closed form: how big is the angular displacement a confabulation
causes, measured against the angles the AR is already living with?

Three reference angles matter, and the answer is the ratio between them:

    AR(true) vs the gold activation   the AR's own reconstruction error --
                                      the noise floor any signal must clear
    AR(true) vs AR(distractor)        the signal, per dose
    AR(true_i) vs AR(true_j)          the ambient spread across conversations

Raw and mean-centred are both reported. Raw is what the critic scores. Hidden
states are strongly anisotropic -- a random pair of 3584-dim Gaussians sits at
90 degrees, but two unrelated AR outputs sit at 45 -- so the raw numbers are
compressed by a shared cone that carries no conversation-specific information.
Centring removes it and roughly doubles every angle, which changes the scale
but not a single ordering or ratio.

Reads the probe cache, so it needs no GPU and can run beside a training job.

    python exp1_identification/angular_span.py
"""
import argparse, json, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, MUT, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e2e1dd"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def angle(a, b):
    """Degrees between a and b, broadcasting over leading axes."""
    return np.degrees(np.arccos(np.clip((unit(a) * unit(b)).sum(-1), -1, 1)))


def summarise(per_conv, rng=None, boot=10000):
    """Mean +/- standard error over CONVERSATIONS.

    The 13 distractors inside one conversation are 13 corruptions of a single
    explanation, so they are not independent draws and pooling them would
    shrink the error bar by a factor of sqrt(13) it has not earned. Callers
    therefore pass one number per conversation (already averaged within it),
    and n is the number of conversations. The bootstrap resamples
    conversations for the same reason."""
    v = np.asarray(per_conv, dtype=float).ravel()
    n = len(v)
    se = float(v.std(ddof=1) / np.sqrt(n))
    out = dict(mean=float(v.mean()), se=se, sd=float(v.std(ddof=1)),
               lo=float(v.min()), hi=float(v.max()), n=n)
    if rng is not None:
        bs = np.array([v[rng.integers(0, n, n)].mean() for _ in range(boot)])
        out["ci95"] = [float(x) for x in np.percentile(bs, [2.5, 97.5])]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="artefacts/probe_cache.npz")
    p.add_argument("--out", default="results/figures/angular_span.png")
    p.add_argument("--json-out", default="results/angular_span.json")
    a = p.parse_args()

    z = np.load(a.cache)
    H, T, D, doses = z["H"], z["T"], z["D"], z["doses"]
    iu = np.triu_indices(len(T), 1)

    # Centring is computed over true AND distractor outputs together, so the
    # mean removed is the AR's output cone rather than anything that separates
    # the two classes.
    flat = np.concatenate([T[:, None, :], D.reshape(len(T), -1, T.shape[1])], 1)
    mu = flat.reshape(-1, T.shape[1]).mean(0)

    rng = np.random.default_rng(0)
    err = angle(T, H)                      # (n_conv,) one number per conversation
    res = {"n_conv": int(len(T)), "K": int(D.shape[2]),
           "doses": [int(d) for d in doses]}
    for tag, (t, d) in {"raw": (T, D), "centred": (T - mu, D - mu)}.items():
        # Average within a conversation FIRST, so every entry below is a
        # 60-sample statistic over conversations rather than a 780-sample one
        # over correlated corruptions.
        res[tag] = {
            "recon_error": summarise(err, rng) if tag == "raw" else None,
            "between_conv": summarise(
                [np.delete(angle(t[i][None, :], t), i).mean()
                 for i in range(len(t))], rng),
            "dose": {int(dd): summarise(angle(t[:, None, :], d[:, i]).mean(1), rng)
                     for i, dd in enumerate(doses)},
        }
    # Paired against each conversation's OWN reconstruction error, which is
    # the comparison that matters: a displacement only becomes visible to the
    # reward if it exceeds the error on that same sample.
    res["vs_own_error"] = {}
    for i, dd in enumerate(doses):
        pc = angle(T[:, None, :], D[:, i]).mean(1)
        res["vs_own_error"][int(dd)] = dict(
            ratio=summarise(pc / err, rng),
            n_exceeding=int((pc > err).sum()), n=int(len(pc)))
    # Displacement and error are both angles, but a ratio between them implies
    # they are commensurable and this is the check that they are not. If the
    # corruption moves the reconstruction PERPENDICULAR to the error it already
    # carries, then the error changes only to second order, and what the reward
    # actually computes is far smaller than the displacement itself.
    ev = unit(H) - unit(T)
    res["orthogonality"] = {}
    for i, dd in enumerate(doses):
        disp = unit(D[:, i]) - unit(T)[:, None, :]
        res["orthogonality"][int(dd)] = dict(
            angle_to_error=summarise(angle(disp, ev[:, None, :]).mean(1), rng),
            error_change=summarise(
                (angle(D[:, i], H[:, None, :]) - err[:, None]).mean(1), rng))

    r = rng.normal(size=(400, H.shape[1]))
    res["random_pair"] = summarise(angle(r[:200], r[200:]))
    res["gold_between_conv"] = summarise(
        [np.delete(angle(H[i][None, :], H), i).mean() for i in range(len(H))])

    floor = res["raw"]["recon_error"]["mean"]
    xs = np.array([res["raw"]["dose"][int(d)]["mean"] for d in doses])
    # Displacement grows as a power of the dose; fitting in logs gives the
    # dose at which the signal would finally equal the AR's own error.
    k, logc = (float(v) for v in np.polyfit(np.log(doses.astype(float)),
                                            np.log(xs.astype(float)), 1))
    res["power_law"] = {"exponent": float(k),
                        "dose_at_recon_error": float(np.exp((np.log(floor) - logc) / k))}

    print(f"n={res['n_conv']} conversations, K={res['K']} distractors/dose\n")
    fmt = lambda s: f"{s['mean']:6.2f} +/- {s['se']:.2f}deg"
    print(f"  {'AR(true) vs gold activation':<32}"
          f"{fmt(res['raw']['recon_error'])}   <- noise floor")
    for d in doses:
        s, v = res["raw"]["dose"][int(d)], res["vs_own_error"][int(d)]
        print(f"  {'AR(true) vs AR(distractor) d=' + str(d):<32}{fmt(s)}   "
              f"ratio to own error {v['ratio']['mean']:.3f} "
              f"+/- {v['ratio']['se']:.3f}   exceeds it in "
              f"{v['n_exceeding']}/{v['n']} convs")
    print(f"  {'AR(true_i) vs AR(true_j)':<32}{fmt(res['raw']['between_conv'])}")
    print(f"  {'random 3584-dim pair':<32}{fmt(res['random_pair'])}")
    print(f"\n  displacement ~ dose^{k:.2f}; reaches the reconstruction error "
          f"at dose ~{res['power_law']['dose_at_recon_error']:.0f}")
    print("\n  BUT the displacement is nearly ORTHOGONAL to the error, so the "
          "ratio above\n  overstates what the reward can see. What it computes "
          "is the CHANGE in error:")
    for dd in doses:
        o = res["orthogonality"][int(dd)]
        print(f"    dose {dd:<3} displacement is {o['angle_to_error']['mean']:5.2f}"
              f" deg from the error direction -> error changes by "
              f"{o['error_change']['mean']:+.4f} deg "
              f"({100 * o['error_change']['mean'] / floor:.3f}% of it)")

    # --- figure ---------------------------------------------------------
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), facecolor=SURFACE,
                                  gridspec_kw=dict(width_ratios=[1.15, 1],
                                                   wspace=0.24))
    for x in (ax, ax2):
        x.set_facecolor(SURFACE)
        x.grid(alpha=0.55, color=GRID, lw=0.8)
        x.set_axisbelow(True)
        for s in x.spines.values():
            s.set_color(GRID)
        x.tick_params(colors=MUT, labelsize=9)

    rows = ([(f"confabulation, dose {d}", res["raw"]["dose"][int(d)], BLUE)
             for d in doses]
            + [("AR reconstruction error", res["raw"]["recon_error"], ORANGE),
               ("two different conversations", res["raw"]["between_conv"], AQUA),
               ("two random vectors", res["random_pair"], MUT)])
    y = np.arange(len(rows))[::-1]
    for yi, (name, s, c) in zip(y, rows):
        ax.plot([s["lo"], s["hi"]], [yi, yi], "-", color=c, lw=2, alpha=0.3,
                solid_capstyle="round")
        ax.plot([s["mean"] - s["se"], s["mean"] + s["se"]], [yi, yi], "-",
                color=c, lw=6, alpha=0.75, solid_capstyle="butt")
        ax.plot(s["mean"], yi, "o", color=c, ms=9, mec=SURFACE, mew=1.6,
                zorder=4)
        ax.annotate(f"{s['mean']:.1f}°", (s["mean"], yi),
                    textcoords="offset points", xytext=(0, 10), ha="center",
                    fontsize=8.5, color=INK)
        ax.annotate(f"±{s['se']:.2f}", (s["mean"], yi),
                    textcoords="offset points", xytext=(0, -16), ha="center",
                    fontsize=7.5, color=MUT)
    ax.set_yticks(y, [r[0] for r in rows], color=INK, fontsize=9.5)
    ax.set_ylim(-0.75, len(rows) - 0.4)   # room for the SE annotations
    ax.set_xscale("log")
    ax.set_xlabel("angle (degrees, log scale) — dot = mean over conversations, "
                  "bar = ±1 SE, line = full range", color=MUT, fontsize=9.5)
    ax.axvline(floor, color=ORANGE, ls="--", lw=1.2, alpha=0.8, zorder=1)
    ax.set_title("A one-word confabulation moves the reconstruction\n"
                 f"{floor / res['raw']['dose'][int(doses[0])]['mean']:.0f}× "
                 "less than the AR is already wrong by",
                 color=INK, fontsize=11.5, loc="left", pad=8)

    ses = np.array([res["raw"]["dose"][int(d)]["se"] for d in doses])
    ax2.errorbar(doses, xs, yerr=ses, fmt="-o", color=BLUE, lw=2, ms=8,
                 mec=SURFACE, mew=1.6, capsize=3, ecolor=BLUE,
                 label="confabulation displacement (±1 SE)", zorder=4)
    grid = np.linspace(1, max(res["power_law"]["dose_at_recon_error"] * 1.3, 40), 200)
    ax2.plot(grid, np.exp(logc) * grid ** k, ":", color=BLUE, lw=1.4, alpha=0.8,
             label=f"fit $\\propto$ dose$^{{{k:.2f}}}$")
    fse = res["raw"]["recon_error"]["se"]
    ax2.axhspan(floor - fse, floor + fse, color=ORANGE, alpha=0.18, lw=0)
    ax2.axhline(floor, color=ORANGE, ls="--", lw=1.6,
                label="AR reconstruction error (±1 SE)")
    ax2.axhline(res["raw"]["between_conv"]["mean"], color=AQUA, ls="-.", lw=1.6,
                label="between-conversation spread")
    ax2.plot(res["power_law"]["dose_at_recon_error"], floor, "X", color=ORANGE,
             ms=13, mec=SURFACE, mew=1.6, zorder=5)
    ax2.annotate(f"signal = noise at\n~{res['power_law']['dose_at_recon_error']:.0f}"
                 " corrupted tokens",
                 (res["power_law"]["dose_at_recon_error"], floor),
                 textcoords="offset points", xytext=(-8, -34), ha="right",
                 fontsize=8.5, color=INK)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    cross = res["power_law"]["dose_at_recon_error"]
    ax2.set_xticks(list(doses) + [round(cross)],
                   [str(int(d)) for d in doses] + [f"~{cross:.0f}"])
    ax2.set_xlabel("corrupted tokens (dose)", color=MUT, fontsize=9.5)
    ax2.set_ylabel("angular displacement (degrees)", color=MUT, fontsize=9.5)
    ax2.set_title("The signal stays below the noise floor\nover the whole "
                  "measured range", color=INK, fontsize=11.5, loc="left", pad=8)
    ax2.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK)

    fig.tight_layout()
    fig.subplots_adjust(left=0.155)   # room for the full category labels
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=170, facecolor=SURFACE)
    json.dump(res, open(a.json_out, "w"), indent=1)
    print(f"\nwrote {a.out} and {a.json_out}")


if __name__ == "__main__":
    main()
