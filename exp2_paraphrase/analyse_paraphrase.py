"""Analysis for the paraphrase-invariance experiment.

The question is whether the channel between the two models is grounded in
English or is a private convention between them. A meaning-preserving rewrite
should cost a grounded code almost nothing and cost a conventional code a great
deal.

The comparison cannot be made on degradation alone, because a paraphrase changes
far more words than a one-word corruption does, and more changed words hurt
whatever the reason. So everything here is expressed against two references:

    SURFACE CHANGE -- how much of the original vocabulary moved. Conditions are
        compared at MATCHED surface change, which is why the mechanical controls
        run to doses of 32 and 64: they bracket the paraphrase's surface change
        so the key comparison interpolates instead of extrapolating.

    THE WRONG-CONVERSATION FLOOR -- another conversation's TRUE explanation,
        fluent and wholly about something else. Degradation is reported as a
        FRACTION of the distance to that floor, so 0 means "no damage" and 1
        means "as bad as describing the wrong activation entirely". Without it,
        "reconstruction fell by X degrees" has no scale.

The conditions differ along two axes, and reading them together is what
identifies the answer:

    condition        meaning     surface change
    paraphrase       preserved   large
    topk             mostly kept small to moderate
    corruption/donor changed     small to large
    shuffle          destroyed   zero (same words, new order)

If the code is grounded, paraphrase should sit near zero while corruptions at
the same surface change sit high. If the surface form IS the code, they will sit
together and "it is written in English" is doing no work.

    python exp2_paraphrase/analyse_paraphrase.py
"""
import argparse, json, os
from collections import defaultdict

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, MUT, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e2e1dd"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = ("#2a78d6", "#eb6834", "#1baf7a",
                                       "#eda100", "#e87ba4")
STYLE = {
    "paraphrase": (BLUE, "o", "paraphrase (meaning preserved)"),
    "topk": (AQUA, "^", "verbaliser's own alternatives"),
    "corruption": (ORANGE, "s", "donor corruption (meaning changed)"),
    "mechanical": (YELLOW, "D", "mechanical high-dose / shuffle"),
    "other_conversation": (MAGENTA, "*", "another conversation's true text"),
    # Added for 07: fluent counterfactual rivals, the cell of the
    # surface-change x meaning-change square that was empty until then.
    "rival": ("#7b4bd6", "P", "fluent counterfactual rival (07)"),
}


def per_conv(rows, pred, field):
    """One value per conversation, so the conversation stays the unit."""
    by = defaultdict(list)
    for r in rows:
        if pred(r) and np.isfinite(r.get(field, np.nan)):
            by[r["conv"]].append(float(r[field]))
    return {c: float(np.mean(v)) for c, v in by.items() if v}


def summarise(d, rng=None, boot=10000):
    """Mean AND median AND tail counts, because the mean alone misleads here.

    Damage is concentrated rather than spread: at n=20 the light rewrite had a
    mean of 0.060 and a median of 0.021, the gap driven by a single conversation
    at 0.640. Reporting only the mean would suggest every explanation degrades a
    little when in fact almost all are untouched and a few break badly -- and
    those two situations have opposite implications for monitoring.
    """
    v = np.array(list(d.values()), float)
    if len(v) < 2:
        return dict(mean=float("nan"), se=float("nan"), n=len(v))
    out = dict(mean=float(v.mean()), se=float(v.std(ddof=1) / np.sqrt(len(v))),
               median=float(np.median(v)), n=int(len(v)),
               p25=float(np.percentile(v, 25)), p75=float(np.percentile(v, 75)),
               max=float(v.max()),
               frac_above={str(t): float((v > t).mean())
                           for t in (0.05, 0.10, 0.25, 0.50)})
    if rng is not None:
        bs = np.array([v[rng.integers(0, len(v), len(v))].mean()
                       for _ in range(boot)])
        out["ci95"] = [float(x) for x in np.percentile(bs, [2.5, 97.5])]
    return out


def cluster_permutation_ks(rows_a, rows_b, field="rel", n_perm=20000, seed=0):
    """KS STATISTIC with a null built by permuting labels within conversation.

    The statistic is a valid measure of distributional distance whatever the
    dependence structure; only its usual p-value assumes independent draws,
    which is false when each conversation contributes several rows. Permuting
    inside a conversation destroys the association with the condition while
    leaving that conversation's own level intact.
    """
    xa = np.array([r[field] for r in rows_a], float)
    xb = np.array([r[field] for r in rows_b], float)
    d_obs = stats.ks_2samp(xa, xb).statistic
    lab = np.array([0] * len(xa) + [1] * len(xb))
    conv = np.array([r["conv"] for r in rows_a] + [r["conv"] for r in rows_b])
    vals = np.concatenate([xa, xb])
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = lab.copy()
        for c in np.unique(conv):
            m = conv == c
            perm[m] = rng.permutation(lab[m])
        null[i] = stats.ks_2samp(vals[perm == 0], vals[perm == 1]).statistic
    return dict(D=float(d_obs), p=float((null >= d_obs).mean()),
                null_mean=float(null.mean()),
                p_analytic=float(stats.ks_2samp(xa, xb).pvalue),
                n_a=len(xa), n_b=len(xb))


def cluster_ols(y, X, groups, names):
    """OLS with cluster-robust (CR1) standard errors on the conversation."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    XtX = np.linalg.pinv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(groups):
        m = groups == g
        u = X[m].T @ resid[m]
        meat += np.outer(u, u)
    G, N, k = len(np.unique(groups)), len(y), X.shape[1]
    V = (G / (G - 1)) * ((N - 1) / (N - k)) * XtX @ meat @ XtX
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    return {n: dict(coef=float(b), se=float(s),
                    t=float(b / s) if s > 0 else float("nan"),
                    p=float(2 * stats.t.sf(abs(b / s), G - 1)) if s > 0 else 1.0)
            for n, b, s in zip(names, beta, se)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="results/paraphrase_n60.json")
    p.add_argument("--out", default="results/paraphrase_analysis_n60.json")
    p.add_argument("--fig", default="results/figures/paraphrase_invariance_n60.png")
    p.add_argument("--n-perm", type=int, default=20000)
    a = p.parse_args()

    d = json.load(open(a.data))
    rows = d["rows"]
    rng = np.random.default_rng(0)

    # The floor, per conversation: how far a fluent description of a DIFFERENT
    # activation lands. Everything else is scaled against it.
    base = {r["conv"]: r["base_err"] for r in rows}
    floor = per_conv(rows, lambda r: r["kind"] == "other_conversation", "err")
    # POOLED denominator, deliberately. A per-conversation (floor - base) is
    # occasionally small, and dividing by it produced heavy tails -- a control
    # showing 3.55 +/- 2.36 of the floor, which is an artefact of the divisor
    # rather than a property of the condition. Using the average distance to the
    # floor keeps the same interpretation, 0 = undamaged and 1 = as bad as
    # describing the wrong activation, while making the scale stable.
    denom = float(np.mean([floor[c] - base[c] for c in floor
                           if floor[c] > base[c]]))
    for r in rows:
        r["delta"] = r["err"] - base[r["conv"]]
        r["rel"] = r["delta"] / denom if denom > 0 else float("nan")
    res_denom = denom

    res = {"n_conv": d["n_conv"], "floor_denominator_deg": res_denom,
           "judge": d.get("judge", {}),
           "gate_failures": d.get("gate_failures"),
           "with_quotations": d.get("with_quotations"),
           "retrieval": d["retrieval"]}

    print(f"Scale: the average distance from a true explanation to a fluent "
          f"description of\nthe WRONG activation is {denom:.2f} degrees. "
          f"Everything below is quoted as a\nfraction of that, so 0 means "
          f"undamaged and 1 means as bad as describing the\nwrong activation.\n")
    print("MANIPULATION CHECKS  (if these fail, nothing below means anything)")
    j = d.get("judge", {})
    if j:
        v = j.get("validation", {})
        print(f"  judge on identical pairs      : "
              f"{v.get('identity_called_same', float('nan')):.0%} say 'same'")
        print(f"  judge on heavy corruptions    : "
              f"{v.get('corruption_called_different', float('nan')):.0%} say "
              f"'different'")
        print(f"  judge usable                  : {j.get('usable')}")
        for st in d["strengths"]:
            if st in j:
                print(f"  meaning preserved, {st:<11}: {j[st]:.0%}")
    print(f"  rewrites failing the gates    : {d.get('gate_failures')}")
    print(f"  rewrites containing quotations: {d.get('with_quotations')} "
          f"(restored verbatim from a mask, so never reworded)")

    print("\nSURFACE CHANGE by condition (are the arms where we think?)")
    conds = sorted({(r["kind"], r["strength"]) for r in rows
                    if r["kind"] != "orig"},
                   key=lambda k: (k[0], k[1]))
    res["conditions"] = {}
    print(f"  {'condition':<30}{'surface':>8}{'delta deg':>11}"
          f"{'rel mean':>11}{'rel med':>9}{'>0.25':>7}{'n':>4}")
    for kind, st in conds:
        sel = lambda r, k=kind, s=st: r["kind"] == k and r["strength"] == s
        surf = summarise(per_conv(rows, sel, "surface"))
        dlt = summarise(per_conv(rows, sel, "delta"), rng, boot=2000)
        rel = summarise(per_conv(rows, sel, "rel"), rng, boot=2000)
        res["conditions"][f"{kind}/{st}"] = dict(surface=surf, delta=dlt, rel=rel)
        fa = rel.get("frac_above", {}).get("0.25", float("nan"))
        print(f"  {kind + '/' + st:<30}{surf['mean']:>8.2f}"
              f"{dlt['mean']:>7.2f}±{dlt['se']:<4.2f}"
              f"{rel['mean']:>7.3f}±{rel['se']:<4.3f}"
              f"{rel.get('median', float('nan')):>9.3f}{fa:>7.0%}{dlt['n']:>4}")

    # ---- the key comparison -------------------------------------------------
    print("\nTHE KEY COMPARISON: at MATCHED surface change, does a")
    print("meaning-preserving rewrite hurt less than a meaning-changing one?")
    print("Done as paired contrasts between conditions with similar surface")
    print("change, NOT by regression. An earlier version fitted damage on")
    print("surface change plus a paraphrase indicator and reported the")
    print("paraphrase costing 0.234 less; that was an artefact. The")
    print("relationship is strongly convex -- local slope runs +0.18, +2.10,")
    print("+1.85, +4.98, +7.17 across surface bins -- so a straight line sits")
    print("above the data exactly where the paraphrase points are, and the")
    print("indicator absorbed the curvature rather than any effect of meaning.")

    def paired(a_key, b_key):
        (ak, ast_), (bk, bst) = (a_key.split("/", 1), b_key.split("/", 1))
        A = per_conv(rows, lambda r: r["kind"] == ak and r["strength"] == ast_, "rel")
        B = per_conv(rows, lambda r: r["kind"] == bk and r["strength"] == bst, "rel")
        ks = sorted(set(A) & set(B))
        if len(ks) < 3:
            return None
        dv = np.array([A[c] - B[c] for c in ks])
        w = stats.wilcoxon(dv) if np.any(dv) else None
        return dict(n=len(dv), mean=float(dv.mean()),
                    se=float(dv.std(ddof=1) / np.sqrt(len(dv))),
                    median_a=float(np.median([A[c] for c in ks])),
                    median_b=float(np.median([B[c] for c in ks])),
                    t_p=float(stats.ttest_1samp(dv, 0).pvalue),
                    wilcoxon_p=float(w.pvalue) if w else float("nan"),
                    surface_a=float(np.mean([r["surface"] for r in rows
                                             if r["kind"] == ak and r["strength"] == ast_])),
                    surface_b=float(np.mean([r["surface"] for r in rows
                                             if r["kind"] == bk and r["strength"] == bst])))

    res["matched"] = {}
    print(f"\n  {'contrast':<46}{'surf':>11}{'mean diff':>13}"
          f"{'medians':>16}{'t p':>9}{'W p':>9}")
    for a_key, b_key in (("paraphrase/light", "corruption/dose16"),
                         ("paraphrase/heavy", "corruption/dose16"),
                         ("paraphrase/light", "topk/dose16"),
                         ("paraphrase/heavy", "mechanical/donor32"),
                         ("mechanical/shuffle", "corruption/dose16")):
        r_ = paired(a_key, b_key)
        if not r_:
            continue
        res["matched"][f"{a_key} - {b_key}"] = r_
        print(f"  {a_key + ' - ' + b_key:<46}"
              f"{r_['surface_a']:.2f}/{r_['surface_b']:.2f}"
              f"{r_['mean']:>+10.3f}±{r_['se']:<4.3f}"
              f"{r_['median_a']:>8.3f}/{r_['median_b']:<7.3f}"
              f"{r_['t_p']:>9.3g}{r_['wilcoxon_p']:>9.3g}")
    print("\n  Where the mean and the Wilcoxon disagree, the tail is the reason:")
    print("  a paraphrase typically hurts LESS than a matched corruption, and")
    print("  occasionally hurts far more, which pulls the mean back to parity.")

    para = [r for r in rows if r["kind"] == "paraphrase"
            and np.isfinite(r["rel"])]
    meaning_changed = [r for r in rows
                       if r["kind"] in ("corruption", "mechanical")
                       and r["strength"] != "shuffle"
                       and np.isfinite(r["rel"])]
    if para and meaning_changed:
        ks = cluster_permutation_ks(para, meaning_changed, n_perm=a.n_perm)
        res["ks"] = ks
        print(f"\n  Distributions overall (all surface changes pooled, so this "
              f"describes\n  the two families rather than a matched contrast): "
              f"KS D={ks['D']:.3f},\n  cluster-permutation p={ks['p']:.4g} "
              f"(null mean {ks['null_mean']:.3f})")

    # ---- equivalence: is the paraphrase cost negligible? --------------------
    pc = per_conv(rows, lambda r: r["kind"] == "paraphrase", "rel")
    if len(pc) > 2:
        v = np.array(list(pc.values()))
        se = v.std(ddof=1) / np.sqrt(len(v))
        tightest = float(abs(v.mean()) + stats.t.ppf(0.95, len(v) - 1) * se)
        res["equivalence"] = dict(mean=float(v.mean()), se=float(se),
                                  tightest_bound=tightest, n=len(v))
        print(f"\nEQUIVALENCE: paraphrase costs {v.mean():.3f} ± {se:.3f} of the "
              f"wrong-conversation distance;\n  the smallest bound rejectable "
              f"at 5% is {tightest:.3f}. Read as: a meaning-preserving rewrite\n"
              f"  costs at most {tightest:.0%} of what describing the wrong "
              f"activation costs.")

    # The countable statements. The rel scale is a ratio of DISTANCES, not a
    # proportion of anything -- it cannot be read as "X% of explanations are
    # grounded". These two can.
    print("\nCOUNTABLE FORM (rel is a distance ratio, not a proportion, so the "
          "claims\nthat can be stated as counts are these):")
    for st in d["strengths"]:
        r = res["conditions"].get(f"paraphrase/{st}", {}).get("rel", {})
        fa = r.get("frac_above", {})
        if fa:
            print(f"  {st:<6} typical explanation loses {r['median']:.3f} of "
                  f"the wrong-activation distance (median);")
            print(f"         {fa['0.25']:.0%} of conversations lose more than "
                  f"0.25, {fa['0.5']:.0%} more than 0.50.")

    print("\nFUNCTIONAL TEST: can a rewrite still identify its own activation?")
    for k, v in d["retrieval"].items():
        print(f"  {k:<8} top-1 {v['top1']:.3f}   MRR {v['mrr']:.3f}   "
              f"(chance {1 / v['n']:.3f})")

    json.dump(res, open(a.out, "w"), indent=1)
    make_figure(rows, res, d, a.fig)
    print(f"\nwrote {a.out} and {a.fig}")


def make_figure(rows, res, d, path):
    """Four panels chosen for the claims actually being made.

    NOT a violin plot, deliberately. With around 40 conversations per condition
    a kernel density estimate invents shape, and it is least trustworthy exactly
    where the interesting cases are -- the sparse tail. The claims here are
    about medians and about exceedance ("8% lose more than 0.25"), so the figure
    shows every conversation as a point, and gives the cumulative distribution,
    from which an exceedance fraction is a horizontal read.
    """
    fig = plt.figure(figsize=(16, 8.6), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.24,
                          left=0.075, right=0.985, top=0.86, bottom=0.08)
    ax_sc = fig.add_subplot(gs[0, 0])
    ax_st = fig.add_subplot(gs[:, 1])
    ax_cd = fig.add_subplot(gs[1, 0])
    for ax in (ax_sc, ax_st, ax_cd):
        ax.set_facecolor(SURFACE)
        ax.grid(alpha=0.55, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.tick_params(colors=MUT, labelsize=9)

    def per_conversation(kind, st):
        by = defaultdict(list)
        for r in rows:
            if (r["kind"] == kind and r["strength"] == st
                    and np.isfinite(r.get("rel", np.nan))):
                by[r["conv"]].append(r["rel"])
        return np.array([np.mean(x) for x in by.values()])

    for kind, (c, m, lab) in STYLE.items():
        pts = [(r["surface"], r["rel"]) for r in rows
               if r["kind"] == kind and np.isfinite(r.get("rel", np.nan))]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax_sc.scatter(xs, ys, s=20, color=c, marker=m, alpha=0.45,
                      edgecolor=SURFACE, lw=0.5, label=lab)
    ax_sc.axhline(1.0, color=MAGENTA, ls="--", lw=1.3)
    ax_sc.annotate("as bad as describing the wrong activation", (0.01, 1.0),
                   textcoords="offset points", xytext=(2, 4), fontsize=7.5,
                   color=INK)
    ax_sc.axhline(0.0, color=INK, ls=":", lw=1.1)
    ax_sc.set_xlabel("surface change (fraction of the original's words gone)",
                     color=MUT, fontsize=9.5)
    ax_sc.set_ylabel("damage / wrong-activation distance", color=MUT,
                     fontsize=9.5)
    ax_sc.set_title("At matched surface change, meaning-preserving rewrites\n"
                    "cost less than meaning-changing edits",
                    color=INK, fontsize=10.5, loc="left", pad=8)
    ax_sc.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="upper left")

    order = [k for k in res["conditions"]
             if np.isfinite(res["conditions"][k]["rel"].get("median", np.nan))]
    order.sort(key=lambda k: res["conditions"][k]["rel"]["median"])
    jitter = np.random.default_rng(0)
    for i, key in enumerate(order):
        kind, st = key.split("/", 1)
        v = per_conversation(kind, st)
        if not len(v):
            continue
        c = STYLE.get(kind, (MUT,))[0]
        ax_st.scatter(v, i + jitter.uniform(-0.16, 0.16, len(v)), s=16,
                      color=c, alpha=0.5, edgecolor="none")
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax_st.plot([q1, q3], [i, i], "-", color=c, lw=4, alpha=0.85,
                   solid_capstyle="butt")
        ax_st.plot(med, i, "|", color=INK, ms=16, mew=2.2)
    ax_st.axvline(1.0, color=MAGENTA, ls="--", lw=1.3)
    ax_st.axvline(0.25, color=MUT, ls=":", lw=1.1)
    ax_st.axvline(0.0, color=INK, lw=1.1)
    ax_st.set_yticks(range(len(order)), [k.replace("/", "  ") for k in order],
                     color=INK, fontsize=8.5)
    ax_st.set_xlabel("damage / wrong-activation distance\n"
                     "dot = one conversation, bar = interquartile range, "
                     "tick = median", color=MUT, fontsize=9)
    ax_st.set_title("Damage is CONCENTRATED, not spread: most explanations\n"
                    "barely move and a few break",
                    color=INK, fontsize=10.5, loc="left", pad=8)

    for key, c, lab in (("paraphrase/light", BLUE, "paraphrase (light)"),
                        ("paraphrase/heavy", "#6da7ec", "paraphrase (heavy)"),
                        ("topk/dose16", AQUA, "top-k, 16 words"),
                        ("corruption/dose16", ORANGE, "donor, 16 words"),
                        ("mechanical/shuffle", YELLOW, "word-order shuffle")):
        kind, st = key.split("/", 1)
        v = np.sort(per_conversation(kind, st))
        if not len(v):
            continue
        ax_cd.step(v, np.arange(1, len(v) + 1) / len(v), where="post",
                   color=c, lw=2, label=lab)
    ax_cd.axvline(0.25, color=MUT, ls=":", lw=1.1)
    ax_cd.set_xlim(-0.05, 0.75)
    ax_cd.set_ylim(0, 1.02)
    ax_cd.set_xlabel("damage / wrong-activation distance", color=MUT,
                     fontsize=9.5)
    ax_cd.set_ylabel("fraction of conversations at or below", color=MUT,
                     fontsize=9.5)
    ax_cd.set_title("Exceedance read off directly: at 0.25, how many\n"
                    "conversations are still below the line?",
                    color=INK, fontsize=10.5, loc="left", pad=8)
    ax_cd.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")

    fig.suptitle(f"Paraphrase invariance (n = {res['n_conv']} conversations): "
                 f"how much of the channel rides on wording, not meaning?",
                 color=INK, fontsize=13, x=0.006, ha="left")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=170, facecolor=SURFACE)


if __name__ == "__main__":
    main()
