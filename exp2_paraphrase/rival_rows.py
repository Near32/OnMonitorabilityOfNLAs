"""Re-score the rivals onto the PARAPHRASE experiment's axes, and merge.

Folder 07 reports discrimination (AUC within a lineup). Folder 06 reports
damage as a fraction of the distance to a description of a DIFFERENT activation
(0 = free, 1 = as bad as describing something else). Those are different
questions, but they are measured on the same reconstructions, and the 2x2 that
motivated 07 only closes if the conditions sit on ONE pair of axes:

                        meaning preserved        meaning changed
    small surface       topk (06)                donor dose 1-4 (04/06)
    large surface       paraphrase (06)          donor 32/64 -- but incoherent
                                                 RIVALS (07)  <- the empty cell

So this emits rival rows in exactly the schema `analyse_paraphrase.py` reads
-- conv, kind, strength, err, base_err, surface -- and merges them into the
paraphrase rows, so one figure carries every condition. No generation: it
reloads the cached rival texts and re-embeds, which is a few minutes of GPU.

    PYTHONPATH=nla python exp2_paraphrase/rival_rows.py
    python exp2_paraphrase/analyse_paraphrase.py \
        --data results/paraphrase_plus_rivals.json \
        --out results/paraphrase_plus_rivals_analysis.json \
        --fig results/figures/rivals_on_paraphrase_axes.png
"""
import argparse, json, os, sys

import numpy as np
import torch
from tqdm.auto import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "..", "exp2_paraphrase"))
from paraphrase import surface_change, tokens
from nla_inference import NLACritic


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--texts", default="artefacts/rival_texts.jsonl")
    p.add_argument("--gold-cache", default="artefacts/gold_activations.npz")
    p.add_argument("--paraphrase", default="results/paraphrase_n60.json",
                   help="rows to merge into; omit to emit rivals alone")
    p.add_argument("--ar-dir", default="models/ar")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="results/paraphrase_plus_rivals.json")
    a = p.parse_args()

    texts = [json.loads(l) for l in open(a.texts, encoding="utf-8")]
    cache = np.load(a.gold_cache)
    golds = {int(c): torch.from_numpy(h) for c, h in zip(cache["conv"], cache["H"])}
    texts = [r for r in texts if r["conv"] in golds]
    print(f"{len(texts)} conversations with a cached activation")

    critic = NLACritic(a.ar_dir, device=a.device)
    unit = lambda v: v / v.norm().clamp_min(1e-12)
    ang = lambda x, y: float(np.degrees(np.arccos(np.clip(
        float((unit(x) * unit(y)).sum()), -1, 1))))

    rows, base_vec = [], {}
    with torch.no_grad():
        for r in tqdm(texts, desc="embedding", unit="conv"):
            c, g = r["conv"], golds[r["conv"]].float()
            v0 = critic.reconstruct(r["orig"]).float()
            base_vec[c] = v0
            base_err = ang(v0, g)
            rows.append(dict(conv=c, kind="orig", strength="-", n_changed=0,
                             surface=0.0, err=base_err, base_err=base_err,
                             disp=0.0))
            for n, variants in r["rivals"].items():
                for m in variants:
                    if not m:
                        continue
                    v = critic.reconstruct(m).float()
                    rows.append(dict(
                        conv=c, kind="rival", strength=f"n{n}",
                        n_changed=len(set(tokens(r["orig"])) - set(tokens(m))),
                        surface=surface_change(r["orig"], m),
                        err=ang(v, g), base_err=base_err, disp=ang(v, v0)))
        # the floor, built the same way 06 builds it: the NEXT conversation's
        # true description, fluent and wholly about something else.
        for i, r in enumerate(texts):
            c = r["conv"]
            other = texts[(i + 1) % len(texts)]["conv"]
            v = base_vec[other]
            rows.append(dict(
                conv=c, kind="other_conversation", strength="true-but-wrong",
                n_changed=0, surface=float("nan"),
                err=ang(v, golds[c].float()),
                base_err=ang(base_vec[c], golds[c].float()),
                disp=ang(v, base_vec[c])))

    out = dict(rows=rows, n_conv=len(texts), retrieval={}, judge={},
               strengths=sorted({r["strength"] for r in rows
                                 if r["kind"] == "rival"}))
    if a.paraphrase and os.path.exists(a.paraphrase):
        d = json.load(open(a.paraphrase))
        # Keep only the conversations 07 also covers, so every condition in the
        # merged figure is measured on the SAME conversations and the strip
        # plot compares like with like.
        keep = {r["conv"] for r in rows}
        pr = [r for r in d["rows"] if r["conv"] in keep]
        seen = {(r["conv"], r["kind"], r["strength"]) for r in pr}
        out["rows"] = pr + [r for r in rows
                            if (r["conv"], r["kind"], r["strength"]) not in seen]
        out["retrieval"] = d.get("retrieval", {})
        out["judge"] = d.get("judge", {})
        out["n_conv"] = len(keep)
        print(f"merged: {len(pr)} paraphrase rows on {len(keep)} shared "
              f"conversations + {len(out['rows']) - len(pr)} rival rows")
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {len(out['rows'])} rows, kinds "
          f"{sorted({r['kind'] for r in out['rows']})}")


if __name__ == "__main__":
    main()
