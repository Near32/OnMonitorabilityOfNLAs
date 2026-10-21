"""Stage 1: is the information THERE, or does the AR merely not use it?

Section 04 shows the AR cannot pick the true explanation out of its own dose-1
confabulations — it sits at chance. That establishes what the AR *does*, not
what is *available* to it. This script decides between the two:

  capacity    the distinction is present in AR(z) and the released readout
              simply does not exploit it -> fixable by fine-tuning
  equilibrium the distinction is absent from AR(z) -> no fine-tune conjures it

Method. For each conversation we hold the activation h, the AR's reconstruction
of the true explanation, and of its K confabulated variants. The AR's own score
is cos(h, AR(z)) — its MSE is exactly 2(1-cos). We train a low-rank bilinear
score s(h, r) = <P h, Q r> with InfoNCE over {true, K distractors}, split by
CONVERSATION so no explanation appears in both train and test.

Primary metric is AUC, threshold-free and directly comparable to the base AR's
own scoring. Reported with top-1 retrieval (chance 1/(K+1)) and the base AR's
AUC, which at dose 1 should be ~0.5 because that arm is at chance.

    PYTHONPATH=nla python exp1_identification/probe_ar.py
"""

import argparse, json, os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traces", default="artefacts/traces/hardneg_donor_pos_traces.jsonl")
    p.add_argument("--train-doses", default="1,2,4",
                   help="doses the probe is FITTED on")
    p.add_argument("--eval-doses", default="1,2,4,8,16",
                   help="doses it is TESTED on; 8 and 16 are unseen, so the "
                        "gap between them and 1-2-4 measures generalisation "
                        "rather than memorisation of one corruption level")
    p.add_argument("--train-frac", type=float, default=0.5,
                   help="fraction of CONVERSATIONS used for fitting")
    p.add_argument("--cache", default="artefacts/probe_cache.npz")
    p.add_argument("--ar-dir", default="models/ar")
    p.add_argument("--extract-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--gold-cache", default="artefacts/cache/golds.pt",
                   help="written by hardneg.py; the extraction model is loaded "
                        "only for conversations it lacks")
    p.add_argument("--extract-index", type=int, default=21)
    p.add_argument("--min-position", type=int, default=50)
    p.add_argument("--position-offset", type=int, default=2)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--rank", type=int, default=256)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/probe_results.json")
    return p.parse_args()


def build_cache(args):
    """h, AR(true), AR(distractors) per conversation. Slow part; cached."""
    from dr import load_conversations
    from hardneg import gold_cache
    from nla_inference import NLACritic

    doses = [int(x) for x in args.eval_doses.split(",")]
    # {conv: {"orig": text, dose: [texts]}}
    by_sample = defaultdict(lambda: defaultdict(list))
    with open(args.traces, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("dose") not in doses:
                continue
            by_sample[r["conv_index"]]["orig"] = r["original"]
            by_sample[r["conv_index"]][r["dose"]].append(r["modified"])
    # keep only conversations that have every dose, so the per-dose comparison
    # is over the SAME conversations rather than a shifting subset
    convs_needed = sorted(c for c, v in by_sample.items()
                          if all(len(v.get(d, [])) for d in doses))
    k_min = min(min(len(by_sample[c][d]) for d in doses) for c in convs_needed)
    print(f"{len(convs_needed)} conversations with all doses {doses}, "
          f"K={k_min} distractors each")

    conversations = load_conversations(args.shard, max(convs_needed) + 1)
    golds = gold_cache(args, conversations)       # before the AR is loaded
    critic = NLACritic(args.ar_dir, device=args.device)

    H, T, D, keep = [], [], [], []
    for ci in convs_needed:
        entry = by_sample[ci]
        gold = golds[ci]
        if gold is None or not entry.get("orig"):
            continue
        H.append(gold.numpy())
        T.append(critic.reconstruct(entry["orig"]).float().numpy())
        D.append(np.stack([np.stack([critic.reconstruct(m).float().numpy()
                                     for m in entry[d][:k_min]])
                           for d in doses]))            # (n_doses, K, d)
        keep.append(ci)
        if len(keep) % 10 == 0:
            print(f"  cached {len(keep)}/{len(convs_needed)}", flush=True)
    np.savez_compressed(args.cache, H=np.stack(H), T=np.stack(T),
                        D=np.stack(D), conv=np.array(keep),
                        doses=np.array(doses))
    print(f"wrote {args.cache}")


def unit(x, axis=-1):
    return x / np.clip(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12, None)


def auc(pos, neg):
    """Mean over conversations of P(true > a random distractor of ITS OWN set).

    The retrieval task is within-conversation, so pooling every true score
    against every distractor score across conversations lets between-
    conversation score offsets swamp the signal. That is not a subtlety: on
    this data the pooled statistic read 0.536 at dose 1 while the within-set
    statistic read 0.903, which is the difference between concluding
    EQUILIBRIUM and CAPACITY.
    """
    return float(np.mean([(p > row).mean() for p, row in zip(pos, neg)]))


def evaluate(sh, st, sd, label):
    """sh unused; st (n,), sd (n,K). Returns AUC and top-1."""
    a = auc(st, sd)
    top1 = float((st[:, None] > sd).all(axis=1).mean())
    return {"auc": float(a), "top1": top1, "chance_top1": 1.0 / (sd.shape[1] + 1),
            "n": int(len(st)), "label": label}


def main():
    args = parse_args()
    if not os.path.exists(args.cache):
        build_cache(args)
    z = np.load(args.cache)
    H, T, D = unit(z["H"]), unit(z["T"]), unit(z["D"])
    if "doses" not in z:
        raise SystemExit(f"{args.cache} predates the dose-aware format; delete "
                         f"it and rerun so the cache is rebuilt with all doses.")
    doses = [int(x) for x in z["doses"]]
    n, nd, K, d = D.shape
    train_doses = [int(x) for x in args.train_doses.split(",")]
    di = {int(dv): i for i, dv in enumerate(doses)}
    print(f"cache: {n} conversations, doses {[int(x) for x in doses]}, K={K}, d={d}")

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n)
    ntr = int(args.train_frac * n)
    tr, te = idx[:ntr], idx[ntr:]
    print(f"split by conversation: {len(tr)} train / {len(te)} test")
    print(f"fitted on doses {train_doses}; evaluated on {[int(x) for x in doses]} "
          f"({[int(x) for x in doses if int(x) not in train_doses]} unseen)")

    # --- baseline: the AR's own score, cos(h, AR(z)) ------------------------
    base_t = (H * T).sum(-1)
    base_d = (H[:, None, None, :] * D).sum(-1)          # (n, nd, K)

    # --- probe: low-rank bilinear metric, InfoNCE --------------------------
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Ht = torch.tensor(H, dtype=torch.float32, device=dev)
    Tt = torch.tensor(T, dtype=torch.float32, device=dev)
    Dt = torch.tensor(D, dtype=torch.float32, device=dev)

    torch.manual_seed(args.seed)
    P = nn.Linear(d, args.rank, bias=False).to(dev)
    Q = nn.Linear(d, args.rank, bias=False).to(dev)
    opt = torch.optim.Adam(list(P.parameters()) + list(Q.parameters()), lr=args.lr)
    tr_t = torch.tensor(tr, device=dev)
    tdi = torch.tensor([di[t] for t in train_doses], device=dev)

    for ep in range(args.epochs):
        opt.zero_grad()
        h = P(Ht[tr_t])                                        # (b, r)
        pos = (h * Q(Tt[tr_t])).sum(-1, keepdim=True)          # (b, 1)
        # pool the training doses: one row per (conversation, train dose)
        negs = Q(Dt[tr_t][:, tdi])                             # (b, td, K, r)
        neg = torch.einsum("br,btkr->btk", h, negs)
        logits = torch.cat([pos.unsqueeze(1).expand(-1, len(tdi), -1), neg], dim=2)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, K + 1),
            torch.zeros(len(tr_t) * len(tdi), dtype=torch.long, device=dev))
        loss.backward(); opt.step()
        if ep % 100 == 0:
            print(f"  epoch {ep:4d} loss {loss.item():.4f}", flush=True)

    with torch.no_grad():
        h = P(Ht); qt = Q(Tt); qd = Q(Dt)
        pt = (h * qt).sum(-1).cpu().numpy()
        pd_ = torch.einsum("br,bdkr->bdk", h, qd).cpu().numpy()

    per_dose = {}
    print(f"\n{'dose':>5}{'base AUC':>10}{'probe AUC':>11}{'base top1':>11}"
          f"{'probe top1':>12}{'chance':>8}{'seen?':>7}")
    for dv in doses:
        i = di[int(dv)]
        b = evaluate(None, base_t[te], base_d[te, i], f"base d{int(dv)}")
        q = evaluate(None, pt[te], pd_[te, i], f"probe d{int(dv)}")
        seen = "train" if int(dv) in train_doses else "UNSEEN"
        per_dose[int(dv)] = {"base": b, "probe": q, "seen": seen}
        print(f"{int(dv):>5}{b['auc']:>10.3f}{q['auc']:>11.3f}{b['top1']:>11.3f}"
              f"{q['top1']:>12.3f}{q['chance_top1']:>8.3f}{seen:>7}")

    # headline: dose 1, the dose at which confabulation actually occurs
    # Gate on the SMALLEST dose: one substituted word is where confabulation
    # actually lives and is the hardest case, so it is the honest gate.
    probe_auc = per_dose[min(per_dose)]["probe"]["auc"]
    train_check = evaluate(None, pt[tr], pd_[tr, di[train_doses[0]]],
                           "probe (train split, overfit check)")
    verdict = ("CAPACITY: the distinction is present in AR(z); the released "
               "readout does not use it -> stage 2 is worth running"
               if probe_auc > 0.60 else
               "EQUILIBRIUM: the distinction is not recoverable from AR(z); "
               "fine-tuning will not conjure it -> skip stage 2")
    print(f"\ntrain-split AUC at dose {train_doses[0]}: "
          f"{train_check['auc']:.3f} (overfit check)")
    print(f"\n{verdict}")
    out = {"args": vars(args), "per_dose": per_dose, "verdict": verdict,
           "probe": {"auc": probe_auc}, "train_check": train_check}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
