"""Variant A: DR under top-k replacement of high-entropy tokens, swept by dose.

A "fork token" is a position where the AV's own next-token distribution was
uncertain, so it could plausibly have written something else. We corrupt the
top-N such positions simultaneously — each token replaced by a DIFFERENT token
drawn from that position's top-k candidates — and sweep N over 1, 2, 4, 8, 16.

Why a dose sweep. A single-token swap moves the AR's reconstruction by ~0.1%
(DR ~= 1.000), while destroying all word order moves it to the predict-the-mean
baseline (DR ~= 5.2, see dr.py). Those are the two ends of a curve, and the
question Hypothesis C actually poses is where in between the AR starts to
notice — i.e. how much semantic content must change before reconstruction
degrades. A single dose cannot answer that; the curve can.

Threshold is a PERCENTILE of the observed entropy distribution, not a fixed
nats value. §5.4's "entropy > 0.8 nats" is an absolute cut whose meaning drifts
with explanation length and sharpness; a percentile keeps the fork/non-fork
split comparable across samples, and §9.1's limitation about arbitrariness
applies to the percentile instead.

Control: the same doses applied to the LOWEST-entropy positions. Hypothesis C
is tested by the fork-versus-non-fork contrast, not the absolute DR.

    PYTHONPATH=nla python nla/fork.py --n-samples 20
"""

import argparse
import csv
import json

import numpy as np
import torch
from tqdm.auto import tqdm

from dr import (EXPLANATION_RE, aggregate, gold_activation, load_conversations)
from nla_inference import NLAClient, NLACritic
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--top-k", type=int, default=5,
                   help="topk mode: candidate pool size, excluding the original token")
    p.add_argument("--candidate-mode", choices=("topk", "topp"), default="topk",
                   help="topk (default): the k most likely tokens, sampled "
                        "UNIFORMLY — so a zero-entropy position still yields k "
                        "alternatives, all of which the AV considered impossible. "
                        "topp: the nucleus whose cumulative probability reaches "
                        "--top-p, sampled PROPORTIONALLY to probability, as in "
                        "real nucleus sampling. At a confident position the "
                        "nucleus may hold only the chosen token, leaving nothing "
                        "to swap in — which is the point: it tests whether the "
                        "low-entropy effect is an artefact of forcing implausible "
                        "insertions.")
    p.add_argument("--top-p", type=float, default=0.9,
                   help="nucleus mass for --candidate-mode topp")
    p.add_argument("--entropy-percentile", type=float, default=80.0,
                   help="threshold mode: positions at or above this percentile are forks")
    p.add_argument("--position-filter", choices=("none", "content", "entity"),
                   default="none",
                   help="none (default): any body token is eligible. content: "
                        "exclude closed-class words, punctuation and mid-word "
                        "fragments, so corruptions land on meaning-bearing words "
                        "rather than producing 'the -> a'. entity: only "
                        "capitalised words and numbers — the name/number/quote "
                        "population the confabulation work is actually about.")
    p.add_argument("--perturbation", choices=("replace", "swap"), default="replace",
                   help="replace: swap each chosen token for a different top-k "
                        "candidate (changes identity, keeps position). swap: "
                        "permute the chosen tokens among themselves in random "
                        "pairs (keeps the multiset EXACTLY, changes order). At "
                        "matched dose the two differ only in that respect, which "
                        "is what separates 'the AR reads structure' from 'we "
                        "perturbed more tokens'.")
    p.add_argument("--swap-partner", choices=("within", "distance"), default="within",
                   help="within: pair the chosen positions with each other "
                        "(default). distance: swap each chosen token with a body "
                        "token exactly --swap-distance away, so mean displacement "
                        "is identical across groups BY CONSTRUCTION — this is what "
                        "disentangles entropy from displacement, since fork tokens "
                        "cluster (mean distance 25-43) while low-entropy function "
                        "words are spread out (47-50).")
    p.add_argument("--swap-distance", type=int, default=40,
                   help="displacement for --swap-partner distance")
    p.add_argument("--pool-mode", choices=("threshold", "rank"), default="threshold",
                   help="threshold: fork pool = body positions above the entropy "
                        "percentile (default, backward compatible). rank: pools are "
                        "the highest- and lowest-entropy body positions in order, so "
                        "the dose itself sets how deep into the ordering we reach — "
                        "needed for doses larger than the percentile pool.")
    p.add_argument("--doses", default="1,2,4,8,16",
                   help="comma-separated token counts to corrupt simultaneously")
    p.add_argument("--n-draws", type=int, default=5,
                   help="independent candidate draws per (sample, dose)")
    p.add_argument("--extract-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--av-dir", default="models/av")
    p.add_argument("--ar-dir", default="models/ar")
    p.add_argument("--extract-index", type=int, default=21)
    p.add_argument("--min-position", type=int, default=50)
    p.add_argument("--position-offset", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="dr_fork.csv")
    p.add_argument("--trace", default="dr_fork_traces.jsonl",
                   help="per-corruption log of original and modified text")
    p.add_argument("--weave-project", default=None,
                   help="also log traces to weave (pip install --target vendor weave)")
    return p.parse_args()


def generate_with_scores(av_model, client, activation, max_new_tokens):
    """Greedy AV generation, keeping the per-step next-token distributions.

    Running the AV locally is what makes fork detection cheap: output_scores
    returns the logits at every generated step, so nothing in any serving layer
    needs changing. With inputs_embeds and no input_ids, generate() returns only
    the generated tokens, so sequences and scores are index-aligned.

    Returns (ids [T], entropies [T] in nats, scores tuple).
    """
    embeds_np, _ = client._build_embeds(activation, None)
    embeds = torch.from_numpy(embeds_np).unsqueeze(0).to(av_model.device, av_model.dtype)
    attn = torch.ones(embeds.shape[:2], dtype=torch.long, device=av_model.device)
    with torch.no_grad():
        out = av_model.generate(
            inputs_embeds=embeds, attention_mask=attn,
            max_new_tokens=max_new_tokens, do_sample=False,
            return_dict_in_generate=True, output_scores=True,
            pad_token_id=client.tokenizer.pad_token_id or client.tokenizer.eos_token_id,
        )
    entropies = []
    for step in out.scores:
        logp = torch.log_softmax(step[0].float(), dim=-1)
        entropies.append(float(-(logp.exp() * logp).sum()))
    return out.sequences[0], np.array(entropies), out.scores


def explanation_from_ids(tokenizer, ids):
    """Decode ids and pull the <explanation> body, or None if it is absent."""
    text = tokenizer.decode(ids, skip_special_tokens=False)
    match = EXPLANATION_RE.search(text)
    return match.group(1).strip() if match else None


# Closed-class words: the discourse scaffolding an explanation is mostly built
# from. Corrupting these produces "the -> a", which is not the phenomenon the
# confabulation literature is about. Deliberately hand-rolled rather than
# pulling in NLTK for one frozenset.
STOPWORDS = frozenset("""
a an the this that these those it its it's they them their there here
and or but nor so yet for as if then than because while although though
of in on at by to from with without into onto over under about across
is are was were be been being am do does did doing have has had having
will would shall should can could may might must
i you he she we us our your my me him her his hers
not no nor very just also too more most much many few some any all
what which who whom whose when where why how
one two three first second next last other another same such own
""".split())


def token_class(text):
    """Rough class of a decoded token: 'entity', 'content', or 'function'.

    Word-initial tokens in this BPE carry a leading space; continuations do
    not, and corrupting a mid-word fragment produces garbage rather than a
    substitution, so fragments are never eligible.
    """
    if not text.startswith((" ", "\n")) and text.strip():
        return "fragment"
    w = text.strip()
    if not w:
        return "function"
    if w.isdigit():
        return "entity"
    if not w.isalpha():
        return "function"
    if w[0].isupper():
        return "entity"          # capitalised mid-sentence: name-like
    if w.lower() in STOPWORDS or len(w) < 3:
        return "function"
    return "content"


def body_positions(tokenizer, ids, position_filter="none"):
    """Token indices lying strictly inside the <explanation> body.

    The lowest-entropy positions in a generation are the tag tokens themselves
    ('<', 'ex', 'planation', '>', '</', '<|im_end|>'), all at entropy 0.0000.
    Corrupting one destroys the delimiter, extraction returns None, and the
    draw is discarded — which silently emptied the entire low-entropy control
    group on the first run. Restricting BOTH pools to the body means fork and
    control differ in uncertainty and in nothing else.

    Cumulative decoding is O(T^2) but T is ~145 here, so it costs nothing and
    avoids guessing which sub-tokens spell the tags.
    """
    full = tokenizer.decode(ids, skip_special_tokens=False)
    open_at = full.find("<explanation>")
    close_at = full.find("</explanation>")
    if open_at < 0 or close_at < 0:
        return []
    open_at += len("<explanation>")
    out, prev = [], 0
    for i in range(len(ids)):
        end = len(tokenizer.decode(ids[: i + 1], skip_special_tokens=False))
        if prev >= open_at and end <= close_at:
            if position_filter == "none":
                out.append(i)
            else:
                cls = token_class(tokenizer.decode([int(ids[i])]))
                if position_filter == "content" and cls in ("content", "entity"):
                    out.append(i)
                elif position_filter == "entity" and cls == "entity":
                    out.append(i)
        prev = end
    return out


def alternatives(scores, ids, index, top_k, mode="topk", top_p=0.9):
    """Candidate ids at `index`, with the originally chosen token removed.

    topk: requesting top_k+1 and filtering guarantees k real alternatives even
        when the chosen token is itself in the top-k — otherwise a position
        could silently offer fewer candidates, or none. Weights are uniform.
    topp: the nucleus reaching cumulative mass top_p, weighted by (renormalised)
        probability. Unlike topk this pool ADAPTS to the model's certainty, so a
        near-deterministic position legitimately yields no alternatives and is
        left untouched — the realised count is logged rather than assumed.

    Returns (ids, weights) where weights is None for uniform sampling.
    """
    chosen = int(ids[index])
    if mode == "topp":
        probs = torch.softmax(scores[index][0].float(), dim=-1)
        srt, idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(srt, dim=-1)
        # Inclusive cut: keep the token that carries the mass past top_p.
        keep = int(torch.searchsorted(cum, torch.tensor(top_p, device=cum.device))) + 1
        ids_keep = idx[:keep].tolist()
        w_keep = srt[:keep].tolist()
        pairs = [(c, w) for c, w in zip(ids_keep, w_keep) if c != chosen]
        if not pairs:
            return [], None
        cand = [c for c, _ in pairs]
        wt = np.array([w for _, w in pairs], dtype=float)
        return cand, wt / wt.sum()
    top = torch.topk(scores[index][0], top_k + 1).indices.tolist()
    return [c for c in top if c != chosen][:top_k], None


def corrupt(tokenizer, ids, scores, positions, top_k, rng, mode="replace",
            partner="within", distance=40, body=None,
            candidate_mode="topk", top_p=0.9):
    """Perturb every chosen position at once.

    replace: each token becomes a DIFFERENT token from that position's top-k —
        identity changes, position and token count are preserved.
    swap: the chosen positions are paired at random and their tokens exchanged —
        the multiset is preserved EXACTLY, only order changes. Pairing is random
        across the whole selection rather than between neighbours, because a
        swap between two adjacent similar tokens is a far weaker perturbation
        than a long-distance one; mean_distance is returned so that assumption
        can be checked post-hoc rather than trusted.

    Returns (explanation_text or None, swap descriptions, mean swap distance).
    """
    out = ids.clone()
    swaps = []
    if mode == "swap" and partner == "distance":
        # Each chosen token trades places with a body token exactly `distance`
        # away (sign chosen at random, falling back to the other direction at
        # the edges). Displacement is then identical across groups, so any
        # remaining fork/control difference is attributable to entropy alone.
        # The partner is an arbitrary body token, which perturbs positions
        # outside the selection — but identically for both groups, so the
        # contrast stays clean even though the absolute DR is not comparable
        # to the "within" arm.
        used, dists = set(), []
        for j in positions:
            for delta in ([distance, -distance] if rng.random() < 0.5
                          else [-distance, distance]):
                p = j + delta
                if p in body and p not in used and j not in used \
                        and int(ids[p]) != int(ids[j]):
                    out[j], out[p] = ids[p], ids[j]
                    used.update((j, p))
                    dists.append(abs(delta))
                    swaps.append({"pos": int(j), "with": int(p),
                                  "was": tokenizer.decode([int(ids[j])]),
                                  "now": tokenizer.decode([int(ids[p])])})
                    break
        return (explanation_from_ids(tokenizer, out), swaps,
                float(np.mean(dists)) if dists else 0.0)

    if mode == "swap":
        order = list(positions)
        rng.shuffle(order)
        dists = []
        # floor(len/2) pairs; an odd position is left untouched rather than
        # forced into a 3-cycle, so "dose" stays interpretable as tokens moved.
        for a, b in zip(order[::2], order[1::2]):
            if int(ids[a]) == int(ids[b]):
                continue  # identical tokens: swapping them changes nothing
            out[a], out[b] = ids[b], ids[a]
            dists.append(abs(int(a) - int(b)))
            swaps.append({"pos": int(a), "with": int(b),
                          "was": tokenizer.decode([int(ids[a])]),
                          "now": tokenizer.decode([int(ids[b])])})
        mean_dist = float(np.mean(dists)) if dists else 0.0
        return explanation_from_ids(tokenizer, out), swaps, mean_dist

    for j in positions:
        alts, weights = alternatives(scores, ids, j, top_k,
                                     mode=candidate_mode, top_p=top_p)
        if not alts:
            continue  # nucleus held only the chosen token: nothing plausible
        cand = int(rng.choice(alts, p=weights))
        swaps.append({"pos": int(j),
                      "was": tokenizer.decode([int(ids[j])]),
                      "now": tokenizer.decode([cand])})
        out[j] = cand
    return explanation_from_ids(tokenizer, out), swaps, 0.0


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    doses = [int(d) for d in args.doses.split(",")]

    weave_run = None
    if args.weave_project:
        import weave  # lazy: keeps the default path dependency-free
        weave_run = weave.init(args.weave_project)

    tokenizer = AutoTokenizer.from_pretrained(args.extract_model)
    extract_model = AutoModelForCausalLM.from_pretrained(
        args.extract_model, dtype=torch.bfloat16, device_map="auto")
    av_model = AutoModelForCausalLM.from_pretrained(
        args.av_dir, dtype=torch.bfloat16, device_map="auto")
    client = NLAClient(args.av_dir, device=args.device)
    critic = NLACritic(args.ar_dir, device=args.device)

    conversations = load_conversations(args.shard, args.n_samples)
    rows, n_done = [], 0
    trace = open(args.trace, "w", encoding="utf-8")

    bar = tqdm(total=args.n_samples, desc="samples", unit="conv")
    for i, msgs in enumerate(conversations):
        if n_done >= args.n_samples:
            break
        gold, n_tokens, _ = gold_activation(
            extract_model, tokenizer, msgs, args.extract_index,
            args.position_offset, args.min_position)
        if gold is None:
            continue

        ids, entropies, scores = generate_with_scores(
            av_model, client, gold, args.max_new_tokens)
        # The cap only matters if it binds: explanations run ~136 tokens, so
        # 384 is inert. If this ever fires, the generation was truncated and
        # the closing tag lost — raise --max-new-tokens rather than ignore it.
        assert len(ids) < args.max_new_tokens, (
            f"generation hit --max-new-tokens={args.max_new_tokens}; truncated")
        original = explanation_from_ids(tokenizer, ids)
        if original is None:
            print(f"  skip conv {i}: no closing </explanation>")
            continue
        mse_original, _ = critic.score(original, gold)

        body = set(body_positions(tokenizer, ids, args.position_filter))
        if not body:
            print(f"  skip conv {i}: could not locate explanation body")
            continue
        # Percentile over BODY entropies only — tag tokens sit at entropy 0 and
        # would drag the percentile down, admitting near-deterministic
        # positions into the "fork" pool.
        body_ent = np.array([entropies[j] for j in sorted(body)])
        threshold = float(np.percentile(body_ent, args.entropy_percentile))
        if args.pool_mode == "rank":
            # Rank-based: the dose sets how deep into the entropy ordering we
            # reach. A percentile cut caps the pool at ~27 of ~136 body tokens,
            # so larger doses would be silently skipped. mean_entropy per dose
            # then records how the fork/control gap narrows with dose — at the
            # largest doses the pools approach a partition of the body, which
            # is where the contrast is structurally weakest.
            fork_pool = [int(j) for j in np.argsort(-entropies) if j in body]
            control_pool = [int(j) for j in np.argsort(entropies) if j in body]
        else:
            fork_pool = [int(j) for j in np.argsort(-entropies)
                         if j in body and entropies[j] >= threshold]
            # Same-sized pool from the other end of the SAME population.
            control_pool = [int(j) for j in np.argsort(entropies)
                            if j in body][:len(fork_pool)]
        n_expl_tokens = len(body)
        n_expl_chars = len(original)

        line = [f"[{n_done:3d}] orig={mse_original:.3f} H*={threshold:.2f} "
                f"pool={len(fork_pool)}"]
        for group, pool in (("fork", fork_pool), ("nonfork", control_pool)):
            for dose in tqdm(doses, desc=f"  {group}", leave=False, unit="dose"):
                if dose > len(pool):
                    continue
                # Fork and control must stay disjoint, or the "contrast"
                # compares a set against itself. Skip rather than raise, so an
                # oversized dose degrades the sweep instead of ending the run.
                if 2 * dose > len(body) and args.perturbation == "swap":
                    continue
                positions = pool[:dose]
                rng = np.random.default_rng(args.seed + 1000 * n_done + dose)
                mses, dists, nswaps = [], [], []
                for draw in range(args.n_draws):
                    text, swaps, mean_dist = corrupt(
                        tokenizer, ids, scores, positions, args.top_k, rng,
                        mode=args.perturbation, partner=args.swap_partner,
                        distance=args.swap_distance, body=body,
                        candidate_mode=args.candidate_mode, top_p=args.top_p)
                    if text is None or text == original:
                        continue
                    mse, _ = critic.score(text, gold)
                    mses.append(mse)
                    dists.append(mean_dist)
                    nswaps.append(len(swaps))
                    record = {"sample": n_done, "conv_index": i, "group": group,
                              "dose": dose, "draw": draw, "swaps": swaps,
                              "n_expl_tokens": n_expl_tokens,
                              "n_expl_chars": n_expl_chars,
                              "n_modified_chars": len(text),
                              "perturbation": args.perturbation,
                              "position_filter": args.position_filter,
                              "candidate_mode": args.candidate_mode,
                              "top_p": args.top_p, "top_k": args.top_k,
                              "mean_swap_distance": mean_dist,
                              "n_swaps": len(swaps),
                              "mse_original": mse_original, "mse_modified": mse,
                              "original": original, "modified": text}
                    trace.write(json.dumps(record) + "\n")
                    if weave_run:
                        weave.publish(record, name=f"s{n_done}-{group}-d{dose}-{draw}")
                if not mses:
                    continue
                rows.append({
                    "sample": n_done, "conv_index": i, "group": group, "dose": dose,
                    "n_expl_tokens": n_expl_tokens, "n_expl_chars": n_expl_chars,
                    "position_filter": args.position_filter,
                    "perturbation": args.perturbation,
                    "candidate_mode": args.candidate_mode,
                    "top_p": args.top_p, "top_k": args.top_k,
                    "mean_swap_distance": float(np.mean(dists)) if dists else 0.0,
                    # At high dose the distance constraint cannot always be
                    # satisfied (each swap consumes two positions), so the
                    # realised count can fall below the dose. Logged, not
                    # assumed.
                    "n_swaps_realised": float(np.mean(nswaps)) if nswaps else 0.0,
                    "pool_size": len(pool),
                    "dose_fraction": dose / n_expl_tokens,
                    "mean_entropy": float(np.mean([entropies[j] for j in positions])),
                    "original_mse": mse_original,
                    "distractor_mses": mses,
                    "mean_distractor": float(np.mean(mses)),
                    "var_distractor": float(np.var(mses)),
                    "discrimination_ratio": float(np.mean(mses)) / mse_original,
                })
                if group == "fork":
                    line.append(f"d{dose}:{rows[-1]['discrimination_ratio']:.3f}")
        bar.set_postfix_str(f"orig={mse_original:.3f} pool={len(fork_pool)} "
                            f"len={n_expl_tokens}")
        bar.write("  ".join(line))
        n_done += 1
        bar.update(1)

    bar.close()
    trace.close()
    if not rows:
        raise SystemExit("no usable samples")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        fields = ["sample", "conv_index", "group", "dose", "position_filter",
                  "perturbation", "candidate_mode", "top_p", "top_k",
                  "mean_swap_distance", "n_swaps_realised", "n_expl_tokens",
                  "n_expl_chars", "pool_size", "dose_fraction", "mean_entropy",
                  "original_mse", "mean_distractor", "var_distractor",
                  "discrimination_ratio", "distractor_mses"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, "distractor_mses": json.dumps(r["distractor_mses"])})

    print(f"\nwrote {args.out} and {args.trace}")
    print(f"{'group':<9}{'dose':>5}{'n':>4}{'DR':>9}{'se':>8}{'orig':>8}{'dist':>8}")
    for group in ("fork", "nonfork"):
        for dose in doses:
            subset = [r for r in rows if r["group"] == group and r["dose"] == dose]
            if not subset:
                continue
            s = aggregate(subset, seed=args.seed)
            print(f"{group:<9}{dose:>5}{s['n_samples']:>4}{s['macro_dr']:>9.4f}"
                  f"{s['se_macro_dr']:>8.4f}{s['mean_original']:>8.4f}"
                  f"{s['mean_distractor']:>8.4f}")


if __name__ == "__main__":
    main()
