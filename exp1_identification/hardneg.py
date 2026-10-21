"""Confabulation-like discrimination: target vs. perturbed versions of ITSELF.

The cross-pairing experiment used distractors from other conversations, which
differ from the target in essentially every content word — bag-of-embeddings
within-source cosine 0.990 vs between-source 0.839, 1-NN source purity 1.00.
So its 97% top-1 was measured on a far easier task than confabulation
discrimination, and says little about whether the AR can catch a confabulated
specific.

Here the distractors are the target's OWN explanation with `dose` content
tokens replaced from the AV's own top-k or nucleus — function words, punctuation
and fragments excluded, so every substitution lands on a meaning-bearing word.
That is a confabulation by construction: same topic, same structure, same
register, different specifics.

This is the §5 discriminative metric as originally specified, with a defined
chance baseline of 1/(K+1).

It holds one 7B model at a time: the extraction model writes the gold
activations to --gold-cache (shared by every run and by probe_ar.py), then the
AV writes every distractor text, then the AR scores them. The AV's greedy
explanations are cached in --gen-cache, so a run that needs no AV scores (the
donor modes without --trace) loads no AV for conversations already explained.

    PYTHONPATH=nla python exp1_identification/hardneg.py --n-samples 100
"""

import argparse
import bisect
import csv
import gc
import json
import os
import re
from collections import defaultdict

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dr import gold_activation, load_conversations
from fork import (alternatives, body_positions, corrupt, explanation_from_ids,
                  generate_with_scores, token_class)
from nla_inference import NLAClient, NLACritic

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def char_offsets(tokenizer, ids):
    """Cumulative character offset of each token in the decoded string.

    Computed once per generation (O(T) decodes) and reused for every swap;
    doing it per swap would be O(T x swaps) and dominate the run.
    """
    offs, prev = [], 0
    for i in range(len(ids)):
        offs.append(prev)
        prev = len(tokenizer.decode(ids[: i + 1], skip_special_tokens=False))
    return offs


def sentences_with_spans(text):
    """[(start, end, sentence)] over `text`, split on sentence-final punctuation."""
    out, pos = [], 0
    for part in SENT_SPLIT.split(text):
        start = text.find(part, pos)
        if start < 0:
            start = pos
        out.append((start, start + len(part), part))
        pos = start + len(part)
    return out


def tag_by_position(nlp, text, offsets):
    """BPE position -> Penn tag, aligned through character offsets.

    spaCy gives every token an `idx` character offset and `char_offsets` gives
    the same for BPE tokens, so the two are joined by a bisect rather than by
    string matching, which would break on subword pieces and punctuation.
    """
    doc = nlp(text)
    starts = [t.idx for t in doc]
    tags = [t.tag_ for t in doc]
    out = {}
    for j, c in enumerate(offsets):
        # A word-initial BPE token is " alive", so its offset points at the
        # SPACE and a naive bisect returns the previous word's tag. Advance past
        # whitespace first, or every substitution is matched to the wrong class.
        while c < len(text) and text[c].isspace():
            c += 1
        i = bisect.bisect_right(starts, c) - 1
        out[j] = tags[i] if 0 <= i < len(tags) else None
    return out


def input_vocabulary(messages):
    """Lower-cased alphabetic words appearing in the conversation.

    Membership here is the proxy for "grounded". It is a proxy, not falsity: an
    explanation can use a word absent from the input and still be thematically
    true — the NLA paper's "faithful in theme, wrong in specifics". These levers
    guarantee UNGROUNDEDNESS; only a judge could certify falsity.
    """
    text = " ".join(m["content"] for m in messages).lower()
    return set(re.findall(r"[a-z]{3,}", text))


def coarse(tag):
    """NN/NNS/NNP -> NN, VB/VBD/VBG -> VB, etc. Fallback when a fine bucket is
    empty; keeps the substitution in the right broad class even when tense or
    number cannot be matched."""
    return tag[:2] if tag else None


def build_donor_pool(explain, tokenizer, golds, n_donors=12, nlp=None):
    """Content words harvested from OTHER conversations' explanations.

    Substituting one of these produces a fluent sentence asserting something
    about the input that is not there — a confabulation by construction. The
    price is that the substitute is NOT drawn from the AV's distribution at that
    position, so donor swaps are less locally plausible than top-p ones; that
    is the trade being made deliberately, and `p_substitute` is logged so the
    implausibility can be quantified rather than assumed.
    """
    pool = [] if nlp is None else defaultdict(list)
    for ci in range(len(golds) - 1, len(golds) - 1 - n_donors, -1):
        if ci < 0 or golds[ci] is None:
            continue
        ids, _, _ = explain(ci)
        text = explanation_from_ids(tokenizer, ids)
        if not text:
            continue
        if nlp is None:
            for w in text.split():
                if token_class(" " + w) in ("content", "entity"):
                    tok_ids = tokenizer(" " + w, add_special_tokens=False)["input_ids"]
                    if len(tok_ids) == 1:
                        pool.append(tok_ids[0])
        else:
            # Tag donors in THEIR own context: "closing" is NN in "the closing
            # sentence" and VBG in "closing the file", so a type-level tag
            # would mismatch about as often as no tag at all.
            for t in nlp(text):
                if token_class(" " + t.text) not in ("content", "entity"):
                    continue
                tok_ids = tokenizer(" " + t.text, add_special_tokens=False)["input_ids"]
                if len(tok_ids) == 1:
                    pool[t.tag_].append(tok_ids[0])
    if nlp is None:
        return sorted(set(pool))
    merged = {k: sorted(set(v)) for k, v in pool.items()}
    for k, v in list(merged.items()):                      # coarse fallback buckets
        merged.setdefault(coarse(k), [])
        merged[coarse(k)] = sorted(set(merged[coarse(k)]) | set(v))
    return merged


def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def gold_cache(args, conversations):
    """Gold activation (or None) of every conversation, by index.

    Computed with the extraction model alone and saved to --gold-cache, so the
    seven runs and probe_ar.py share it and never hold the extraction model
    next to the AV or the AR. load_conversations returns prefixes of one list,
    so a longer list only extends the cache.
    """
    key = [args.extract_model, args.shard, args.extract_index,
           args.position_offset, args.min_position]
    golds = []
    if os.path.exists(args.gold_cache):
        saved = torch.load(args.gold_cache)
        if saved["key"] == key:
            golds = saved["golds"]
    if len(golds) >= len(conversations):
        return golds[:len(conversations)]
    tokenizer = AutoTokenizer.from_pretrained(args.extract_model)
    extract = AutoModelForCausalLM.from_pretrained(
        args.extract_model, dtype=torch.bfloat16, device_map="auto")
    for msgs in tqdm(conversations[len(golds):], desc="gold activations"):
        golds.append(gold_activation(extract, tokenizer, msgs, args.extract_index,
                                     args.position_offset, args.min_position)[0])
    del extract
    free_gpu()
    os.makedirs(os.path.dirname(args.gold_cache) or ".", exist_ok=True)
    torch.save({"key": key, "golds": golds}, args.gold_cache)
    return golds


def swap_probabilities(scores, ids, swaps, entropies, tokenizer, top_k):
    """Attach the AV's own distribution facts to each swap.

    p_original and p_substitute are what make plausibility analysable post-hoc:
    top-k ignores probability, so a swap at a confident position inserts a token
    the AV rated ~0. Logging both, plus the entropy and the full candidate set,
    means that can be quantified later without re-running the AV.
    """
    for s in swaps:
        j = s["pos"]
        probs = torch.softmax(scores[j][0].float(), dim=-1)
        sub_ids = tokenizer(s["now"], add_special_tokens=False)["input_ids"]
        sub_id = sub_ids[0] if len(sub_ids) == 1 else None
        top = torch.topk(probs, top_k + 1)
        s["entropy"] = float(entropies[j])
        s["p_original"] = float(probs[int(ids[j])])
        s["p_substitute"] = float(probs[sub_id]) if sub_id is not None else None
        s["candidates"] = [[tokenizer.decode([int(i)]), float(p)]
                           for p, i in zip(top.values.tolist(), top.indices.tolist())]
    return swaps


def sentence_context(swaps, offsets, orig_text, mod_text, full_orig):
    """Attach the containing sentence, before and after, to each swap.

    Token positions index the FULL generation (tags included), so offsets are
    taken against that string and then mapped into the extracted explanation by
    sentence index — replacement never inserts or deletes tokens, so sentence
    numbering is stable. When it is not (a substituted token introducing or
    removing terminal punctuation), the after-sentence is recorded as None
    rather than guessed.
    """
    o_sents = sentences_with_spans(orig_text)
    m_sents = sentences_with_spans(mod_text)
    f_sents = sentences_with_spans(full_orig)
    for s in swaps:
        c = offsets[s["pos"]] if s["pos"] < len(offsets) else -1
        k = next((i for i, (a, b, _) in enumerate(f_sents) if a <= c < b), None)
        # The full generation carries the <explanation> tag as its own leading
        # sentence-ish chunk, so align by matching text rather than index.
        sent = f_sents[k][2] if k is not None else None
        j = next((i for i, (_, _, t) in enumerate(o_sents)
                  if sent and t and t in sent), None)
        s["sentence_before"] = o_sents[j][2] if j is not None else sent
        s["sentence_after"] = (m_sents[j][2]
                               if j is not None and j < len(m_sents) else None)
    return swaps


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--n-distractors", type=int, default=15,
                   help="K; chance top-1 is 1/(K+1)")
    p.add_argument("--doses", default="1,2,4,8,16")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--candidate-mode",
                   choices=("topk", "topp", "donor", "donor_pos"), default="topk",
                   help="topk/topp draw substitutes from the AV's own distribution "
                        "at that position, which mostly yields paraphrases: only "
                        "~13%% of such swaps are lexical confabulations, and 62%% "
                        "are unlabelable because these explanations describe rather "
                        "than quote. donor instead substitutes a content word taken "
                        "from a DIFFERENT conversation's explanation, so the "
                        "resulting claim is about something demonstrably absent "
                        "from this input — a confabulation by construction rather "
                        "than by adjudication. donor_pos additionally matches "
                        "the Penn POS tag of the word being replaced, so the "
                        "sentence stays grammatical and the manipulation isolates "
                        "falsity from ungrammaticality — which matters because "
                        "the AR is independently known to react to word order.")
    p.add_argument("--spacy-model", default="en_core_web_sm")
    p.add_argument("--donor-absent-only", action="store_true",
                   help="LEVER 1: reject donors whose word already appears in "
                        "THIS conversation's input. Without it, falsity holds "
                        "only probabilistically — a donor from another "
                        "conversation can coincidentally be true here. With it, "
                        "the inserted word is lexically ungrounded by "
                        "construction, which is the operational definition the "
                        "'fixing rewards' post uses.")
    p.add_argument("--min-donor-distance", type=float, default=None,
                   help="LEVER 3: reject donors whose cosine similarity to the "
                        "replaced token exceeds this, in the AR's input "
                        "embedding space. Removes accidental near-paraphrases "
                        "(creative -> additional) while keeping genuine "
                        "confabulations (journey -> hypertension). 0.13 is the "
                        "~90th percentile of random in-use pairs (see 03).")
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--position-filter", choices=("none", "content", "entity"),
                   default="content")
    p.add_argument("--extract-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--gold-cache", default="artefacts/cache/golds.pt",
                   help="gold activations per conversation, written on first use")
    p.add_argument("--gen-cache", default="artefacts/cache/explanations.pt",
                   help="the AV's greedy token ids per conversation")
    p.add_argument("--av-dir", default="models/av")
    p.add_argument("--ar-dir", default="models/ar")
    p.add_argument("--extract-index", type=int, default=21)
    p.add_argument("--min-position", type=int, default=50)
    p.add_argument("--position-offset", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="hardneg.csv")
    p.add_argument("--summary", default="hardneg.json")
    p.add_argument("--trace", default=None,
                   help="optional JSONL of every distractor: the swap list, both "
                        "texts and both MSEs. Off by default because the ranks "
                        "are the result; needed to inspect WHAT was substituted.")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    doses = [int(d) for d in args.doses.split(",")]

    # ---- stage 1: the extraction model alone
    conversations = load_conversations(args.shard, args.n_samples)
    golds = gold_cache(args, conversations)

    # ---- stage 2: the AV alone writes every distractor text
    tokenizer = AutoTokenizer.from_pretrained(args.extract_model)
    # Greedy, so a cached generation is the one the AV would produce again.
    gen_key = [os.path.realpath(args.av_dir), args.max_new_tokens,
               args.extract_model, args.shard, args.extract_index,
               args.position_offset, args.min_position]
    gens = {}
    if os.path.exists(args.gen_cache):
        saved = torch.load(args.gen_cache)
        if saved["key"] == gen_key:
            gens = saved["gens"]
    n_cached = len(gens)
    need_scores = bool(args.trace) or args.candidate_mode in ("topk", "topp")
    av = client = None

    def explain(ci):
        """(ids, entropies, scores) of the AV on conversation ci; scores is
        None when served from the cache."""
        nonlocal av, client
        if not need_scores and ci in gens:
            ids, ent = gens[ci]
            return ids, ent.numpy(), None
        if av is None:
            av = AutoModelForCausalLM.from_pretrained(
                args.av_dir, dtype=torch.bfloat16, device_map="auto")
            client = NLAClient(args.av_dir, device=args.device)
        ids, ent, scores = generate_with_scores(av, client, golds[ci],
                                                args.max_new_tokens)
        gens[ci] = (ids.cpu(), torch.from_numpy(ent))
        return ids, ent, scores

    embed_n = None
    if args.min_donor_distance is not None:
        from nla_inference import load_embedding_only
        W = load_embedding_only(args.ar_dir, dtype=torch.float32).weight.detach()
        embed_n = W / W.norm(dim=1, keepdim=True).clamp_min(1e-12)
    nlp = None
    if args.candidate_mode == "donor_pos":
        import spacy
        nlp = spacy.load(args.spacy_model, disable=["ner", "lemmatizer", "parser"])
    donor_pool = []
    if args.candidate_mode in ("donor", "donor_pos"):
        # Harvested from the TAIL of the list so donors never overlap the
        # conversations being scored.
        donor_pool = build_donor_pool(explain, tokenizer, golds, nlp=nlp)
        if nlp is None:
            print(f"donor pool: {len(donor_pool)} distinct content tokens")
        else:
            n = sum(len(v) for k, v in donor_pool.items() if len(k) > 2)
            print(f"donor pool: {n} tokens across {len([k for k in donor_pool if len(k) > 2])} "
                  f"fine tags; largest: "
                  f"{sorted(((len(v), k) for k, v in donor_pool.items()), reverse=True)[:4]}")
    # One entry per sample: its original and, per dose, [(text, trace record)].
    samples, done = [], 0
    bar = tqdm(total=args.n_samples, desc="AV: samples", unit="conv")

    for ci, msgs in enumerate(conversations):
        if done >= args.n_samples:
            break
        gold = golds[ci]
        if gold is None:
            continue
        ids, entropies, scores = explain(ci)
        original = explanation_from_ids(tokenizer, ids)
        if original is None:
            continue
        full_original = tokenizer.decode(ids, skip_special_tokens=False)
        offsets = (char_offsets(tokenizer, ids)
                   if (args.trace or args.candidate_mode == "donor_pos") else [])
        pos_tags = (tag_by_position(nlp, full_original, offsets)
                    if args.candidate_mode == "donor_pos" else {})
        in_vocab = input_vocabulary(msgs) if args.donor_absent_only else None
        pool = body_positions(tokenizer, ids, args.position_filter)
        if len(pool) < max(doses):
            continue
        sample = {"sample": done, "ci": ci, "original": original,
                  "pool_size": len(pool), "doses": {}}

        for dose in doses:
            rng = np.random.default_rng(args.seed + 1000 * done + dose)
            items = sample["doses"][dose] = []
            for _ in range(args.n_distractors):
                # Positions are drawn at random from the eligible pool, not by
                # entropy rank: a confabulation is not confined to the AV's
                # most uncertain slots, and ranking would make the distractor
                # set deterministic across draws.
                positions = rng.choice(pool, size=dose, replace=False).tolist()
                if args.candidate_mode in ("donor", "donor_pos"):
                    swapped = ids.clone()
                    swaps = []
                    for j in positions:
                        if args.candidate_mode == "donor_pos":
                            tag = pos_tags.get(j)
                            bucket = donor_pool.get(tag) or donor_pool.get(coarse(tag))
                        else:
                            bucket = donor_pool
                        if not bucket:
                            continue              # no same-class donor available
                        # Levers 1 and 3 filter the bucket rather than rejecting
                        # after drawing, so a constrained bucket still yields a
                        # substitution instead of silently dropping the whole
                        # distractor.
                        if in_vocab is not None:
                            bucket = [c for c in bucket
                                      if tokenizer.decode([c]).strip().lower()
                                      not in in_vocab]
                        if args.min_donor_distance is not None and bucket:
                            orig = int(ids[j])
                            sims = embed_n[bucket] @ embed_n[orig]
                            bucket = [c for c, sim in zip(bucket, sims.tolist())
                                      if sim <= args.min_donor_distance]
                        if not bucket:
                            continue
                        cand = int(rng.choice(bucket))
                        if cand == int(ids[j]):
                            continue
                        swaps.append({"pos": int(j),
                                      "was": tokenizer.decode([int(ids[j])]),
                                      "now": tokenizer.decode([cand]),
                                      "tag": pos_tags.get(j),
                                      "bucket_size": len(bucket)})
                        swapped[j] = cand
                    text = explanation_from_ids(tokenizer, swapped)
                else:
                    text, swaps, _ = corrupt(tokenizer, ids, scores, positions,
                                             args.top_k, rng, mode="replace",
                                             candidate_mode=args.candidate_mode,
                                             top_p=args.top_p)
                if text is None or text == original:
                    continue
                record = None
                if args.trace:
                    swaps = swap_probabilities(scores, ids, swaps, entropies,
                                               tokenizer, args.top_k)
                    swaps = sentence_context(swaps, offsets, original, text,
                                             full_original)
                    # The MSE and cosine fields are filled in by the AR below.
                    record = {
                        "sample": done, "conv_index": ci, "dose": dose,
                        "distractor": len(items), "group": "hardneg",
                        "swaps": swaps, "n_swaps": len(swaps),
                        "perturbation": "replace",
                        "candidate_mode": args.candidate_mode,
                        "top_p": args.top_p, "top_k": args.top_k,
                        "position_filter": args.position_filter,
                        "pool_size": len(pool),
                        "n_expl_tokens": len(ids),
                        "n_expl_chars": len(original),
                        "n_modified_chars": len(text),
                        "mse_original": None, "cos_original": None,
                        "mse_modified": None, "cos_modified": None,
                        "delta_mse": None,
                        "mean_entropy": float(np.mean([s["entropy"] for s in swaps]))
                                        if swaps else None,
                        "original": original, "modified": text,
                        "full_generation": full_original}
                items.append((text, record))
        samples.append(sample)
        bar.update(1); done += 1
    bar.close()
    av = client = scores = None
    free_gpu()
    if len(gens) > n_cached:
        os.makedirs(os.path.dirname(args.gen_cache) or ".", exist_ok=True)
        torch.save({"key": gen_key, "gens": gens}, args.gen_cache)

    # ---- stage 3: the AR alone scores them
    critic = NLACritic(args.ar_dir, device=args.device)
    trace = open(args.trace, "w", encoding="utf-8") if args.trace else None
    rows = []
    for sample in tqdm(samples, desc="AR: samples", unit="conv"):
        gold = golds[sample["ci"]]
        mse_orig, cos_orig = critic.score(sample["original"], gold)
        for dose, items in sample["doses"].items():
            mses = []
            for text, record in items:
                mse, cos = critic.score(text, gold)
                mses.append(mse)
                if trace:
                    record.update(mse_original=mse_orig, cos_original=cos_orig,
                                  mse_modified=mse, cos_modified=cos,
                                  delta_mse=mse - mse_orig)
                    trace.write(json.dumps(record) + "\n")
            if not mses:
                continue
            arr = np.array(mses)
            # Lower MSE is better, so the target wins if it beats every distractor.
            rank = int((arr < mse_orig).sum()) + 1
            rows.append({
                "sample": sample["sample"], "conv_index": sample["ci"], "dose": dose,
                "n_distractors": len(arr), "original_mse": mse_orig,
                "mean_distractor_mse": float(arr.mean()),
                "min_distractor_mse": float(arr.min()),
                "rank": rank, "correct": int(rank == 1),
                "margin": float(arr.min() - mse_orig),
                "pool_size": sample["pool_size"],
            })
    if trace:
        trace.close()

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    summary = {}
    print(f"\n{'dose':>5}{'n':>5}{'top1':>8}{'chance':>8}{'mrr':>8}"
          f"{'margin':>9}{'orig':>8}{'minDist':>9}")
    for dose in doses:
        sub = [r for r in rows if r["dose"] == dose]
        if not sub:
            continue
        ranks = np.array([r["rank"] for r in sub])
        k = float(np.mean([r["n_distractors"] for r in sub]))
        s = {"n": len(sub), "top1": float((ranks == 1).mean()),
             "chance": 1.0 / (k + 1), "mrr": float((1.0 / ranks).mean()),
             "mean_margin": float(np.mean([r["margin"] for r in sub])),
             "mean_original": float(np.mean([r["original_mse"] for r in sub])),
             "mean_min_distractor": float(np.mean([r["min_distractor_mse"] for r in sub])),
             "se_top1": float(np.std((ranks == 1).astype(float), ddof=1) / np.sqrt(len(sub)))}
        summary[dose] = s
        print(f"{dose:>5}{s['n']:>5}{s['top1']:>8.3f}{s['chance']:>8.3f}"
              f"{s['mrr']:>8.3f}{s['mean_margin']:>+9.4f}"
              f"{s['mean_original']:>8.3f}{s['mean_min_distractor']:>9.3f}")

    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "summary": summary}, f, indent=2)
    print(f"\nwrote {args.out} and {args.summary}")


if __name__ == "__main__":
    main()
