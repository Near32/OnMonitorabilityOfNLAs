"""Discrimination Ratio for a released NLA over WildChat samples (Variant C).

One sample = one WildChat conversation. For each: recompute the layer-20
activation near the end of the reply, have the AV verbalize it, then score the
AV's explanation and N word-shuffled versions of it through the AR.

DR = mean(MSE_distractor) / MSE_original, where the critic's MSE is 2*(1-cos)
because it L2-normalizes both vectors to sqrt(d_model) first. Range [0, 4],
2.0 = orthogonal. Direction-only: magnitude is discarded.

Raw per-shuffle MSEs are written to the CSV as a JSON column so any
aggregation (pooled vs. mean-of-means, variance within vs. across samples) can
be recomputed without re-running the models.

    PYTHONPATH=nla python nla/dr.py --n-samples 20
"""

import argparse
import csv
import json
import random
import re

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla_inference import NLAClient, NLACritic

EXPLANATION_RE = re.compile(r"<explanation>(.*?)</explanation>", re.S)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-samples", type=int, default=20,
                   help="conversations to score")
    p.add_argument("--n-shuffles", type=int, default=10,
                   help="word-shuffled distractors per explanation")
    p.add_argument("--extract-model", default="Qwen/Qwen2.5-7B-Instruct")
    # NOT "./av": the repo root is on sys.path, so a directory named av/ becomes
    # a namespace package shadowing PyAV, which transformers probes for video
    # support. The probe then fails and transformers silently drops every
    # torch-backed class — surfacing as "Could not find Qwen2ForCausalLM".
    p.add_argument("--av-dir", default="models/av")
    p.add_argument("--ar-dir", default="models/ar")
    # HF returns embeddings at index 0, so block N's output is hidden_states[N+1].
    # Qwen L20 -> 21; Gemma-12B L32 -> 33, Gemma-27B L41 -> 42, Llama-70B L53 -> 54.
    p.add_argument("--extract-index", type=int, default=21,
                   help="hidden_states index (= layer + 1)")
    # NLA datagen used min_position=50, so shorter sequences are off-distribution.
    p.add_argument("--min-position", type=int, default=50,
                   help="skip conversations shorter than this many tokens")
    # A rendered conversation ends '...', '<|im_end|>', '\n', so offset 0 is the
    # newline and offset 1 is <|im_end|> itself — both special tokens, and the
    # worked example shows <|im_end|> reconstructing at 0.583 against ~0.19 for
    # ordinary text. Offset 2 is the last real token of the reply.
    p.add_argument("--position-offset", type=int, default=2,
                   help="tokens back from the end to extract at")
    p.add_argument("--position-offsets", default=None,
                   help="comma-separated offsets to sweep in ONE process, "
                        "reusing the loaded models (adds an `offset` column). "
                        "Overrides --position-offset when given.")
    # Published explanations run ~650 chars plus <explanation> tags, which is at
    # the edge of 200 tokens; truncating loses the closing tag, so the regex
    # fails and the raw tagged text gets scored instead.
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="AV sampling temperature; 0 = greedy")
    p.add_argument("--shard", type=int, default=0, help="WildChat parquet shard")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="dr_wildchat.csv")
    return p.parse_args()


def load_conversations(shard: int, limit: int):
    """WildChat-1M conversations, English and non-toxic, as message lists."""
    path = hf_hub_download(
        "allenai/WildChat-1M", f"data/train-{shard:05d}-of-00014.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(path, columns=["conversation", "language", "toxic"])
    df = df[(df["language"] == "English") & (~df["toxic"])]
    out = []
    for conv in df["conversation"].head(limit * 4):
        msgs = [{"role": t["role"], "content": t["content"]} for t in conv]
        if msgs:
            out.append(msgs)
    return out


def gold_activation(model, tokenizer, messages, extract_index, offset, min_position):
    """Activation near the end of a rendered conversation, or None if too short."""
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=False)
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=4096).to(model.device)
    n_tokens = inputs["input_ids"].shape[1]
    if n_tokens < min_position:
        return None, n_tokens, None
    position = n_tokens - 1 - offset
    # A negative index would silently wrap to the END of the sequence, scoring
    # a special token while reporting the requested offset. Skip instead.
    if position < 0 or position < min_position // 2:
        return None, n_tokens, None
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    token = tokenizer.decode(inputs["input_ids"][0, position])
    return out.hidden_states[extract_index][0, position].float().cpu(), n_tokens, token


def verbalize(av_model, client, activation, max_new_tokens, temperature):
    """Run the AV locally: inject the activation, generate, strip the tags.

    Reuses NLAClient._build_embeds for the injection so the injection char,
    its neighbour validation and the injection_scale rescaling stay identical
    to the reference implementation. Only the transport changes — HF generate
    instead of an SGLang endpoint.
    """
    embeds_np, _ = client._build_embeds(activation, None)
    embeds = torch.from_numpy(embeds_np).unsqueeze(0).to(
        av_model.device, av_model.dtype)
    attn = torch.ones(embeds.shape[:2], dtype=torch.long, device=av_model.device)
    with torch.no_grad():
        gen = av_model.generate(
            inputs_embeds=embeds, attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=client.tokenizer.pad_token_id
            or client.tokenizer.eos_token_id,
        )
    # With inputs_embeds and no input_ids, generate returns only new tokens.
    text = client.tokenizer.decode(gen[0], skip_special_tokens=False)
    match = EXPLANATION_RE.search(text)
    if match:
        return match.group(1).strip(), True

    # No closing tag: the generation hit max_new_tokens mid-explanation. Scoring
    # the raw text would feed the AR a literal "<explanation>" tag plus a
    # truncated sentence, which is a silently different experiment — so strip
    # the tags that are present and say so, rather than returning it quietly.
    print(f"  WARNING: no </explanation> in {len(text)} chars — truncated? "
          f"raise --max-new-tokens")
    cleaned = text.replace("<explanation>", "").replace("</explanation>", "")
    for special in (client.tokenizer.eos_token, client.tokenizer.pad_token):
        if special:
            cleaned = cleaned.replace(special, "")
    return cleaned.strip(), False


def shuffle_explanation(explanation: str, seed: int) -> str:
    """Same words, destroyed order — preserves the token multiset exactly."""
    words = explanation.split()
    random.Random(seed).shuffle(words)
    return " ".join(words)


def score_sample(critic, explanation, gold, n_shuffles, seed_base):
    """Original MSE plus one MSE per shuffled distractor. Returns raw lists."""
    mse_original, cos_original = critic.score(explanation, gold)
    mses = [critic.score(shuffle_explanation(explanation, seed_base + s), gold)[0]
            for s in range(n_shuffles)]
    return {
        "original_mse": mse_original,
        "original_cos": cos_original,
        "distractor_mses": mses,
        "mean_distractor": float(np.mean(mses)),
        "var_distractor": float(np.var(mses)),
        "discrimination_ratio": float(np.mean(mses)) / mse_original,
    }


def aggregate(rows, n_bootstrap=1000, seed=0):
    """Macro, micro, pooled spread, and standard errors.

    macro: mean of per-sample DRs — every sample weighted equally, so a sample
        whose original reconstructed badly (small denominator) can dominate.
    micro: pooled numerator / pooled denominator, divided once — large-MSE
        samples weigh more, and no single small denominator can inflate it.

    The pooled variance is over every (sample, shuffle) MSE, so it carries the
    between-sample spread that averaging per-sample variances would discard.

    THE INDEPENDENT UNIT IS THE CONVERSATION, NOT THE SHUFFLE. The shuffles
    within one sample share an activation, an explanation and a token multiset,
    so they are strongly correlated; dividing by the number of distractor
    scores would understate the standard error by roughly sqrt(n_shuffles).
    Hence se_macro divides by sqrt(len(rows)), and micro — a ratio of pooled
    means, with no closed-form SE — is bootstrapped by resampling whole
    conversations, which respects the clustering by construction.
    """
    all_d = [m for r in rows for m in r["distractor_mses"]]
    originals = [r["original_mse"] for r in rows]
    per_sample_dr = [r["discrimination_ratio"] for r in rows]
    n = len(rows)

    macro = float(np.mean(per_sample_dr))
    # ddof=1: sample standard deviation, since these 20 conversations are a
    # sample from WildChat rather than the population.
    se_macro = float(np.std(per_sample_dr, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        d = np.mean([m for i in idx for m in rows[i]["distractor_mses"]])
        o = np.mean([rows[i]["original_mse"] for i in idx])
        boot.append(d / o)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    return {
        "n_samples": n,
        "n_distractor_scores": len(all_d),
        "macro_dr": macro,
        "se_macro_dr": se_macro,
        "macro_dr_ci95_lo": macro - 1.96 * se_macro,
        "macro_dr_ci95_hi": macro + 1.96 * se_macro,
        "micro_dr": float(np.mean(all_d)) / float(np.mean(originals)),
        "se_micro_dr_boot": float(np.std(boot, ddof=1)),
        "micro_dr_ci95_lo": float(lo),
        "micro_dr_ci95_hi": float(hi),
        "mean_original": float(np.mean(originals)),
        "se_original": float(np.std(originals, ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
        "var_original": float(np.var(originals)),
        "mean_distractor": float(np.mean(all_d)),
        "var_distractor_pooled": float(np.var(all_d)),
        "var_distractor_within_mean": float(np.mean([r["var_distractor"] for r in rows])),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.extract_model)
    extract_model = AutoModelForCausalLM.from_pretrained(
        args.extract_model, dtype=torch.bfloat16, device_map="auto")
    av_model = AutoModelForCausalLM.from_pretrained(
        args.av_dir, dtype=torch.bfloat16, device_map="auto")
    client = NLAClient(args.av_dir, device=args.device)
    critic = NLACritic(args.ar_dir, device=args.device)

    offsets = ([int(o) for o in args.position_offsets.split(",")]
               if args.position_offsets else [args.position_offset])
    conversations = load_conversations(args.shard, args.n_samples)
    rows, skipped = [], 0

    # One bar over the whole sweep; models load once for all offsets.
    bar = tqdm(total=args.n_samples * len(offsets), desc="samples", unit="conv")
    for offset in offsets:
      done_this_offset = 0
      for i, msgs in enumerate(conversations):
        if done_this_offset >= args.n_samples:
            break
        gold, n_tokens, token = gold_activation(
            extract_model, tokenizer, msgs, args.extract_index,
            offset, args.min_position)
        if gold is None:
            skipped += 1
            continue

        explanation, tagged = verbalize(
            av_model, client, gold, args.max_new_tokens, args.temperature)
        row = score_sample(critic, explanation, gold, args.n_shuffles,
                           args.seed * 1000 + i * args.n_shuffles)
        # Explanation length is logged, not just the text: the AV emits a
        # near-fixed ~136-token template, so a sample that deviates is worth
        # finding post-hoc rather than discovering as unexplained variance.
        row.update(sample=len(rows), conv_index=i, n_tokens=n_tokens,
                   offset=offset,
                   token=token, tagged=tagged, explanation=explanation,
                   n_expl_chars=len(explanation),
                   n_expl_words=len(explanation.split()))
        rows.append(row)
        done_this_offset += 1
        bar.write(f"[off{offset:>3} {row['sample']:3d}] tok={token!r:<14} len={n_tokens:<5} "
                  f"expl={row['n_expl_words']:<4}w "
                  f"DR={row['discrimination_ratio']:.3f} "
                  f"orig={row['original_mse']:.3f} dist={row['mean_distractor']:.3f}")
        bar.update(1)

    bar.close()
    if not rows:
        raise SystemExit(f"no usable samples (skipped {skipped} as too short)")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        fields = ["sample", "conv_index", "offset", "n_tokens", "token", "tagged",
                  "n_expl_chars", "n_expl_words",
                  "original_mse", "original_cos", "mean_distractor",
                  "var_distractor", "discrimination_ratio", "distractor_mses",
                  "explanation"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, "distractor_mses": json.dumps(r["distractor_mses"])})

    summary = aggregate(rows)
    print(f"\nwrote {args.out}  (skipped {skipped} short conversations)")
    for k, v in summary.items():
        print(f"  {k:26s} {v:.4f}" if isinstance(v, float) else f"  {k:26s} {v}")


if __name__ == "__main__":
    main()
