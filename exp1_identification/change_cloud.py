"""Word cloud of the CHANGES between paired texts, for any experiment here.

Every experiment in this repository produces before/after text pairs -- a
corruption trace, a paraphrase, a counterfactual rival -- and the first question
about any of them is the same: what did it actually change? A frequency cloud of
`was -> now` pairs answers it in one glance and is the cheapest guard against a
pipeline that is technically running but semantically doing the wrong thing. The
paraphrase pipeline generated continuations instead of rewrites for a whole
validated run before anyone looked; this is the thing that would have caught it.

Changes are extracted by word-level diff rather than by trusting the pipeline's
own log, so it also works on outputs that never recorded a swap list.

    # counterfactual rivals, all claim counts (07)
    python exp1_identification/change_cloud.py --jsonl artefacts/rival_texts.jsonl \
        --orig orig --mod 'rivals.*' --out results/figures/rival_cloud.png

    # one claim count only
    python exp1_identification/change_cloud.py --jsonl artefacts/rival_texts.jsonl \
        --orig orig --mod 'rivals.4' --title 'Rivals, 4 claims changed'

    # paraphrases (06)
    python exp1_identification/change_cloud.py --jsonl artefacts/paraphrase_texts.jsonl \
        --orig orig --mod 'heavy'

    # corruption traces (04) -- one record per distractor, not nested
    python exp1_identification/change_cloud.py --jsonl artefacts/traces/hardneg_donor_pos_traces.jsonl \
        --orig original --mod modified
"""
import argparse, difflib, json, re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Same palette as the other figures, so clouds from different experiments can
# sit next to each other in the write-up without looking like different tools.
SURFACE, INK, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e"

PUNCT = re.compile(r"^[\W_]*$")


def pluck(record, path):
    """Values at a dotted path, where '*' expands a dict's values or a list.

    Deliberately tiny: the only shapes in this repository are a string, a list
    of strings, and a dict of lists of strings, so a full JSONPath would be
    four dependencies and no extra reach.
    """
    cur = [record]
    for part in path.split("."):
        nxt = []
        for c in cur:
            if part == "*":
                nxt += list(c.values()) if isinstance(c, dict) else list(c)
            elif isinstance(c, dict) and part in c:
                nxt.append(c[part])
        cur = nxt
    out = []
    for c in cur:
        out += c if isinstance(c, list) else [c]
    return [x for x in out if isinstance(x, str) and x.strip()]


def changes(a, b, min_len=1):
    """[(was, now)] by word-level diff, skipping punctuation-only edits.

    Punctuation-only differences are excluded because restoring a masked
    quotation can move a full stop across the closing quote mark, which is an
    artefact of the pipeline rather than a change anyone made.
    """
    aw, bw = a.split(), b.split()
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, aw, bw).get_opcodes():
        if tag == "equal":
            continue
        was, now = " ".join(aw[i1:i2]), " ".join(bw[j1:j2])
        if PUNCT.match(was.strip()) and PUNCT.match(now.strip()):
            continue
        if re.sub(r"\W", "", was) == re.sub(r"\W", "", now):
            continue                     # same words, punctuation moved
        if len(was) < min_len and len(now) < min_len:
            continue
        out.append((was.strip() or "(none)", now.strip() or "(deleted)"))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jsonl", required=True)
    p.add_argument("--orig", default="orig", help="dotted path to the source text")
    p.add_argument("--mod", default="rivals.*",
                   help="dotted path to the changed text(s); '*' expands a "
                        "dict's values or a list")
    p.add_argument("--out", default="change_cloud.png")
    p.add_argument("--title", default=None)
    p.add_argument("--max-words", type=int, default=180)
    p.add_argument("--min-len", type=int, default=3)
    p.add_argument("--top", type=int, default=25,
                   help="how many pairs to also print, because a cloud is for "
                        "noticing and a list is for checking")
    a = p.parse_args()

    pairs, n_rec, n_texts = Counter(), 0, 0
    for line in open(a.jsonl, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        src = pluck(r, a.orig)
        if not src:
            continue
        n_rec += 1
        for m in pluck(r, a.mod):
            n_texts += 1
            for was, now in changes(src[0], m, a.min_len):
                # ASCII arrow, not U+2192: WordCloud's bundled DroidSansMono has no
                # glyph for it and renders every key with a tofu box.
                pairs[f"{was} -> {now}"] += 1
    if not pairs:
        raise SystemExit(f"no changes found: check --orig/--mod against "
                         f"{a.jsonl}")

    print(f"{n_rec} records, {n_texts} changed texts, "
          f"{sum(pairs.values())} changes, {len(pairs)} distinct")
    for pair, k in pairs.most_common(a.top):
        print(f"  {k:>4}x  {pair}")

    wc = WordCloud(width=1400, height=760, background_color=SURFACE,
                   colormap="Blues", prefer_horizontal=0.95,
                   max_words=a.max_words, min_font_size=9,
                   collocations=False, random_state=0,
                   ).generate_from_frequencies(pairs)
    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(a.title or f"Changes made  ({a.jsonl}, {a.mod})"
                            "  —  size ∝ frequency",
                 color=INK, fontsize=12, weight="semibold", loc="left", pad=10)
    fig.text(0.01, 0.01, f"{sum(pairs.values())} changes over {n_texts} texts, "
             f"{len(pairs)} distinct original→substitute pairs; "
             f"punctuation-only edits excluded",
             color=INK_MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(a.out, facecolor=SURFACE)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
