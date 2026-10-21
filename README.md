# On the Monitorability of Natural Language Autoencoders

Code to reproduce the experiments of *On the Monitorability of Natural Language
Autoencoders: Identification vs Discrimination* (Kevin Yandoka Denamganaï),
[arXiv:2610.18727](https://arxiv.org/abs/2610.18727).

The paper measures one released Natural Language Autoencoder (NLA),
[`kitft/nla-qwen2.5-7b-L20-{av,ar}`](https://huggingface.co/kitft/nla-qwen2.5-7b-L20-ar),
on WildChat conversations:

- **Experiment 1, identification without discrimination.** The AR, used as a
  surrogate monitor, picks an explanation's own activation out of 40 almost
  every time (0.950), yet ranks the true account above a fluent rival that
  falsifies one claim only 0.602 of the time (chance is 0.5).
- **Experiment 2, no private dialect.** Rewording an explanation by an outside
  model barely damages the reconstruction (median 0.021 of the distance to
  another conversation's account), while shuffling the same words costs 0.152.

## Reproduction from artefacts

This re-runs the AR over the released explanations, rivals, judge verdicts,
paraphrases and activations, recomputes every statistic, redraws the figures,
and checks each number against the paper. It needs only the AR, takes about
13 minutes, and matches all 58 checked numbers exactly.

The artefacts are a gated dataset on Hugging Face, because they quote WildChat
conversations: accept its terms on the dataset page and run `hf auth login`
before `setup.sh`.

```bash
./setup.sh                                  # .venv, NLA client, artefacts, AR (11 GB)
bash exp1_identification/run_rescore.sh     # about 5 min
bash exp2_paraphrase/run_rescore.sh         # about 8 min
python check_results.py                     # PASS/FAIL for every number
```

Outputs go to `results/` (JSON) and `results/figures/` (PNG).
`check_results.py` reads `exp*/expected.json`; a number passes when it rounds
to the value printed in the paper.

| paper result | produced by | expected |
|---|---|---|
| Identification: true account beats 320/320 accounts of other activations | `exp1_identification/rivals.py` | AUC 1.000 |
| 40-way retrieval, original / rival | `rivals.py` | 0.950 / 0.949 |
| Judge-confirmed 1-claim rival (Figure: forest plot) | `rivals.py`, `plot_forest.py` | 0.602, SE 0.067, n = 35, CI [0.471, 0.728] |
| Rival minus matched mechanical, 1/2/4 claims | `rivals.py` | +0.145, +0.099, +0.143 (p = 0.048, 0.096, 0.075) |
| Judge confirms 1/2/4-claim rivals | `rivals.py` (from released verdicts) | 48%, 71%, 75% |
| Table: six mechanical baselines at one word | `mechanical_table.py` | 0.480 to 0.509 |
| Figure: angular span (1-token edit, AR error, between conversations) | `angular_span.py` | 1.61°, 22.00°, 45.43° |
| Error increase: rival, other conversation, light paraphrase | `exp2_paraphrase/rival_rows.py` + `analyse_paraphrase.py` | 0.06°, 23.86°, 1.78° |
| Figure: rival word cloud | `change_cloud.py` | — |
| Paraphrase vs shuffle damage (Figure: paraphrase invariance) | `exp2_paraphrase/paraphrase.py` + `analyse_paraphrase.py` | 0.021, 0.152 |
| Retrieval before / after rewriting | `paraphrase.py` | 0.917 / 0.917 |
| Authorship effect | `analyse_paraphrase.py` | +0.073 ± 0.034, p = 0.041 |

The full list, with the JSON path of each number, is in the two `expected.json`
files.

It does not re-run anything that needs a language model other than the AR.
In particular the judge's own validation (97.5% *same* on identical pairs,
100% *different* on heavy corruptions) is not re-run: the released verdicts
are used as they are. The reproduction from scratch re-runs it.

### Options

- `./setup.sh --local-artefacts DIR` copies the artefacts from `DIR` instead of
  downloading them.
- `./setup.sh --ar-dir DIR` uses an AR checkpoint already on disk (it must be
  revision `e2c9e57` of `kitft/nla-qwen2.5-7b-L20-ar`).
- `TORCH_INDEX_URL=... ./setup.sh` picks the torch wheel index (default:
  CUDA 12.8). If `.venv` already has a working torch, `setup.sh` leaves it
  alone, which is how the AMD GPU below was set up: create `.venv` and install
  the torch build for your GPU into it first.
- `setup.sh` needs Python 3.11. On Debian/Ubuntu without `python3.11-venv` it
  falls back to `virtualenv`.

## Reproduction from scratch

This samples WildChat, generates the explanations, rivals, judge verdicts
and paraphrases, and scores them. It also re-runs the judge's validation.
Every script holds one 7B model at a time (about 16 GB of GPU memory).

```bash
./setup.sh --with-av                        # adds the AV, spacy and its English model
mv artefacts artefacts.released             # this writes a fresh artefacts/
bash exp1_identification/run_full.sh        # about 4.5 h
bash exp2_paraphrase/run_full.sh            # about 40 min
python check_results.py
```

`N=100 bash exp1_identification/run_full.sh` (and the same `N` for Experiment
2) runs every stage on 100 conversations instead of the paper's sizes; the
numbers of `check_results.py` then only serve as a reference.

Run at the paper's sizes on the GPU below, every artefact that comes from
greedy decoding or the AR (the explanations, all mechanical corruptions and
their CSVs, the activations, the probe cache) is byte-identical to the
released one, so 32 of the 58 numbers match exactly. Qwen's sampled rivals and
paraphrases come out identical for 32 of 40 and 50 of 60 conversations; the
rest diverge, apparently through GPU nondeterminism in batched sampling. The
numbers that depend on them move within the paper's confidence intervals: the
judge-confirmed 1-claim rival AUC is 0.605, 95% CI [0.469, 0.735], against
0.602 [0.471, 0.728].

## Hardware

- Tested on one AMD Radeon 8060S (128 GB unified memory) under ROCm 7.0, with
  torch 2.10 (nightly), Python 3.11. Not yet tested on an NVIDIA GPU.
- The reproduction from artefacts loads only the AR, in bfloat16 (about 11 GB
  of GPU memory and disk).
- The reproduction from scratch also loads the AV and Qwen2.5-7B-Instruct, one
  at a time (about 15 GB each, so about 41 GB of disk for the three models).

Floating-point results can differ slightly across GPUs and kernels. Since most
statistics compare cosines between reconstructions, a near-tie may flip on
different hardware and move a number in its last printed digit.

## Credits and licences

- Code: MIT (see `LICENSE`).
- NLA checkpoints: `kitft/nla-qwen2.5-7b-L20-{av,ar}`, Apache-2.0, downloaded
  from Hugging Face and not redistributed here. The inference client
  `nla/nla_inference.py` is fetched by `setup.sh` from
  [kitft/nla-inference](https://github.com/kitft/nla-inference) (Apache-2.0)
  at a pinned commit.
- Conversations: [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M)
  (ODC-BY). The released artefacts contain excerpts derived from it and carry
  the same licence; see the dataset card.
- Qwen2.5-7B-Instruct (Apache-2.0) writes the rivals and paraphrases and acts
  as the judge.

```bibtex
@inproceedings{zhao2024wildchat,
  title={WildChat: 1M Chat{GPT} Interaction Logs in the Wild},
  author={Wenting Zhao and Xiang Ren and Jack Hessel and Claire Cardie and Yejin Choi and Yuntian Deng},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=Bl8u7ZRlbM}
}
@misc{qwen2025qwen25technicalreport,
  title={Qwen2.5 Technical Report},
  author={Qwen Team},
  year={2025},
  eprint={2412.15115},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2412.15115}
}
```

## Citation

```bibtex
@misc{denamganaï2026monitorabilitynaturallanguageautoencoders,
      title={On the Monitorability of Natural Language Autoencoders: Identification vs Discrimination},
      author={Kevin Yandoka Denamganaï},
      year={2026},
      eprint={2610.18727},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2610.18727},
}
```
