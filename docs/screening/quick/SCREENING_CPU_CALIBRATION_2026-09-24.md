# CPU screening crossover measurements — 2026-09-24

**Status: complete. No RankSEG code or default changed.**

Follow-up: [uniformly subsampled real probabilities](SCREENING_REAL_SCALES_2026-09-25.md)
tests 40 new source inputs across 12 sizes, without synthetic distributions,
and separates entirely class-pruned inputs from the active-input speedups.

The earlier proposed CPU cutoff of **65,536 was not a measured crossover**.
This finer experiment finds useful screening much earlier on the measured
workloads, while also retaining cases that are slower even at 65,536.

The data support **4,096 as a candidate small-input boundary for these workloads**,
not a universal rule. A size-only policy cannot guarantee a speedup for every
probability distribution. No cutoff has been implemented.

## Protocol

- Same i9-12900K host and PyTorch 2.11.0+cu130 / Python 3.10.12 environment as the [whole-input CPU comparison](SCREENING_CPU_2026-09-24.md). CPU tensors, float32, RMA/Dice, smooth=0, pruning_prob=0.5, max_score. No Triton execution.
- D: **256, 512, 1,024, 2,048, 4,096, 6,144, 8,192, 12,288, 16,384, 24,576, 32,768, 49,152, 65,536**. D is spatial elements per class, not B×C×D.
- **32 synthetic profiles**: (B,C) = (1,1), (1,3), (1,16), (4,3); sparse, uniform, dense-candidate, all-pruned distributions; seeds 3401 and 7819. C=1 uses multilabel; other shapes use multiclass. Independent multiclass stress probabilities need not sum to one; these are not additional datasets.
- **8 real-cache source inputs**: VOC/Cityscapes/ADE20K cached rows 0 and 49 each; KiTS row 0 of folds 0 and 4. Selection was fixed before timing, independent of outcomes. Cache revision: `1884f0766268cdc62730e696f24dcc913d551b35`.
- For each real input and each D, select D unique, evenly spaced spatial positions, shared across channels. Preserve original probabilities exactly: no interpolation or renormalization. KiTS retains foreground channel 1 only. These are **derived workloads**, not full images, new full-cohort results, or ground-truth Dice/IoU evaluations.
- 1/4/8 CPU threads; **three rounds** with rotated thread-block order and shuffled workload order (shuffle seeds 12345–12347). Each block has two warm-ups per method and nine measured calls per method in alternating order. Timing jobs are serial.
- Measure public `RankSEG.predict` on already-prepared tensors. Exclude extraction, routing copies, loading, hashing, counts and correctness checks. Consequently, these times should not be directly substituted for the earlier whole-input runner timings.
- Per configuration, report the median of the three per-round medians for each method and their ratio. Preserve every individual timing. No runtime autotuning or production dispatch override.
- **Win** means every round has off/on > 1.05; **loss** means every round < 0.95; otherwise **mixed/near**. This predeclared reporting margin is not a confidence interval or an inference threshold. A mixed/near result is not necessarily slower.

Total: **1,560 configurations**, **4,680 timing blocks**, **84,240 timed predictions**.

## Results by input size

Real-derived rows have 8 profiles × 3 thread settings = 24 configurations per D.
Synthetic rows have 32 profiles × 3 thread settings = 96 configurations per D.

| D | Real win / mixed / loss | Real speedup range | Synthetic win / mixed / loss |
| ---: | ---: | ---: | ---: |
| 256 | 18 / 3 / 3 | 0.867–3.326× | 36 / 15 / 45 |
| 512 | 15 / 8 / 1 | 0.937–3.388× | 46 / 14 / 36 |
| 1,024 | 24 / 0 / 0 | 1.102–4.185× | 49 / 11 / 36 |
| 2,048 | 19 / 5 / 0 | 1.032–6.204× | 60 / 11 / 25 |
| 4,096 | 24 / 0 / 0 | 1.159–9.582× | 72 / 0 / 24 |
| 6,144 | 24 / 0 / 0 | 1.241–11.425× | 72 / 6 / 18 |
| 8,192 | 24 / 0 / 0 | 1.226–15.962× | 72 / 5 / 19 |
| 12,288 | 24 / 0 / 0 | 1.288–18.292× | 70 / 8 / 18 |
| 16,384 | 24 / 0 / 0 | 1.305–22.543× | 71 / 9 / 16 |
| 24,576 | 24 / 0 / 0 | 1.451–39.330× | 72 / 17 / 7 |
| 32,768 | 24 / 0 / 0 | 1.537–32.745× | 72 / 18 / 6 |
| 49,152 | 24 / 0 / 0 | 1.741–46.092× | 76 / 14 / 6 |
| 65,536 | 24 / 0 / 0 | 1.739–55.566× | 76 / 14 / 6 |

Overall: real-derived **292 wins, 16 mixed/near, 4 losses**; synthetic **844 wins,
142 mixed/near, 262 losses**. No losing configurations were removed from these
counts. Large ratios on all-pruned inputs reflect the existing early exit instead
of sorting an inactive class; they are not evidence of pixel-screening work.

### Real-derived workload differences

The following is the smallest tested D for which **every remaining tested size**,
both source inputs and all three thread settings meet the >1.05×-in-every-round
criterion. This describes only the discrete tested grid through 65,536; it is
not a claim about untested sizes or CPUs.

| Source | C | First tested sustained-win D | Speedup at D=4,096, across inputs/threads |
| --- | ---: | ---: | ---: |
| VOC | 21 | 1,024 | 1.159–2.315× |
| Cityscapes | 19 | 4,096 | 1.312–1.888× |
| ADE20K | 150 | 256 | 1.866–3.181× |
| KiTS foreground | 1 | 4,096 | 1.500–9.582× |

All real-derived configurations with D≥4,096 are wins, and none uses a
candidate-budget fallback. At D=4,096, the two VOC inputs have 2/21 and 4/21
active classes, Cityscapes 6/19 and 6/19, ADE20K 15/150 and 6/150, and KiTS 1/1
and 0/1. Whole-class pruning is therefore an important part of these workloads.
In particular, the second KiTS sample is entirely class-pruned at this size;
its large speedup should not be attributed to sorting a smaller candidate set.

Small-size losses remain visible. For example, VOC row 0, D=256, one thread:
**0.305732 → 0.352538 ms**, **0.867×**, an extra **46.806 µs**. The corresponding
four/eight-thread cases also lose. Cityscapes row 49, D=512, eight threads loses
**30.581 µs** (0.937×).

### Synthetic distribution, batch and channel effects

Using the same sustained-win definition across both seeds and all thread settings:

| B, C | Sparse | Uniform random | Dense candidates | All pruned |
| --- | ---: | ---: | --- | ---: |
| 1, 1 | 2,048 | 4,096 | No sustained-win suffix | 256 |
| 1, 3 | 2,048 | 4,096 | No sustained-win suffix | 256 |
| 1, 16 | 256 | 16,384 | No sustained-win suffix | 256 |
| 4, 3 | 256 | 24,576 | No sustained-win suffix | 256 |

This stricter 5% reporting margin should not be mistaken for the point where
screening becomes faster at all: **every measured round of every non-dense
synthetic configuration with D≥4,096 is faster than off**. Three configurations
at D=12,288 or 16,384 fall into mixed/near because one round improves by only
2.1–4.9%, below the 1.05× margin; none of those three actually regresses.
The data also demonstrate why “first observed win” and “sustained material win”
are different and why a single crossing should not be assumed monotonic.

Dense candidates are deliberately 0.49 everywhere except one 0.51 entry per
class, causing candidate-budget fallback. They remain in the results and do
not support a universal size-only speedup guarantee.

Concrete single-channel, four-thread examples (seed 3401):

| Distribution | D | Off ms | On ms | Speedup | Extra time when slower |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform | 2,048 | 0.172293 | 0.178191 | 0.967× | 5.898 µs |
| Uniform | 4,096 | 0.259584 | 0.200644 | 1.294× | — |
| Uniform | 65,536 | 3.435432 | 0.792058 | 4.337× | — |
| Dense candidates | 4,096 | 0.130999 | 0.195889 | 0.669× | 64.890 µs |
| Dense candidates | 65,536 | 0.778036 | 0.887369 | 0.877× | 109.333 µs |

The last case is about **14.1% more time**, despite D=65,536. Falling back to
full sort cannot recover the screening work already performed.

## Correctness and audit

- Both paths checked for every configuration: **3,120/3,120** independent exhaustive float64 binary-objective checks passed the unchanged four-float32-eps budget. Maximum regret: **0.585900 eps**. Reference optima are reused only for exactly matching input recipes/hashes across thread settings.
- Input hashes match across all rounds and thread counts; no probabilities were modified.
- Per-method outputs are bitwise repeatable across all three rounds at fixed thread count. This does **not** mean on/off outputs match: **141 configurations** have differences, with a maximum differing-output-element fraction of **0.048828%**.
- Source hashes are unchanged. Post-run checks verify configuration coverage/uniqueness, three rounds, nine positive finite timings per method/round, exact timing medians, objective status consistency and recomputed summaries.
- Full mini-benchmark suite: **175 passed, 16 skipped**, including **43 new calibration tests**. RankSEG core code and default remain unchanged.
- The objective budget and empirical timing evidence are regression checks, not universal mathematical or performance guarantees. No full-image Dice/IoU claims are made for the spatially subsampled inputs.

## Implications for `auto`

1. **Do not adopt 65,536 as an established CPU crossover.** It would unnecessarily exclude many measured wins and still would not eliminate dense-candidate losses.
2. **4,096 has measured support as a candidate boundary** for these real-derived inputs and non-dense synthetic workloads, with the limits above. It is not a replacement magic number or a guarantee on every distribution/hardware combination.
3. An automatic policy must state its tradeoff: optimizing the common measured workloads may accept occasional dense-candidate regressions. A policy that promises never to be slower is not supported by these data.
4. Preserve the user override and the existing all-pruned shortcut. Do not add a costly extra probability scan merely to predict whether screening will help without measuring that overhead too.
5. CUDA/Triton and CUDA/PyTorch require their own evidence. These CPU measurements do not calibrate their cutoffs.

This experiment **does not implement `auto`, change its proposed default, or
silently select a new production threshold**.

## Reproduction

From `rankseg-benchmark`, with the existing pinned caches and sibling source checkouts:

```bash
../env/bin/python scripts/calibrate_screening_cpu.py \
  --output-dir artifacts/screening-cpu-calibration-replay
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ../env/bin/python -m pytest -q
```

Outputs refuse overwrites. Local raw artifacts, ignored rather than committed:
`artifacts/screening-cpu-calibration-2026-09-24/results.json` and `timings.jsonl`.
The result file records source/cache hashes, exact profiles, individual timing
samples, output hashes, counts and objective diagnostics. No checkpoints, new
inference or package/environment changes were required.
