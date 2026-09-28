# Auto-default acceptance — representative nnU-Net volumes

**Historical experiment:** the subsequent release decision keeps
`safe_screening=False` as the default. Auto remains opt-in. These measurements
describe the preserved auto-default snapshot; no recorded results or artifacts
have been rewritten to describe the new default.

Completed on 2026-09-25. This is a **41-volume subset**, not a repeat of the
778-volume complete-cohort acceptance. The local RankSEG candidate now defaults
to `safe_screening="auto"`; no new package has been published by this run.

## Scope and protocol

- Two lexicographically first case IDs from each of five held-out folds, for
  Liver, Pancreas, HepaticVessel and Lung, plus the existing `liver_43` index
  regression: **11 / 10 / 10 / 10 volumes**. Selection does not use results.
- Reused verified OOF ensemble probability caches and fold provenance. Each
  case's probabilities originate from models whose training fold excluded it.
  No inference, checkpoint download, retraining or cache replacement was done.
- RTX 3090, Python 3.10.12, Torch 2.8.0+cu128, Triton 3.4.0, float32, B=1,
  four CPU threads. RMA Dice, smooth=0, pruning=0.5, multiclass max-score.
- Auto uses screening on all selected volumes: CUDA + Triton and at least
  **1,280,000 probability entries B*C*D**. No candidate-density/count retries.
  `Off` means explicit `safe_screening=False`; optimized full sort is a separate
  auxiliary control. Historical JSON key `versus_default_full` means **Off**, not
  the new default.
- Two warm-ups and five synchronized timings per method, rotating method order.
  Latency is the mean of per-volume medians. Auxiliary dispatch controls are
  outside timing. GPU benchmarks run serially.
- Decoder timing excludes loading, transfers, compilation, numerical checks,
  metrics and connected-component postprocessing. Official postprocessing is
  applied identically to all methods for quality evaluation.
- Quality uses whole-volume foreground Dice / IoU, averaged over cases per
  label, then foreground labels, retaining the repository's undefined-empty
  convention. IoU is evaluated, not independently optimized.
- Memory is **incremental peak allocated GPU MiB**, excluding resident input
  probabilities, cached/reserved memory and CUDA context. It is not total GPU
  usage or network inference memory.

## Results on the selected volumes

Scores are percentages. A dash means incomplete coverage, **not** an omitted
failure: original Off ran out of GPU memory on six Liver volumes. Do not compare
its five completed volumes against eleven-volume averages.

| Dataset | Method | Completed | Dice | IoU | Mean ms | Mean / maximum peak MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Liver | Argmax | 11/11 | 80.2925 | 73.1934 | 7.2281 | 2363.82 / 5016.00 |
| Liver | Off | 5/11 | — | — | — | 6 OOM |
| Liver | Optimized full | 11/11 | 78.3817 | 71.4349 | 301.2919 | 8757.36 / 14212.00 |
| Liver | Auto | 11/11 | 78.3961 | 71.4594 | 11.4446 | 1545.28 / 2508.00 |
| Pancreas | Argmax | 10/10 | 52.3264 | 39.4406 | 0.6712 | 210.20 / 348.00 |
| Pancreas | Off | 10/10 | 52.5260 | 39.2645 | 76.2754 | 3368.00 / 5568.00 |
| Pancreas | Optimized full | 10/10 | 52.5263 | 39.2648 | 61.5838 | 1788.80 / 2958.00 |
| Pancreas | Auto | 10/10 | 52.5280 | 39.2668 | 2.7275 | 315.30 / 522.00 |
| HepaticVessel | Argmax | 10/10 | 71.1588 | 57.7968 | 0.3050 | 91.60 / 132.00 |
| HepaticVessel | Off | 10/10 | 72.8789 | 59.5766 | 33.8706 | 1469.80 / 2112.00 |
| HepaticVessel | Optimized full | 10/10 | 72.8811 | 59.5804 | 27.2071 | 780.40 / 1122.00 |
| HepaticVessel | Auto | 10/10 | 72.8817 | 59.5808 | 1.5881 | 137.50 / 198.00 |
| Lung | Argmax | 10/10 | 75.7985 | 65.8255 | 1.7416 | 695.00 / 1178.00 |
| Lung | Off | 10/10 | 73.5587 | 62.9646 | 219.2853 | 7645.00 / 12958.00 |
| Lung | Optimized full | 10/10 | 73.5640 | 62.9714 | 185.9722 | 4170.00 / 7068.00 |
| Lung | Auto | 10/10 | 73.5580 | 62.9635 | 6.5111 | 955.63 / 1619.75 |

RankSEG does **not** beat argmax on every dataset: on these selected Liver and
Lung volumes, Dice and IoU are lower for both optimized full sort and auto.
Pancreas Dice improves slightly but IoU decreases. These outcomes must not be
hidden or presented as screening-induced gains; this acceptance primarily tests
screening correctness, runtime and memory, not full-cohort clinical performance.

### Matched comparisons

Both methods in each row use exactly the same cases. Speedup is the ratio of
mean decoder latencies; memory reduction uses mean incremental peaks.

| Dataset | Reference | Paired volumes | Auto speedup | Peak reduction | Dice Δ (pp) | IoU Δ (pp) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Liver | Off | 5 | 30.19× | 90.62% | -0.00113 | -0.00297 |
| Liver | Optimized full | 11 | 26.33× | 82.35% | +0.01436 | +0.02444 |
| Pancreas | Off | 10 | 27.97× | 90.64% | +0.00202 | +0.00225 |
| Pancreas | Optimized full | 10 | 22.58× | 82.37% | +0.00173 | +0.00198 |
| HepaticVessel | Off | 10 | 21.33× | 90.64% | +0.00284 | +0.00424 |
| HepaticVessel | Optimized full | 10 | 17.13× | 82.38% | +0.00057 | +0.00048 |
| Lung | Off | 10 | 33.68× | 87.50% | -0.00072 | -0.00113 |
| Lung | Optimized full | 10 | 28.56× | 77.08% | -0.00596 | -0.00798 |

### Screening proportion and memory

H+L counts certified binary-positive/negative **class-probability entries**, not
unique voxels or final multiclass labels. Already-pruned classes are excluded
from the active-entry denominator. Effective sorting avoidance equals H+L here;
there are no candidate-workspace full-sort retries.

| Dataset | H+L / active entries | Class-pruned / all entries | Peak reduction vs optimized full |
| --- | ---: | ---: | ---: |
| Liver | 99.93480% | 0.00000% | 82.35% |
| Pancreas | 99.99740% | 3.29845% | 82.37% |
| HepaticVessel | 99.99443% | 0.00000% | 82.38% |
| Lung | 99.99971% | 3.97122% | 77.08% |

Screening removes most sorting work, not all input scans, scores, output masks
or assignment workspace, so its fraction is not a memory-reduction percentage.

### Large-volume regressions

- Known `liver_43` regression, shape **[1, 3, 155, 512, 512]**: auto completes
  in **5.2705 ms**, using **466.00 MiB** incremental peak, versus optimized
  full's **91.9891 ms / 2638.00 MiB**.
- Largest selected volume `liver_113`, shape **[1, 3, 836, 512, 512]**: auto
  completes in **17.8080 ms / 2508.00 MiB**, versus optimized full's
  **489.3882 ms / 14212.00 MiB**; original Off OOMs. This is the largest
  **selected** volume, not a claim to have rerun the largest full-cohort case.
- No CUDA indexing errors or auto/optimized-full OOMs occurred in this run.

## Correctness and limitations

- All **117 completed RankSEG method-volume objective checks** passed: Off 35,
  optimized full 41, auto 41. The six Off OOMs remain unverified, not passed.
- The independent oracle sorts every probability and evaluates **every prefix**
  in float64, using bounded temporary workspace. It does not reuse screening
  bounds or skip candidates. The existing budget remains **4 float32 eps**.
- Maximum regret: **0.63399 eps for auto**, 2.55845 eps across all completed
  controls. All 41 volumes have completed oracle checks; none were waived.
- Independent summary validation recomputes overlap scores from TP/FP/FN,
  timing medians and paired coverage. Argmax foreground counts match the
  previously published per-case evidence for every selected case and label.
- OOF provenance and probability/label/source hashes are recorded. Frozen
  RankSEG and harness hashes stayed unchanged throughout the run.
- Auto does not promise bitwise Off masks. Largest absolute dataset-mean Dice
  change versus optimized full is **0.01437 pp**; worst individual case-label
  decline is **0.05421 pp** (`lung_005`, label 1). Binary-objective epsilon
  bounds do not bound final multiclass or ground-truth metric changes.
- This is empirical acceptance on specified data/hardware, not a proof over
  all inputs, hardware or a guarantee that auto is never slower.

## Reproduction and evidence

Raw reports, per-dataset summaries, untimed screening counts, logs, selected case
IDs and `run_status.json` are in the ignored local directory
`outputs/screening-auto-default-2026-09-25/`. Previous complete-cohort reports
and cached probabilities were not overwritten. The frozen package is also
preserved under the mini benchmark's `artifacts/screening-auto-default-2026-09-25/snapshot/`.

From this repository, using the existing medical environment and caches:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 env/bin/python -B \
  scripts/run_auto_acceptance_subset.py \
  --rankseg-path ../rankseg-benchmark/artifacts/screening-auto-default-2026-09-25/snapshot \
  --output-dir outputs/screening-auto-default-rerun
```

The driver refuses to overwrite an existing output directory. Related complete
mini-cache, CPU and multi-scale evidence is in
[`rankseg-benchmark`'s acceptance report](../quick/SCREENING_AUTO_DEFAULT_ACCEPTANCE_2026-09-25.md).
