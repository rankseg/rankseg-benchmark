# RankSEG Benchmark

**Same probabilities. No retraining.** Compare [RankSEG](https://github.com/rankseg/rankseg)
with standard segmentation decoding across four benchmark areas: hosted
probabilities, MONAI Bundles, nnU-Net, and TIAToolbox.

| Benchmark | What it evaluates | Start here |
| :--- | :--- | :--- |
| **Quick** | Hosted probabilities: VOC, Cityscapes, ADE20K, and KiTS | [Guide and results](docs/quick/README.md) |
| **MONAI** | Complete Model Zoo Bundle inference on pancreas and prostate test cohorts | [Results and reproduction](#monai-complete-model-zoo-bundle-evaluation) |
| **nnU-Net** | 16 datasets and 2,181 cases, with OOF or independent-test evaluation | [Results and evidence](#nnu-net-systematic-medical-benchmark) |
| **TIAToolbox** | Pretrained nucleus and tissue segmentation models on CoNSeP, Kumar, and BCSS | [Results and reproduction](#tiatoolbox-nucleus-and-tissue-segmentation) |

## Quick start

Requires **Python 3.10+**. From a fresh environment:

```bash
git clone https://github.com/rankseg/rankseg-benchmark.git
cd rankseg-benchmark
python -m pip install -e . "rankseg==0.0.7"

# Five-image smoke run; downloads pre-computed probabilities.
rankseg-bench quick --dataset pascal_voc --limit 5 --device cpu
```

The report includes Dice, IoU, and decoder latency. No checkpoint download or
model inference is needed for Quick. The original command syntax remains valid:
`rankseg-bench --dataset pascal_voc --limit 5 --device cpu`.

## Quick: hosted probabilities

Run a short comparison or evaluate all hosted inputs:

```bash
rankseg-bench quick --list-datasets
rankseg-bench quick --dataset ade20k --metric dice --per-class --json results_ade20k.json
```

Recorded **0.0.6** results: Dice gains of **+0.44 pp** on VOC, **+0.61 pp** on
Cityscapes, **+0.93 pp** on ADE20K, and **+2.37 pp** on KiTS. These are historical
results, not a newly rerun 0.0.7 benchmark. KiTS uses slice → case → fold averaging,
not whole-volume Dice. [Full metrics and protocol](docs/quick/REFERENCE_0.0.6.md).

## MONAI: complete Model Zoo Bundle evaluation

Recorded runs use **RankSEG 0.0.7** with **MONAI 1.4.0**. RankSEG uses
`metric="dice"`, `solver="RMA"`, `output_mode="multiclass"`, and
`safe_screening=True`. Both decoders receive the same probabilities from
frozen Model Zoo checkpoints. The Bundle preprocessing, sliding-window
inference, and remaining postprocessing steps follow their official order.
Predictions are evaluated on the original image grid.

Foreground macro Dice is computed per volume by averaging its foreground
classes, then averaged equally across volumes. Scores below are percentages;
paired gains and 95% bootstrap intervals are in percentage points (pp),
computed from unrounded per-case scores.

| Bundle / cohort | Cases | Argmax Dice (%) | RankSEG Dice (%) | Δ Dice (pp) | Paired 95% CI (pp) | Cases improved |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| [DiNTS Pancreas / MSD Task07 `imagesTs`, post-hoc expert labels](docs/monai/NATIVE_PANCREAS_BUNDLE_007_2026-10-10.md) | 49 | 55.474 | 58.018 | +2.545 | +1.649 to +3.534 | 38/49 |
| [DiNTS Pancreas / external CPTAC-PDA](docs/monai/NATIVE_CPTAC_BUNDLE_007_2026-10-11.md) | 42 | 55.521 | 59.193 | +3.672 | +2.379 to +5.223 | 37/42 |
| [Prostate MRI Anatomy / Prostate158 independent test, reader 1](docs/monai/NATIVE_PROSTATE_BUNDLE_007_2026-10-10.md) | 19 | 81.687 | 81.882 | +0.195 | +0.083 to +0.334 | 16/19 |
| Prostate MRI Anatomy / same test images, reader 2 | 19 | 79.630 | 79.915 | +0.284 | +0.183 to +0.413 | 19/19 |

The two prostate readers annotated the **same 19 images**. Pancreas cohort
gains were driven mainly by tumor Dice: **+4.722 pp** on MSD and **+6.415 pp**
on CPTAC-PDA. The MSD test IDs are separate from its training IDs, and CPTAC
case IDs are distinct from MSD; the released checkpoint's complete training
and selection history cannot be independently audited. These are
comparisons conducted by the RankSEG team using released MONAI Bundles.

The linked cohort reports document data provenance, evaluation details,
per-class scores, decoder timing, and reproduction commands. Runners:
[DiNTS pancreas and CPTAC](rankseg_benchmark/monai/native_pancreas.py) ·
[Prostate](rankseg_benchmark/monai/native_prostate.py).

[Earlier matched-checkpoint experiments](docs/monai/MATCHED_MODEL_ZOO_RESULTS_2026-10-09.md)
remain available separately, including the nine Spleen validation cases used
for checkpoint selection.

## nnU-Net: systematic medical benchmark

The Full-16 rerun with **RankSEG 0.0.7 and safe screening enabled** retains all
**16 datasets and 2,181 cases**, including three dataset-level regressions.
Each row first averages Dice over the valid cases for each foreground
class/region, then averages those class/region means. The final macro-average
weights the 16 datasets equally. Counts follow the study's evaluation units,
including frames for ACDC and cell tracking.

| Dataset | Cases | Argmax Dice (%) | RankSEG Dice (%) | Δ Dice (pp) |
| :--- | ---: | ---: | ---: | ---: |
| CHAOS CT / TotalSegmentator v2 | 20 | 97.086 | 97.092 | +0.006 |
| MSD BrainTumour | 484 | 84.572 | 84.580 | +0.007 |
| MSD Heart | 20 | 93.294 | 93.296 | +0.002 |
| MSD Liver | 131 | 80.999 | 81.855 | +0.856 |
| MSD Hippocampus | 260 | 88.908 | 88.913 | +0.004 |
| MSD Prostate | 32 | 75.957 | 75.979 | +0.022 |
| MSD Lung | 63 | 72.288 | 72.752 | +0.464 |
| MSD Pancreas | 281 | 68.396 | 69.571 | +1.175 |
| MSD HepaticVessel | 303 | 68.631 | 69.164 | +0.533 |
| MSD Spleen | 41 | 97.237 | 97.244 | +0.007 |
| MSD Colon | 126 | 48.918 | 49.159 | +0.241 |
| PROMISE12 independent test | 30 | 91.936 | 91.882 | -0.054 |
| ACDC | 200 | 92.454 | 92.380 | -0.074 |
| CHAOS MRI | 60 | 92.044 | 92.220 | +0.176 |
| SegTHOR | 40 | 91.520 | 91.518 | -0.002 |
| Fluo-C3DH-A549 (manual + simulated) | 90 | 93.944 | 93.953 | +0.009 |
| **Dataset macro-average (16 datasets)** | **2,181** | **83.636** | **83.847** | **+0.211** |

Deltas are computed before rounding. Thirteen datasets improve and three
decline; dataset-macro IoU improves by **+0.19943 pp**. See the
[0.0.7 per-class results and timing caveats](docs/nnunet/FULL16_RANKSEG_0_0_7_SCREENED_2026-10-09.md).

The checksummed public evidence below preserves the **historical 0.0.5 run
without screening** (+0.20671 Dice pp). It has not been silently replaced by
the 0.0.7 rerun.

Install the optional evaluation dependencies and verify the compact published
evidence without downloading images, checkpoints, or probability arrays:

```bash
python -m pip install -e ".[nnunet]"
rankseg-bench nnunet verify-evidence evidence/nnunet
```

The study's historical versions, splits, official postprocessing, and recorded
results are preserved. [Study design and limitations](docs/nnunet/README.md)
· [Fixed protocol](docs/nnunet/BENCHMARK_PROTOCOL.md)
· [Historical generated macro report](evidence/nnunet/full16/RESULTS.md).

A separate [official TotalSegmentator v2 / AMOS22 pilot](docs/nnunet/AMOS22_TOTAL_SEGMENTATOR_V2_PILOT.md)
tests six organs on ten prespecified CT cases with RankSEG 0.0.7. Its Dice
changes are only **+0.01 to +0.23 pp**; it is not part of the Full-16 aggregate
or a substitute for a complete AMOS22 cohort.

## TIAToolbox: nucleus and tissue segmentation

Recorded comparisons use **TIAToolbox 2.1.3** and **RankSEG 0.0.8** with
Dice/RMA, `safe_screening=True`, `pruning_prob=0.5`, and `smooth=0`. Both methods
receive the same model probabilities and use the same subsequent official
postprocessing. Scores are computed per image/ROI, then averaged equally.

| Checkpoint / cohort | Images / ROIs | Baseline Dice (%) | RankSEG Dice (%) | Δ Dice (pp) | Paired 95% CI (pp) | Images improved |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `micronet-consep` / CoNSeP | 14 | 79.659 | 80.113 | +0.454 | +0.226 to +0.662 | 11/14 |
| `micronet-consep` / Kumar | 14 | 61.263 | 64.941 | +3.678 | +2.269 to +5.206 | 14/14 |
| `hovernet_original-consep` / CoNSeP | 14 | 84.707 | 84.728 | +0.021 | -0.059 to +0.098 | 8/14 |
| `hovernet_original-kumar` / Kumar | 14 | 82.525 | 82.631 | +0.106 | -0.035 to +0.250 | 9/14 |
| `fcn_resnet50_unet-bcss` / BCSS | 45 | 59.146 | 60.306 | +1.160 | -0.375 to +3.027 | 21/45 |

The nucleus rows score binary foreground after official instance
postprocessing; BCSS scores a five-class ROI macro Dice including `Other`.
The baseline is argmax for MicroNet and FCN, and the official 0.5 foreground
threshold for HoVer-Net. Deltas and intervals use unrounded per-image scores.

The two CoNSeP rows share the same 14 images, as do the two Kumar rows.
MicroNet/Kumar is a cross-dataset evaluation. HoVer-Net and BCSS intervals
include zero; foreground Dice and instance-separation quality are separate
endpoints. The [full 2.x report](rankseg_benchmark/tiatoolbox/RESULTS_V2.md)
includes per-class BCSS scores, AJI/PQ, and multilabel diagnostics.

[Code and reproduction guide](rankseg_benchmark/tiatoolbox/README.md) ·
[2.x reproduction and migration audit](rankseg_benchmark/tiatoolbox/V2_MIGRATION.md) ·
[Historical 1.6.0 results](rankseg_benchmark/tiatoolbox/RESULTS.md).

## Reproducibility and organization

- The four benchmark areas live under
  `rankseg_benchmark/{quick,monai,nnunet,tiatoolbox}`. Each defines its own
  evaluation rules and scoring units; consult its linked protocol.
- Original commands `rankseg-monai-cache`, `rankseg-monai-demo`, and
  `rankseg-nnunet-bench` remain available. [Migration and workspace guide](docs/MIGRATION.md).
- Checkpoints, images, probability caches, and local outputs stay outside Git.
  Only code, configurations, and compact evidence are versioned.
- [Screening experiments](docs/screening/README.md) are a separate engineering
  topic, not a replacement for the segmentation-quality benchmarks. In RankSEG
  0.0.7, screening remains disabled by default.

## License

Quick/MONAI and the original benchmark code use [BSD-3-Clause](LICENSE).
Imported nnU-Net material retains [Apache-2.0](LICENSES/Apache-2.0.txt).
See [attribution and scope](NOTICE.md). Dataset and checkpoint licenses are separate.
