# RankSEG Benchmark

**Same probabilities. No retraining.** Compare [RankSEG](https://github.com/rankseg/rankseg)
with argmax using three complementary benchmark suites.

| Suite | What it evaluates | Start here |
| :--- | :--- | :--- |
| **Quick** | Hosted probabilities: VOC, Cityscapes, ADE20K, and KiTS | [Guide and results](docs/quick/README.md) |
| **MONAI** | Model Zoo Bundles evaluated on their matching datasets | [Results and protocol](docs/monai/MATCHED_MODEL_ZOO_RESULTS_2026-10-09.md) |
| **nnU-Net** | 16 datasets and 2,181 cases, with OOF or independent-test evaluation | [Full study and evidence](docs/nnunet/README.md) |

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

## MONAI: matched Model Zoo Bundle evaluation

These are **RankSEG-run comparisons**, not results published or endorsed by
MONAI. Released Model Zoo checkpoints are evaluated on their matching labeled
datasets. Argmax and RMA with safe screening decode the same complete 3D
probability volumes; model inference is excluded from decoder timing.

| Bundle / cohort | Cases | Argmax Dice | RMA Dice | Δ Dice (pp) | Decoder ms: Argmax → RMA |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Prostate MRI Anatomy / Prostate158 independent test, reader 1 | 19 | 81.753 | 81.917 | +0.165 | 0.184 → 1.108 |
| Same cohort, reader 2 | 19 | 80.033 | 80.312 | +0.279 | 0.184 → 1.108 |
| Spleen CT / MSD Task09 checkpoint-selection validation | 9 | 95.825 | 95.840 | +0.015 | 0.225 → 1.117 |

The Spleen cases were used for checkpoint selection and are **not** an
independent test. [IoU, case-separation audit, and reproduction steps](docs/monai/MATCHED_MODEL_ZOO_RESULTS_2026-10-09.md).

## nnU-Net: systematic medical benchmark

The Full-16 rerun with **RankSEG 0.0.7 and safe screening enabled** retains all
**16 datasets and 2,181 cases**, including three dataset-level regressions.
Dataset-macro Dice was **83.63646% → 83.84726%** (**+0.21080 pp**); dataset-macro
IoU improved by **+0.19943 pp**. See the [0.0.7 per-dataset results and timing
caveats](docs/nnunet/FULL16_RANKSEG_0_0_7_SCREENED_2026-10-09.md).

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

## Reproducibility and organization

- The three suites live under `rankseg_benchmark/{quick,monai,nnunet}`. Evaluation
  rules remain suite-specific; do not pool their different scoring units.
- An exploratory [TIAToolbox comparison](rankseg_benchmark/tiatoolbox/README.md)
  has its own code, case-level evidence, and [generated results table](rankseg_benchmark/tiatoolbox/RESULTS.md).
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
