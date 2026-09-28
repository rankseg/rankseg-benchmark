# RankSEG Benchmark

**Same probabilities. No retraining.** Compare [RankSEG](https://github.com/rankseg/rankseg)
with argmax using three complementary benchmark suites.

| Suite | What it evaluates | Start here |
| :--- | :--- | :--- |
| **Quick** | Hosted probabilities: VOC, Cityscapes, ADE20K, and KiTS | [Guide and results](docs/quick/README.md) |
| **MONAI** | Frozen BTCV Swin UNETR on external MSD Pancreas/Spleen cases | [Checkpoint, protocol, and reproduction](docs/monai/README.md) |
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

## MONAI: external 3D evaluation

Generate probabilities once from a pinned MONAI checkpoint, then compare decoders
on the same complete resampled 3D volumes:

```bash
python -m pip install -e ".[monai]"
rankseg-bench monai cache --list
# After generating the cache using the MONAI guide:
rankseg-bench monai evaluate \
  --dataset monai_btcv_swin_v058_msd_pancreas \
  --artifact-dir ./artifacts/monai_btcv_swin_v058_msd_pancreas \
  --device cuda --batch-size 1 --warmup 1
```

| External cohort | Volumes | Argmax Dice (%) | RankSEG Dice (%) | Δ (pp) |
| :--- | ---: | ---: | ---: | ---: |
| MSD Pancreas | 20 | 50.30 | 54.87 | +4.57 |
| MSD Spleen | 9 | 94.79 | 94.75 | −0.04 |

These are the recorded **0.0.6** foreground results. The Spleen decline is
retained: improvement is not guaranteed. The declared checkpoint training
datasets exclude these external evaluation datasets; this is a dataset-level
provenance check, not an independent patient-identity audit.
[Data preparation, IoU, timing, and provenance](docs/monai/README.md).

## nnU-Net: systematic medical benchmark

The Full-16 study retains **all 16 datasets and 2,181 cases**, including three
dataset-level regressions. Dataset-macro Dice was **83.63646% → 83.84318%**
(**+0.20671 pp**); the median dataset gain was **+0.00979 pp**.

Install the optional evaluation dependencies and verify the compact published
evidence without downloading images, checkpoints, or probability arrays:

```bash
python -m pip install -e ".[nnunet]"
rankseg-bench nnunet verify-evidence evidence/nnunet
```

The study's historical versions, splits, official postprocessing, and recorded
results are preserved. [Per-dataset results and limitations](docs/nnunet/README.md)
· [Fixed protocol](docs/nnunet/BENCHMARK_PROTOCOL.md)
· [Generated Full-16 report](evidence/nnunet/full16/RESULTS.md).

## Reproducibility and organization

- The three suites live under `rankseg_benchmark/{quick,monai,nnunet}`. Evaluation
  rules remain suite-specific; do not pool their different scoring units.
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
