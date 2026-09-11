# RankSEG Benchmark

Reproducible benchmark for [RankSEG](https://github.com/rankseg/rankseg): measures
**segmentation performance (Dice / IoU) and runtime** against the `argmax` baseline
on **pre-computed probability masks** from frozen segmentation models. It also
provides a reproducible generator for pinned MONAI Model Zoo Bundles and 3D
medical volumes.

The hosted probability benchmarks need no training or segmentation-model
setup. The optional MONAI workflow adds a separate, one-time model inference
stage for users who want to generate probabilities from the pinned checkpoints.

## Quick start

```bash
pip install rankseg-benchmark

# Run RankSEG vs. argmax on ADE20K (downloads pre-computed probs from Hugging Face)
rankseg-bench --dataset ade20k --solver RMA --metric dice
```

You'll get a table like:

```
## Performance

|       | argmax | rankseg-RMA | Improvement |
|-------|--------|-------------|-------------|
| mDice |  51.21 |       52.17 |       +0.96 |
| mIoU  |  40.00 |       40.82 |       +0.82 |

## Runtime

|                 | argmax | rankseg-RMA | Overhead |
|-----------------|--------|-------------|----------|
| mean ms / img   |   1.20 |        8.40 |    +7.20 |
| median ms / img |   1.10 |        7.90 |    +6.80 |
```

## Benchmark results

Performance numbers are reported as percentages. Improvement is RankSEG minus
the argmax baseline.

### Pascal VOC

|       | argmax | rankseg-RMA | Improvement |
|-------|-------:|------------:|------------:|
| mDice |  91.02 |       91.46 |       +0.44 |
| mIoU  |  87.80 |       88.23 |       +0.43 |

### Cityscapes

|       | argmax | rankseg-RMA | Improvement |
|-------|-------:|------------:|------------:|
| mDice |  82.62 |       83.23 |       +0.61 |
| mIoU  |  75.67 |       76.19 |       +0.52 |

### ADE20K

|       | argmax | rankseg-RMA | Improvement |
|-------|-------:|------------:|------------:|
| mDice |  63.98 |       64.92 |       +0.93 |
| mIoU  |  56.95 |       57.67 |       +0.72 |

### KiTS

|       |   argmax |   rankseg-RMA |   Improvement |
|-------|----------|---------------|---------------|
| mDice |    61.16 |         63.53 |          2.37 |
| mIoU  |    54.19 |         56.21 |          2.02 |

### MONAI BTCV Swin UNETR → unseen MSD volumes

These are foreground-only, per-volume means from a frozen
`swin_unetr_btcv_segmentation==0.5.8` checkpoint. The Pancreas cohort contains
20 fixed MSD Task07 volumes; the Spleen cohort contains nine fixed MSD Task09
volumes. RankSEG uses RMA with the Dice objective.

| External test cohort | Volumes | Metric | argmax | rankseg-RMA | Improvement |
|----------------------|--------:|--------|-------:|------------:|------------:|
| MSD Pancreas | 20 | Dice | 50.30 | 54.87 | +4.57 |
| MSD Pancreas | 20 | IoU  | 35.29 | 39.19 | +3.90 |
| MSD Spleen   |  9 | Dice | 94.79 | 94.75 | -0.04 |
| MSD Spleen   |  9 | IoU  | 90.12 | 90.06 | -0.07 |

Decoder-only CUDA latency was measured with one complete resampled 3D volume
per call on an NVIDIA RTX 3090, after one untimed warmup invocation. Model
inference, preprocessing, I/O, and transfers are excluded.

| External test cohort | argmax mean | RankSEG mean | Overhead | argmax median | RankSEG median |
|----------------------|------------:|-------------:|---------:|--------------:|---------------:|
| MSD Pancreas | 0.28 ms | 3.71 ms | +3.42 ms | 0.27 ms | 3.60 ms |
| MSD Spleen   | 0.33 ms | 3.93 ms | +3.60 ms | 0.27 ms | 3.68 ms |

The IoU-objective runs produce the same rounded accuracy here; their RankSEG
mean latencies were 3.30 ms/volume for Pancreas and 3.47 ms/volume for Spleen.
The small Spleen decline is reported intentionally: post-processing gains are
not guaranteed on every already-well-calibrated cohort.

For the release regression, the published RankSEG 0.0.5 wheel and the 0.0.6
candidate were run on the same cached tensors with the Dice objective. Their
Dice and IoU results are exactly equal on both cohorts. Mean RankSEG latency
decreased from 4.22 to 3.71 ms/volume on Pancreas and from 4.48 to 3.93
ms/volume on Spleen (about 12.2% in each run). Latency is hardware- and
run-dependent; the accuracy equality is the compatibility check.

## Available datasets

| Name | Classes | Spatial | Mode | Source model |
|------|--------:|--------:|------|--------------|
| `pascal_voc` | 21 | 2D | multiclass | Pre-computed probs |
| `ade20k` | 150 | 2D | multiclass | Pre-computed probs |
| `cityscapes` | 19 | 2D | multiclass | Pre-computed probs |
| `kits` | 2 | 2D slices | foreground | Pre-computed probs |
| `monai_btcv_swin_v058_msd_pancreas` | 2 | 3D | foreground | MONAI BTCV Swin UNETR 0.5.8 → unseen MSD Pancreas |
| `monai_btcv_swin_v058_msd_spleen` | 2 | 3D | foreground | MONAI BTCV Swin UNETR 0.5.8 → unseen MSD Spleen |

List datasets at runtime:

```bash
rankseg-bench --list-datasets
```

## Common flags

```bash
# Quick smoke run (5 samples) on CPU
rankseg-bench --dataset pascal_voc --limit 5 --device cpu

# IoU metric instead of Dice
rankseg-bench --dataset pascal_voc --metric iou --solver RMA

# Per-class Dice/IoU breakdown (optional — large classes count = long table)
rankseg-bench --dataset ade20k --solver RMA --per-class

# Save full results to JSON (includes per-class arrays)
rankseg-bench --dataset pascal_voc --solver RMA --json results_voc.json

# Test a local RankSEG checkout instead of the installed package
rankseg-bench --dataset pascal_voc --solver RMA --rankseg-path ~/code/rankseg

# Run a generated local MONAI volume cache
rankseg-bench \
    --dataset monai_btcv_swin_v058_msd_pancreas \
    --artifact-dir ./artifacts/monai_btcv_swin_v058_msd_pancreas \
    --solver RMA --metric dice --batch-size 1
```

You can also set `RANKSEG_PATH=~/code/rankseg`. The `--rankseg-path` flag takes
precedence when both are provided. If neither is set, the benchmark imports the
normal installed `rankseg` package from the active Python environment.

## MONAI datasets and checkpoints

MONAI support is split into two stages. Model inference runs once to create a
canonical probability cache; the normal benchmark then compares argmax and
RankSEG on exactly the same tensors. Model inference, preprocessing, disk I/O,
and host/device transfers are not included in decoder timing.

Install the optional generator dependencies and list the pinned targets:

```bash
pip install "rankseg-benchmark[monai]"
rankseg-monai-cache --list
```

Provide an extracted Medical Segmentation Decathlon task directory containing
`imagesTr/` and `labelsTr/`. The generator can download the pinned Bundle and
checkpoint from the MONAI Model Zoo:

```bash
# One-case external Pancreas smoke generation
rankseg-monai-cache \
    --benchmark monai_btcv_swin_v058_msd_pancreas \
    --dataset-root /data/Task07_Pancreas \
    --bundle-dir ./monai_bundles \
    --output-dir ./artifacts/monai_btcv_swin_v058_msd_pancreas \
    --download-bundle --device cuda --limit 1

# Evaluate that exact artifact
rankseg-bench \
    --dataset monai_btcv_swin_v058_msd_pancreas \
    --artifact-dir ./artifacts/monai_btcv_swin_v058_msd_pancreas \
    --device cuda --batch-size 1
```

Remove `--limit 1` to generate the fixed 20-case Pancreas external test
subset. For Spleen, use `monai_btcv_swin_v058_msd_spleen` with an extracted
`Task09_Spleen` directory; its external test subset contains nine fixed cases.

Both targets deliberately use the `swin_unetr_btcv_segmentation==0.5.8`
checkpoint instead of a checkpoint trained on the same MSD task. The official
[Bundle documentation](https://github.com/Project-MONAI/model-zoo/blob/dev/models/swin_unetr_btcv_segmentation/docs/README.md)
identifies BTCV as its supervised task, while MONAI's
[pretraining data loader](https://github.com/Project-MONAI/research-contributions/blob/main/SwinUNETR/Pretrain/utils/data_utils.py)
lists LUNA16, TCIA COVID-19, HNSCC, TCIA Colonography, and LIDC-IDRI as the
self-supervised sources. MSD Task07 and Task09 are not listed among those
sources. Under this official, declared-dataset provenance, the fixed MSD
cohorts are external tests: their cases are not used for checkpoint training
or selection, post-processing parameter selection, or threshold fitting.

Every built-in target is validated to require
`checkpoint_exposure=none_by_declared_dataset_provenance`, false test-case
usage flags, and a benchmark dataset ID distinct from every declared
supervised and self-supervised checkpoint source ID. Those declarations, the
fixed case list, and the pinned checkpoint hashes are copied into every
generated manifest. This is an auditable dataset-level safeguard based on the
upstream project's published provenance; it does not claim an independent
patient-identity audit beyond that metadata.

The 14-channel BTCV softmax is collapsed into a mathematically equivalent
binary distribution `[1 - p(target), p(target)]` before caching. Spleen uses
model channel 1 and MSD label 1. Pancreas uses model channel 11 and evaluates
the whole organ, so MSD pancreas-parenchyma and tumor labels (1 and 2) are
merged into one foreground target. No threshold is fitted on the test cases.

The built-in manifests pin:

- Bundle name and version;
- official model-file checksum, plus generated SHA-256 provenance;
- exact external-test case selection and checkpoint-exposure declaration;
- orientation, spacing, intensity scaling, sliding-window settings, and activation;
- software versions and source image/label hashes.

Artifacts use float32 probabilities. Each case is stored independently and its
SHA-256 is verified when the benchmark reads it. Probabilities and labels are
aligned in the deterministic resampled model space used by the official MONAI
RankSEG tutorial. RankSEG processes each 3D volume as one optimization unit; it
does not silently convert volumes into independent 2D slices.

The generator follows the Bundle's RAS orientation, 1.5×1.5×2.0 mm spacing,
intensity window, 96³ sliding window, 0.5 overlap, and AMP inference. It uses
`sw_batch_size=1` instead of the deployment config's 4 to fit 24 GiB GPUs and
stores the collapsed probabilities as float32. Sliding-window batch size
affects throughput and small floating-point reduction details, not decoder
timing, which starts only after the probability artifact is on the selected
device.

The generator does not download or redistribute the medical datasets. Users
must obtain them under their original licenses and pass the extracted task
directory explicitly.

To reproduce the transparent light- and dark-theme medical visualizations used
by the RankSEG README from the complete Pancreas artifact cache:

```bash
pip install "rankseg-benchmark[monai,demo]"
rankseg-monai-demo \
    --artifact-dir ./artifacts/monai_btcv_swin_v058_msd_pancreas \
    --dataset-root /data/Task07_Pancreas \
    --output ./monai_pancreas_rankseg.png \
    --theme light --rankseg-path /path/to/rankseg --device cuda

rankseg-monai-demo \
    --artifact-dir ./artifacts/monai_btcv_swin_v058_msd_pancreas \
    --dataset-root /data/Task07_Pancreas \
    --output ./monai_pancreas_rankseg_dark.png \
    --theme dark --rankseg-path /path/to/rankseg --device cuda
```

The displayed slice is selected after aggregate evaluation using a fixed rule:
among axial slices with at least 400 foreground pixels, maximize corrected
minus introduced error pixels, with deterministic tie-breakers. Slice
selection is presentation-only and cannot change checkpoint selection,
post-processing parameters, or the reported 20-volume result.

## Bring your own model

The bundled pre-computed probability dumps are loaded from
[`ZixunWang/rankseg-benchmark`](https://huggingface.co/datasets/ZixunWang/rankseg-benchmark),
using its `pascal_voc`, `ade20k`, and `cityscapes` data directories. If you
want to benchmark **your own** model's probabilities, generate a dump with the
provided script and point the registry at your repo:

```bash
python scripts/generate_probs.py \
    --model deeplabv3plus_voc \
    --images-dir /path/to/voc/JPEGImages \
    --labels-dir /path/to/voc/SegmentationClass \
    --split-file /path/to/voc/ImageSets/Segmentation/val.txt \
    --num-classes 21 \
    --output ./probs_voc_mine
```

Then either upload the directory to a Hugging Face dataset repo and add an
entry to `rankseg_benchmark/datasets.py`, or call the benchmark internals
directly:

```python
from rankseg_benchmark.datasets import DatasetSpec
from rankseg_benchmark.runner import run_benchmark, format_report

spec = DatasetSpec(
    name="voc-mine",
    hf_repo="your-username/your-probs-repo",
    hf_split="validation",
    hf_data_dir=None,
    num_classes=21,
    output_mode="multiclass",
    ignore_index=255,
    spatial_dims=2,
)
baseline, rs = run_benchmark(spec, solver="RMA")
print(format_report(baseline, rs, per_class=True))
```

## What gets measured

- **Performance (`mDice` / `mIoU`)**: the per-image metric used in
  the [RankSEG-RMA paper](https://openreview.net/pdf?id=4tRMm1JJhw). For each
  image we compute per-class Dice/IoU, average over the classes that are
  actually active in that image, then average across the dataset. This matches
  the [reference implementation](https://github.com/ZixunWang/RankSEG-RMA/blob/master/exp/metrics/accuracy_metric.py).
- **KiTS**: slice predictions are grouped by case and evaluated with the
  medical benchmark logic from `exp/test.py::test_medical` in RankSEG-RMA. The
  reported mDice/mIoU aligns with `mDiceI`/`mIoUI` and scores foreground only.
  The stored probabilities remain two-channel background/foreground softmax
  outputs for the argmax baseline, but every RankSEG solver receives only the
  foreground channel in multilabel mode. Predictions are converted back to
  binary class labels for evaluation.
- **MONAI Pancreas**: one complete resampled 3D volume is one evaluation unit.
  The unseen BTCV checkpoint's pancreas probability is evaluated against the
  merged MSD pancreas/tumor organ foreground; metrics exclude background.
- **MONAI Spleen**: the unseen BTCV checkpoint's spleen probability and the MSD
  spleen foreground are evaluated as a binary task; metrics exclude background.
- **Runtime**: per-evaluation-unit latency for `RankSEG.predict` vs.
  `torch.argmax`, with CUDA synchronization and a configurable warmup (default
  3 samples). The unit is image, slice, or complete volume as shown in the
  report.
  Warmup inputs remain part of the measured dataset; warmup never removes
  samples from the reported metrics.
- **Per-class gain breakdown** (with `--per-class`): for each class, its
  Dice/IoU averaged over the images where the class is active. Useful for
  showing RankSEG's larger lift on rare classes.

## License

BSD 3-Clause.
