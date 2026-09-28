> Archived pre-migration guide and recorded 0.0.6 results; numbers are unchanged.
> For current entry points, see [the main README](../../README.md).

# RankSEG Benchmark

Compare [RankSEG](https://github.com/rankseg/rankseg) with `argmax` on the same
frozen-model probabilities. Measure **Dice, IoU, and post-processing time**—no
retraining or threshold fitting.

Use hosted probability datasets for a quick comparison, or generate a reusable
3D probability cache with a pinned MONAI checkpoint.

[Quick start](#quick-start) · [Results](#benchmark-results) ·
[Run benchmarks](#run-benchmarks) · [MONAI](#monai-datasets-and-checkpoints) ·
[Evaluation protocol](#evaluation-protocol) · [Custom models](#bring-your-own-model)

## Quick start

Requires **Python 3.10+**. Install this checkout to get all commands below;
the example pins RankSEG to the validated **0.0.6** release.

```bash
git clone https://github.com/rankseg/rankseg-benchmark.git
cd rankseg-benchmark
python -m pip install -e . "rankseg==0.0.6"

# Five-image smoke run; downloads pre-computed probabilities from Hugging Face.
rankseg-bench --dataset pascal_voc --limit 5 --device cpu
```

The report includes Dice/IoU, gains over `argmax`, and mean/median decoder
latency. Remove `--limit 5` for a full-dataset run. CUDA is selected by default
when available; use `--device cpu` or `--device cuda` to choose explicitly.

## Benchmark results

### Segmentation quality

All results below use **RankSEG-RMA with the Dice objective**. Scores are
percentages; **Δ** is RankSEG minus Argmax in percentage points (pp), computed
before rounding. Evaluation units differ by dataset; see the
[evaluation protocol](#evaluation-protocol).

| Dataset | Metric | Argmax | RankSEG-RMA | Δ (pp) |
| :--- | :--- | ---: | ---: | ---: |
| PASCAL VOC | Dice | 91.02 | 91.46 | +0.44 |
| PASCAL VOC | IoU | 87.80 | 88.23 | +0.43 |
| Cityscapes | Dice | 82.62 | 83.23 | +0.61 |
| Cityscapes | IoU | 75.67 | 76.19 | +0.52 |
| ADE20K | Dice | 63.98 | 64.92 | +0.93 |
| ADE20K | IoU | 56.95 | 57.67 | +0.72 |
| KiTS | Dice | 61.16 | 63.53 | +2.37 |
| KiTS | IoU | 54.19 | 56.21 | +2.02 |
| MSD Pancreas | Dice | 50.30 | 54.87 | +4.57 |
| MSD Pancreas | IoU | 35.29 | 39.19 | +3.90 |
| MSD Spleen | Dice | 94.79 | 94.75 | -0.04 |
| MSD Spleen | IoU | 90.12 | 90.06 | -0.07 |

MSD results are foreground-only means over **20 fixed Pancreas volumes** and
**9 fixed Spleen volumes**, using the frozen MONAI
`swin_unetr_btcv_segmentation==0.5.8` checkpoint. Both cohorts are external
tests under the [declared checkpoint provenance](#checkpoint-provenance).
The small Spleen decline is included: gains are not guaranteed on every dataset.

### MONAI decoder latency

Measured on an **NVIDIA RTX 3090**, with one complete resampled 3D volume per
call and one untimed warmup invocation. All values are **ms/volume**; Δ is
computed before rounding. Inference, preprocessing, I/O, and transfers are
excluded.

| Dataset | Argmax mean | RankSEG-RMA mean | Δ mean | Argmax median | RankSEG-RMA median |
| :--- | ---: | ---: | ---: | ---: | ---: |
| MSD Pancreas | 0.28 | 3.71 | +3.42 | 0.27 | 3.60 |
| MSD Spleen | 0.33 | 3.93 | +3.60 | 0.27 | 3.68 |

With the IoU objective, these cohorts produce the same rounded Dice/IoU
scores; RankSEG mean latency was 3.30 ms/volume for Pancreas and 3.47 ms/volume
for Spleen. Timings depend on hardware and run conditions.

<details>
<summary>RankSEG 0.0.5 → 0.0.6 release comparison</summary>

During 0.0.6 release validation, the published 0.0.5 wheel and the 0.0.6
candidate were evaluated on the same cached MONAI tensors with the Dice
objective. Dice and IoU were exactly equal on both cohorts.

| Dataset | 0.0.5 mean (ms/volume) | 0.0.6 candidate mean (ms/volume) |
| :--- | ---: | ---: |
| MSD Pancreas | 4.22 | 3.71 |
| MSD Spleen | 4.48 | 3.93 |

Mean latency decreased by about 12.2% in each run. This is a run-specific
timing comparison, not a universal speedup claim; score equality is the
compatibility check.

</details>

## Run benchmarks

### Available datasets

“Channels” describes the stored probabilities, not the number of scored
classes. KiTS and both MSD targets evaluate foreground only.

| Dataset ID | Channels | Input | Evaluation |
| :--- | ---: | :--- | :--- |
| `pascal_voc` | 21 | Hosted 2D probabilities | Per-image, multiclass |
| `ade20k` | 150 | Hosted 2D probabilities | Per-image, multiclass |
| `cityscapes` | 19 | Hosted 2D probabilities | Per-image, multiclass |
| `kits` | 2 | Hosted 2D slice probabilities | Per-case, foreground |
| `monai_btcv_swin_v058_msd_pancreas` | 2 | Local 3D probability cache | Per-volume, foreground |
| `monai_btcv_swin_v058_msd_spleen` | 2 | Local 3D probability cache | Per-volume, foreground |

```bash
rankseg-bench --list-datasets

# Full ADE20K run with per-class results and JSON export.
rankseg-bench --dataset ade20k --solver RMA --metric dice \
    --per-class --json results_ade20k.json

# Optimize IoU instead of Dice.
rankseg-bench --dataset pascal_voc --solver RMA --metric iou
```

### Common options

| Option | Purpose |
| :--- | :--- |
| `--device cpu` / `--device cuda` | Choose the execution device. |
| `--limit 5` | Run a small subset; for KiTS, the limit applies per fold. |
| `--warmup 1` | Set untimed warmup samples; default: 3. |
| `--batch-size 8` | Batch same-shape inputs; MONAI volume runs use 1. |
| `--per-class` | Include per-class Dice/IoU and gains. |
| `--json results_voc.json` | Save the report and per-class arrays. |
| `--cache-dataset` | Cache a hosted dataset locally instead of streaming. |
| `--cache-dir /path/to/cache` | Set the Hugging Face dataset cache directory. |
| `--rankseg-path /path/to/rankseg` | Benchmark a local RankSEG checkout. |
| `--artifact-dir /path/to/artifacts` | Read a generated MONAI probability cache. |

`--rankseg-path` takes precedence over the `RANKSEG_PATH` environment variable.
If neither is set, the installed RankSEG package is used. Run
`rankseg-bench --help` for all options.

## MONAI datasets and checkpoints

The workflow has two stages: **generate probabilities once**, then **benchmark
decoders on the same cache**. A complete 3D volume remains one optimization
unit; it is not split into independent 2D predictions.

### 1. Prepare the environment and data

From the cloned repository:

```bash
python -m pip install -e ".[monai]"
rankseg-monai-cache --list
```

Obtain MSD Task07 Pancreas or Task09 Spleen under its original license. Pass
an extracted directory containing `imagesTr/` and `labelsTr/`; the generator
does not download or redistribute the medical datasets.

**Checkpoints, downloaded Bundles, datasets, and probability caches stay
local and are excluded by `.gitignore`.** Git contains code, documentation, and the
[pinned metadata and evaluation configuration](../../rankseg_benchmark/monai/monai_specs.json).
`--download-bundle` obtains the specified Bundle and checkpoint from the MONAI
Model Zoo.

### 2. Generate the Pancreas cache

```bash
rankseg-monai-cache \
    --benchmark monai_btcv_swin_v058_msd_pancreas \
    --dataset-root /data/Task07_Pancreas \
    --bundle-dir ./monai_bundles \
    --output-dir ./artifacts/monai_btcv_swin_v058_msd_pancreas \
    --download-bundle --device cuda
```

This generates the fixed **20-volume** cohort. Add `--limit 1` for a one-case
smoke run; a partial cache does not reproduce the full-cohort result or demo.

### 3. Evaluate the cached volumes

```bash
rankseg-bench \
    --dataset monai_btcv_swin_v058_msd_pancreas \
    --artifact-dir ./artifacts/monai_btcv_swin_v058_msd_pancreas \
    --solver RMA --metric dice --device cuda --batch-size 1 --warmup 1 \
    --json results_monai_pancreas.json
```

Use `--metric iou` to run the IoU objective. The report scores only foreground
and measures decoder-only latency.

<details>
<summary>MSD Spleen commands (9 volumes)</summary>

```bash
rankseg-monai-cache \
    --benchmark monai_btcv_swin_v058_msd_spleen \
    --dataset-root /data/Task09_Spleen \
    --bundle-dir ./monai_bundles \
    --output-dir ./artifacts/monai_btcv_swin_v058_msd_spleen \
    --download-bundle --device cuda

rankseg-bench \
    --dataset monai_btcv_swin_v058_msd_spleen \
    --artifact-dir ./artifacts/monai_btcv_swin_v058_msd_spleen \
    --solver RMA --metric dice --device cuda --batch-size 1 --warmup 1 \
    --json results_monai_spleen.json
```

</details>

### Checkpoint provenance

Both targets use **BTCV Swin UNETR 0.5.8**, not a checkpoint trained on the
same MSD task. The official
[Bundle documentation](https://github.com/Project-MONAI/model-zoo/blob/dev/models/swin_unetr_btcv_segmentation/docs/README.md)
identifies BTCV as the supervised task. MONAI's
[pretraining data loader](https://github.com/Project-MONAI/research-contributions/blob/main/SwinUNETR/Pretrain/utils/data_utils.py)
lists LUNA16, TCIA COVID-19, HNSCC, TCIA Colonography, and LIDC-IDRI as
self-supervised sources; MSD Task07 and Task09 are not listed.

Under this declared-dataset provenance, the fixed MSD cases are external
tests: they are not used for checkpoint training or selection, post-processing
parameter selection, or threshold fitting. Each built-in target validates
`checkpoint_exposure=none_by_declared_dataset_provenance`, false test-case
usage flags, and a test dataset ID distinct from every declared supervised
and self-supervised source ID. These declarations are copied into each manifest.

This is an auditable **dataset-level safeguard**, not an independent
patient-identity audit beyond upstream metadata.

<details>
<summary>Preprocessing, foreground mapping, and artifact checks</summary>

The 14-channel BTCV softmax is collapsed to
`[1 - p(target), p(target)]`, preserving the target-versus-rest distribution.

| Target | Model channel | MSD foreground labels |
| :--- | ---: | :--- |
| Pancreas | 11 | 1 and 2: pancreas parenchyma + tumor |
| Spleen | 1 | 1: spleen |

The generator uses RAS orientation, 1.5 × 1.5 × 2.0 mm spacing, the Bundle's
intensity window, a 96³ sliding window, 0.5 overlap, and AMP inference.
`sw_batch_size=1` replaces the deployment setting of 4 to fit 24 GiB GPUs;
this can affect inference throughput and small floating-point reduction
details, but inference is outside decoder timing.

Probabilities are stored as float32, with labels aligned in the deterministic
resampled model space used by the official MONAI RankSEG tutorial. Each case
is stored separately and its SHA-256 is verified when read. Manifests record:

- Bundle name, version, official checkpoint checksum, and generated SHA-256;
- the fixed case list and checkpoint-exposure declarations;
- preprocessing, inference, activation, and coordinate-space settings;
- software versions and source image/label hashes.

</details>

<details>
<summary>Reproduce the README medical demo</summary>

The figure has three large panels: ground truth, Argmax, and RankSEG. Its
Dice scores describe the selected slice; the CLI separately prints the
20-volume mean for the caption. Both themes use transparent backgrounds.

Use the **complete Pancreas cache**:

```bash
python -m pip install -e ".[monai,demo]"

rankseg-monai-demo \
    --artifact-dir ./artifacts/monai_btcv_swin_v058_msd_pancreas \
    --dataset-root /data/Task07_Pancreas \
    --output ./monai_pancreas_rankseg.png \
    --theme light --device cuda

rankseg-monai-demo \
    --artifact-dir ./artifacts/monai_btcv_swin_v058_msd_pancreas \
    --dataset-root /data/Task07_Pancreas \
    --output ./monai_pancreas_rankseg_dark.png \
    --theme dark --device cuda
```

After aggregate evaluation, the renderer selects the axial slice that
maximizes corrected minus introduced error pixels among slices with at least
400 foreground pixels, using deterministic tie-breakers. This selection is
for presentation only; it does not change the checkpoint, post-processing
parameters, or the reported cohort result.

</details>

## Evaluation protocol

### Dice and IoU

| Dataset | Scoring rule |
| :--- | :--- |
| PASCAL VOC, Cityscapes, ADE20K | Compute per-class Dice/IoU for each image, average over classes present in its ground truth, then average across images. |
| KiTS | Compute foreground slice scores, average within each case, then across cases and folds (`mDiceI` / `mIoUI`). |
| MSD Pancreas, MSD Spleen | Score foreground over each complete resampled 3D volume, then average across the fixed cohort. |

KiTS stores two-channel background/foreground probabilities for Argmax. Each
RankSEG solver receives only the foreground channel in `multilabel` mode,
and its mask is converted back to binary class labels for evaluation.
The MONAI targets also score foreground only, using the mappings above.

For custom multilabel tasks, a class is active if it appears in either the
prediction or ground truth. `--per-class` reports each class's mean Dice/IoU
over its active evaluation units.

Metric definitions follow the [RankSEG-RMA paper](https://openreview.net/pdf?id=4tRMm1JJhw),
its [metric implementation](https://github.com/ZixunWang/RankSEG-RMA/blob/master/exp/metrics/accuracy_metric.py),
and the medical protocol in `exp/test.py::test_medical`.

### Runtime

Timing compares `RankSEG.predict` with `torch.argmax`, with CUDA
synchronization where applicable. Units are **ms/image**, **ms/slice**, or
**ms/volume**, as shown in the report. Model inference, preprocessing, I/O,
and host/device transfers are excluded.

The default warmup is 3 samples; the MONAI results above use 1. Warmup adds
untimed calls but never removes samples from metric evaluation. Hardware,
input size, batching, solver, and objective can all affect latency.

## Bring your own model

Hosted benchmarks use
[`ZixunWang/rankseg-benchmark`](https://huggingface.co/datasets/ZixunWang/rankseg-benchmark),
with `pascal_voc`, `ade20k`, `cityscapes`, and `kits/fold*` data directories.
For another model, generate probabilities with your own inference pipeline
and create dataset rows with:

| Field | Shape | Content |
| :--- | :--- | :--- |
| `probs` | `(classes, *spatial)` | Finite probabilities in `[0, 1]`. |
| `label` | `(*spatial)` or `(classes, *spatial)` | Multiclass indices or multilabel masks, on the same spatial grid. |

The repository supplies a MONAI generator, not a generic model-inference
script. Once your rows are hosted in a Hugging Face dataset repository, add
an entry to [`datasets.py`](../../rankseg_benchmark/datasets.py) or use the Python API:

```python
from rankseg_benchmark.datasets import DatasetSpec
from rankseg_benchmark.runner import format_report, run_benchmark

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
baseline, rankseg = run_benchmark(spec, solver="RMA")
print(format_report(baseline, rankseg, per_class=True))
```

## License

[BSD 3-Clause](../../LICENSE).
