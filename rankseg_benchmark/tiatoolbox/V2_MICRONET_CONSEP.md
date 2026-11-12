# MicroNet / CoNSeP with TIAToolbox 2.1.3

Run date: 2026-10-26. This is a separate follow-up to the frozen
[TIAToolbox 1.6.0 benchmarks](RESULTS.md). All 14 CoNSeP test images were
evaluated with the same checkpoint and input files as before.

**Result:** foreground RankSEG improves mean samplewise nucleus Dice by
**0.454 percentage points**, with 11/14 images improving. AJI decreases and
PQ is approximately unchanged. The earlier 5.392-point Dice gain does not
carry over to the 2.1.3 inference pipeline.

## Main comparison

Argmax and RankSEG receive identical saved softmax probabilities. Both use
the unchanged official MicroNet instance postprocessor. Each metric is
computed per image and then averaged; Dice is binary foreground Dice, not
a pooled pixel score. Changes and intervals are in percentage points (pp).

| Endpoint | Argmax | RankSEG foreground | Change (pp) | Paired 95% bootstrap CI (pp) | Improved |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nucleus Dice after postprocessing | 0.796590 | 0.801132 | +0.454 | +0.226 to +0.662 | 11/14 |
| AJI | 0.492571 | 0.479085 | -1.349 | -1.961 to -0.707 | 2/14 |
| PQ | 0.467760 | 0.466972 | -0.079 | -0.446 to +0.329 | 5/14 |
| Foreground Dice before postprocessing | 0.797176 | 0.801316 | +0.414 | +0.173 to +0.633 | 11/14 |

The primary RankSEG configuration decodes only the nucleus probability
with `output_mode="multilabel"`, yielding one binary foreground mask. The
two-channel `output_mode="multiclass"` comparison is secondary:

| Endpoint | Argmax | RankSEG multiclass | Change (pp) | Paired 95% bootstrap CI (pp) | Improved |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nucleus Dice after postprocessing | 0.796590 | 0.800361 | +0.377 | +0.178 to +0.558 | 11/14 |
| AJI | 0.492571 | 0.482419 | -1.015 | -1.490 to -0.536 | 3/14 |
| PQ | 0.467760 | 0.467708 | -0.005 | -0.289 to +0.321 | 5/14 |

## Per-image Dice after official postprocessing

| Image | Argmax | RankSEG foreground | Foreground change (pp) | RankSEG multiclass |
| --- | ---: | ---: | ---: | ---: |
| test_1 | 0.787708 | 0.794432 | +0.672 | 0.793425 |
| test_2 | 0.726829 | 0.735686 | +0.886 | 0.735277 |
| test_3 | 0.798441 | 0.808451 | +1.001 | 0.806932 |
| test_4 | 0.806434 | 0.811536 | +0.510 | 0.810259 |
| test_5 | 0.770240 | 0.775626 | +0.539 | 0.774081 |
| test_6 | 0.834445 | 0.831418 | -0.303 | 0.831457 |
| test_7 | 0.798864 | 0.798610 | -0.025 | 0.798205 |
| test_8 | 0.802040 | 0.810956 | +0.892 | 0.808992 |
| test_9 | 0.802402 | 0.807000 | +0.460 | 0.806723 |
| test_10 | 0.813523 | 0.815643 | +0.212 | 0.814929 |
| test_11 | 0.800095 | 0.807218 | +0.712 | 0.805795 |
| test_12 | 0.799594 | 0.805635 | +0.604 | 0.805349 |
| test_13 | 0.814529 | 0.820961 | +0.643 | 0.819488 |
| test_14 | 0.797120 | 0.792678 | -0.444 | 0.794137 |

All cases, including the three foreground-Dice losses, remain in the table.
The underlying unrounded values are in [the per-image CSV](evidence/micronet_consep_v2.csv).

## Why the old baseline was lower

| Official pipeline | Argmax Dice | RankSEG foreground Dice | Change (pp) |
| --- | ---: | ---: | ---: |
| TIAToolbox 1.6.0, historical run | 0.737604 | 0.791527 | +5.392 |
| TIAToolbox 2.1.3, current run | 0.796590 | 0.801132 | +0.454 |

The main difference is padding outside the image. The old
[`WSIStreamDataset`](https://github.com/TissueImageAnalytics/tiatoolbox/blob/v1.6.0/tiatoolbox/models/engine/semantic_segmentor.py#L420)
uses black padding (`0`), whereas the new
[`WSIPatchDataset`](https://github.com/TissueImageAnalytics/tiatoolbox/blob/0d1495f726aa73118da3e2b61d70ea50784505f7/tiatoolbox/models/dataset/dataset_abc.py#L434)
uses white padding (`255`). MicroNet standardizes each patch, so the padding
also affects predictions inside patches that extend beyond the image.

A controlled replay used the same 2.1.3 model, weights, patch coordinates,
preprocessing and overlap averaging, changing only the padding constant.
Over all 14 images:

- Black padding nearly reproduces the old probability maps: the largest
  per-image mean absolute error is `1.16e-9`; only 8 of 14 million raw argmax
  pixels differ. The maximum absolute probability error is `9.33e-5`.
- White padding nearly reproduces the new maps: the largest per-image mean
  absolute error is `5.91e-8`; only 1 raw argmax pixel differs. The maximum
  absolute probability error is `2.22e-4`. All 14 final Dice scores agree
  exactly with the saved 2.1.3 baseline.
- Holding the 2.1.3 postprocessor fixed, mean Dice is **0.736311** with black
  padding and **0.796590** with white padding. Thus padding accounts for the
  approximately six-point difference in the inference baseline. The replay
  is numerically very close, but its probabilities are not bitwise identical.

There is also a smaller postprocessing change. In 2.1.3,
[`MicroNet.postproc`](https://github.com/TissueImageAnalytics/tiatoolbox/blob/0d1495f726aa73118da3e2b61d70ea50784505f7/tiatoolbox/models/architecture/micronet.py)
returns the connected-component map before hole filling as `predictions`,
while the instance information uses the filled map. The 1.6 implementation
returned the filled map. Replaying the old probabilities with the new
postprocessor gives Dice 0.736312; restoring hole filling recovers the old
0.737604 score exactly on every image. The primary results above use the
unchanged 2.1.3 returned predictions, with no added hole filling.

The new baseline rounds to **Dice 0.80 / AJI 0.49 / PQ 0.47**, matching the
values in the [official MicroNet documentation](https://github.com/TissueImageAnalytics/tiatoolbox/blob/0d1495f726aa73118da3e2b61d70ea50784505f7/tiatoolbox/models/architecture/micronet.py#L402).
This does not establish byte identity with the original authors' evaluation.

## Protocol and checks

- **Environment:** Python 3.12.15, TIAToolbox 2.1.3, PyTorch 2.11.0 with
  CUDA 13.0, torchvision 0.26.0, RankSEG 0.0.8. Model inference and RankSEG
  decoding used an NVIDIA GeForce RTX 3090 (24 GB). The isolated environment
  passed `pip check`; package versions and source hashes are recorded in
  [the provenance and audit metadata](metadata/micronet_consep_v2.json).
- **Inputs:** the same 14 `CoNSeP/Test` images and labels, checkpoint SHA-256
  `f085d591f74cf8de3a7a5a75f08fab3e47859c49bfc7415bc567113dba778279`.
  The checkpoint matches the TIACentre release. The CoNSeP Kaggle mirror
  archive SHA-256 is
  `7f62e3fed9435d5e91568944bca9edd112ff0a67d0940a5300fbd848e5c255dd`;
  its byte identity with the authors' archive remains unverified.
- **Inference:** public `MultiTaskSegmentor`, registered 252×252 input/output
  patches, stride 150, batch size 4, no tissue mask, compilation disabled.
  Each 1000×1000 PNG receives `mpp=0.25` metadata, preserving its original
  pixel scale. Outputs are finite float32 softmax probabilities summing to
  one; there is no extra softmax, calibration, or rescaling.
- **Decoding:** `metric="dice"`, `solver="RMA"`, `safe_screening=True`,
  `pruning_prob=0.5`, `smooth=0`. No retraining or parameter tuning. Neither
  decoder receives the ground truth. There are no ignored pixels; all
  14 ground-truth foreground masks are nonempty.
- **Postprocessing:** direct postprocessing of the cached probabilities
  exactly matches the engine's instance array on all 14 images. The binary
  argmax adapter reproduces that same array on all 14 images. RankSEG uses
  the same adapter and official postprocessor.
- **Metrics:** Dice uses `tiatoolbox.utils.metrics.dice` on `inst_map > 0`.
  AJI and PQ use the existing instance scorer. All three metrics were
  checked against the [HoVer-Net authors' scorer](https://github.com/vqdang/hover_net/blob/67e2ce5e3f1a64a2ece77ad1c24233653a9e0901/metrics/stats_utils.py)
  on `test_1` and `test_2` for all three methods: all six comparisons matched
  exactly. This scorer audit is a subset, while the reported means use all
  14 images.
- **Statistics:** paired image bootstrap, 10,000 resamples, seed 20261026.
  The table uses unrounded per-image scores. With only 14 test images,
  these intervals do not establish generalization to other cohorts.

## Reproduction

Use the input paths and download instructions in [README.md](README.md).
Create a Python 3.12 environment, then run from the repository root:

```bash
python -m pip install -r environments/tiatoolbox-v2.txt
python -m rankseg_benchmark.tiatoolbox.micronet_consep_v2 infer
python -m rankseg_benchmark.tiatoolbox.micronet_consep_v2 evaluate
python -m rankseg_benchmark.tiatoolbox.micronet_consep_v2 report
```

Recompute the main table from the tracked CSV without images, weights or GPU:

```bash
python -m rankseg_benchmark.tiatoolbox.micronet_consep_v2 report --from-evidence
```

The runner keeps inputs, probabilities, official predictions, evaluated
instances and full results under ignored `artifacts/` paths. Historical
1.6 artifacts are optional for the primary comparison; if available, the
evaluator also records the old-baseline replay diagnostics.

For the scorer audit, download the linked, pinned `stats_utils.py` and run:

```bash
python -m rankseg_benchmark.tiatoolbox.micronet_consep_v2 audit \
  --reference-scorer /path/to/stats_utils.py --limit 2
```

The padding diagnostic requires both the historical and new probability
maps and a CUDA GPU:

```bash
python -m rankseg_benchmark.tiatoolbox.micronet_padding_audit
```

## Interpretation

This run supports a modest improvement in binary nucleus Dice on this
checkpoint and cohort. It does not support a general instance-segmentation
improvement: AJI worsens and PQ shows no clear change. The earlier larger
Dice gain occurred under the older inference conditions, including black
padding. For a TIAToolbox 2.x example, the current paired results and these
limitations should be presented together.
