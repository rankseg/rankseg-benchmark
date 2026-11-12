# TIAToolbox 2.1.3 migration across the benchmark cohorts

This follow-up uses the same images, labels and checkpoint files as the
historical TIAToolbox 1.6.0 runs. It regenerates probabilities with the
TIAToolbox 2.1.3 public engines and evaluates paired standard/RankSEG decoding.
Within each run, both decoders receive the same probabilities. Probabilities
are not assumed to be identical between TIAToolbox versions.

The complete score tables are generated in [RESULTS_V2.md](RESULTS_V2.md).
The original [1.6.0 results](RESULTS.md) remain available for comparison.

## Cohorts and inference

| Runner cohort | Registered checkpoint | Evaluation images | Public 2.1.3 engine |
| --- | --- | ---: | --- |
| `micronet_consep` | `micronet-consep` | 14 CoNSeP Test images | `MultiTaskSegmentor` |
| `micronet_kumar` | `micronet-consep` | 14 Kumar test images | `MultiTaskSegmentor` |
| `hovernet_consep` | `hovernet_original-consep` | 14 CoNSeP Test images | `MultiTaskSegmentor` |
| `hovernet_kumar` | `hovernet_original-kumar` | 14 Kumar test images | `MultiTaskSegmentor` |
| `bcss` | `fcn_resnet50_unet-bcss` | 45 BCSS test-center ROIs | `SemanticSegmentor` |

The two CoNSeP models share the same images, as do the two Kumar rows.
MicroNet/Kumar is a cross-dataset evaluation. All input paths and hashes are
recorded in `metadata/*_v2.json`; raw inputs and predictions stay ignored.
The existing [README](README.md) documents input sources and checkpoint hashes.
CoNSeP comes from the previously used Kaggle mirror; its byte identity with
the authors' original archive has not been verified.

All models retain their registered input/output resolutions, patch shapes
and stride. Images receive `mpp=0.25` metadata to retain the original pixel
scale. Inference covers the whole image, with no tissue mask or ground-truth
mask. Torch compilation is disabled. Batch size is 4 for nucleus models and
2 for BCSS, matching the old runs. Model inference and RankSEG decoding use
an NVIDIA RTX 3090 (24 GB); scoring and morphological postprocessing use CPU.

## Paired evaluation and checks

RankSEG uses version 0.0.8, `metric="dice"`, `solver="RMA"`,
`safe_screening=True`, `pruning_prob=0.5`, and `smooth=0`. The decoder source
hash is verified. There is no probability rescaling, extra softmax, parameter
tuning or retraining.

For each image, the runner checks probability shape/range/normalization and
compares direct postprocessing of the saved outputs with the official
engine's saved prediction array. For nucleus models it additionally checks
that replacing the probability input with the baseline binary mask leaves
the instance array unchanged. RankSEG then uses that same mask adapter and
the unchanged official postprocessor. HoVer-Net retains its original HV
and type maps in both arms.

Nucleus Dice is binary Dice on `inst_map > 0`, computed per image and then
averaged. AJI and PQ use the same instance scorer as the preceding benchmark.
BCSS keeps the previous five-class label mapping and nearest-neighbor label
alignment. Raw annotation codes 0 and 7 are ignored only in scoring; both
decoders process the entire probability map. Other is included. Both-empty
class masks score 1, while a false-positive prediction for an absent class
scores 0. Five class scores are averaged within each ROI, then across ROIs.

The primary BCSS output is multiclass. Its independent-class multilabel
result is reported separately because its masks may overlap. MicroNet's
primary result uses the nucleus channel with multilabel decoding to produce
one binary foreground mask; its two-channel multiclass result is secondary.

All cases, including losses, remain in the results. Paired bootstrap
intervals use 10,000 resamples and seed 20261026. Per-image CSVs contain
unrounded scores. These are small exploratory nucleus cohorts; the BCSS
ROI-wise metric does not reproduce the exact aggregation used for the
official published BCSS reference score.

## Version differences and controlled padding replay

The old engine's
[`WSIStreamDataset`](https://github.com/TissueImageAnalytics/tiatoolbox/blob/v1.6.0/tiatoolbox/models/engine/semantic_segmentor.py#L420)
fills out-of-image pixels with 0, while the new
[`WSIPatchDataset`](https://github.com/TissueImageAnalytics/tiatoolbox/blob/0d1495f726aa73118da3e2b61d70ea50784505f7/tiatoolbox/models/dataset/dataset_abc.py#L434)
uses 255. MicroNet standardizes each patch, including these pixels, so this
changes the inputs of boundary patches. HoVer-Net and the BCSS UNet use
different preprocessing; the size of the effect must be measured for each
model and cohort.

The [MicroNet/CoNSeP audit](V2_MICRONET_CONSEP.md) replayed both padding values
on all 14 images. For each remaining cohort, `cohorts_v2 ... padding --limit 1`
replays the fixed first image in manifest order. Model, weights, patch grid,
preprocessing and overlap averaging stay fixed; only the padding constant
changes. Both replays are compared with the corresponding saved old/new
probability maps. These one-image diagnostics establish behavior on those
images, not a whole-cohort padding ablation.

For the fixed first nucleus image in each remaining cohort, the controlled
replay gives the following Dice under the same 2.1.3 postprocessor:

| Cohort / image | Black padding | White padding | Replay versus saved maps |
| --- | ---: | ---: | --- |
| MicroNet / Kumar, `test_same__TCGA-21-5784-01Z-00-DX1` | 0.680246 | 0.678343 | Raw argmax masks and final Dice agree; maximum probability error below 0.000139. |
| HoVer-Net / CoNSeP, `test_1` | 0.852281 | 0.865101 | All probability/HV/type maps agree exactly. |
| HoVer-Net / Kumar, `test_same__TCGA-21-5784-01Z-00-DX1` | 0.829825 | 0.825469 | Both nucleus probability and HV maps agree exactly. |

Keeping the 2.1.3 postprocessor fixed gives the following full-cohort baseline
comparison. This separates probability-map changes from postprocessing changes:

| Cohort | Old probability maps + 2.1.3 postprocessor | New maps + 2.1.3 postprocessor |
| --- | ---: | ---: |
| MicroNet / CoNSeP | 0.736312 | 0.796590 |
| MicroNet / Kumar | 0.590730 | 0.612634 |
| HoVer-Net CoNSeP / CoNSeP | 0.838820 | 0.847074 |
| HoVer-Net Kumar / Kumar | 0.827551 | 0.825251 |
| FCN ResNet50 UNet / BCSS | 0.592789 | 0.591457 |

The baseline change depends on the model and cohort: for example, the new
HoVer-Net/Kumar baseline is slightly lower. White padding should not be
interpreted as a universal improvement. The primary comparison follows the
official 2.1.3 pipeline regardless of the direction of this change.

**BCSS has an additional configuration change:** its registered stride is
[256 pixels in 1.6.0](https://github.com/TissueImageAnalytics/tiatoolbox/blob/v1.6.0/tiatoolbox/data/pretrained_model.yaml#L611)
and [450 pixels in 2.1.3](https://github.com/TissueImageAnalytics/tiatoolbox/blob/0d1495f726aa73118da3e2b61d70ea50784505f7/tiatoolbox/data/pretrained_model.yaml#L614).
The input/output patches remain 1024/512 pixels. The new run uses the official
450-pixel stride for both decoders; old/new BCSS differences cannot be
attributed to padding alone. Replaying the fixed first ROI with the new
stride and white padding reproduces its new argmax mask exactly. Black
padding with the new stride does not reproduce the old map.

The same fixed ROI (`TCGA-E2-A14N-DX1`) was also replayed with the historical
stride, giving a two-factor diagnostic under the same model and scorer:

| Stride | Black padding macro Dice | White padding macro Dice | Patches |
| --- | ---: | ---: | ---: |
| 256 (historical) | 0.682088 | 0.682096 | 672 |
| 450 (current) | 0.679696 | 0.679632 | 216 |

Stride 256 with black padding reproduces the old argmax map exactly; stride
450 with white padding reproduces the new argmax map exactly. Both matching
probability replays have maximum error below `6e-8`. For this ROI, changing
stride has a larger effect than changing the padding color. This diagnostic
does not change the main 45-ROI comparison's registered 450-pixel stride.

MicroNet also changed which instance map its postprocessor returns: 1.6
returns the hole-filled map, while 2.1.3 returns the map before hole filling.
The primary comparisons always use the unmodified postprocessor for that
version. HoVer-Net's small-object removal calls were adapted to the newer
scikit-image API with equivalent size cutoffs. Its forward computation and
the BCSS UNet forward computation are unchanged in the inspected source.
Applying the new HoVer-Net postprocessor to the old saved maps also recovers
every old baseline Dice exactly for both 14-image cohorts.

## BCSS class-presence audit

The following is a diagnostic; the primary five-class metric still includes
all 45 ROIs with the same scoring rule for both decoders.

| Class | ROIs with class annotated | Argmax Dice on those ROIs | RankSEG multiclass | RankSEG multilabel |
| --- | ---: | ---: | ---: | ---: |
| Tumor | 45 | 0.828064 | 0.826528 | 0.829403 |
| Stroma | 45 | 0.759316 | 0.749677 | 0.777072 |
| Inflammatory | 40 | 0.710506 | 0.738721 | 0.737347 |
| Necrosis | 23 | 0.686171 | 0.691195 | 0.694644 |
| Other | 42 | 0.367706 | 0.340791 | 0.339153 |

Of the 22 ROIs with no annotated necrosis, argmax predicts some necrosis on
valid pixels in every ROI. Both RankSEG modes remove all such predictions
on three ROIs: `TCGA-E2-A14R-DX1`, `TCGA-GM-A2DI-DX1`, and
`TCGA-LL-A73Y-DX1`. Their necrosis Dice switches from 0 to 1 under the
both-empty convention. These three switches alone contribute **1.333 pp**
to the five-class, 45-ROI macro gain (`3 / (45 × 5)`). The total multiclass
macro gain is **1.160 pp**; the contribution from all remaining class/ROI
scores is therefore **−0.173 pp**. This is a valid false-positive reduction
under the chosen metric, but does not demonstrate a broad improvement in
segmentation overlap. On the 23 ROIs with annotated necrosis, multiclass
necrosis Dice improves by only **0.502 pp**.

There are no absent-class score switches for the other four classes.
Multilabel macro Dice improves by **1.746 pp** overall; subtracting the same
three switches leaves **0.413 pp** of contribution from the remaining
class/ROI scores. Its output still permits overlapping classes.

## Reproduction

From the repository root, use Python 3.12 and the existing input paths:

```bash
python -m pip install -r environments/tiatoolbox-v2.txt

# MicroNet / CoNSeP uses the runner from the first migration experiment.
python -m rankseg_benchmark.tiatoolbox.micronet_consep_v2 infer
python -m rankseg_benchmark.tiatoolbox.micronet_consep_v2 evaluate

for cohort in micronet_kumar hovernet_consep hovernet_kumar bcss; do
  python -m rankseg_benchmark.tiatoolbox.cohorts_v2 "$cohort" infer
  python -m rankseg_benchmark.tiatoolbox.cohorts_v2 "$cohort" evaluate
  python -m rankseg_benchmark.tiatoolbox.cohorts_v2 "$cohort" report
done
```

Use `--limit 1` for a pilot. Inference and evaluation resume completed cases
only with matching provenance. A fresh primary evaluation requires the
images, labels and weights, but does not require old probability maps.
When old maps exist, the evaluator also records how the old maps score
under the new postprocessor.

The optional padding replay requires both old and new saved probabilities:

```bash
python -m rankseg_benchmark.tiatoolbox.cohorts_v2 hovernet_consep padding --limit 1
# BCSS: additionally replay the historical stride as an isolated diagnostic.
python -m rankseg_benchmark.tiatoolbox.cohorts_v2 bcss padding --limit 1 --legacy-bcss-stride
# Recompute class-presence counts and conditional summaries from annotations.
python -m rankseg_benchmark.tiatoolbox.cohorts_v2 bcss presence
```

Rebuild the combined table from the small evidence CSVs, without a GPU or
TIAToolbox installation:

```bash
python -m rankseg_benchmark.tiatoolbox.table_v2 \
  --output rankseg_benchmark/tiatoolbox/RESULTS_V2.md
```

After a full run, use `--from-artifacts` to report its results. Add
`--check-artifacts` only when comparing with the frozen original run; it
checks byte-for-byte equality of the CSVs. All checkpoints, images,
probability maps and engine outputs remain under ignored `artifacts/` paths.
