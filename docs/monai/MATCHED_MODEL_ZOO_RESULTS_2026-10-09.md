# RankSEG evaluation of matched MONAI Model Zoo Bundles — 2026-10-09

Released MONAI Model Zoo checkpoints were evaluated on their matching labeled
datasets by RankSEG, not by MONAI. Argmax and RankSEG-RMA with safe screening
decoded the *same* probability volume.
The benchmark code is [`matched_official.py`](../../rankseg_benchmark/monai/matched_official.py).

| Bundle / evaluation cohort | Cases | Reader | Argmax Dice | RMA Dice | Δ Dice | Argmax IoU | RMA IoU | Δ IoU | Argmax ms/volume | RMA ms/volume | Δ ms |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Prostate MRI Anatomy 0.3.6](https://github.com/Project-MONAI/model-zoo/blob/dev/models/prostate_mri_anatomy/docs/README.md) / Prostate158 independent test | 19 | 1 | 81.753 | 81.917 | +0.165 pp | 69.779 | 69.987 | +0.207 pp | 0.184 | 1.108 | +0.924 |
| Same | 19 | 2 | 80.033 | 80.312 | +0.279 pp | 67.742 | 68.078 | +0.336 pp | 0.184 | 1.108 | +0.924 |
| [Spleen CT 0.6.1](https://github.com/Project-MONAI/model-zoo/blob/dev/models/spleen_ct_segmentation/docs/README.md) / MSD Task09 validation | 9 | — | 95.825 | 95.840 | +0.015 pp | 92.025 | 92.053 | +0.027 pp | 0.225 | 1.117 | +0.892 |

Dice and IoU are case-level foreground macro means, then averaged across cases;
background is excluded. A class absent from ground truth is omitted for that
case. Values are percentages, whereas `pp` is the absolute percentage-point
difference. Timing is decoder-only on an RTX 3090: two warmups and seven
alternating Argmax/RMA calls per case, synchronized CUDA elapsed time, then
the mean of per-case medians. Model inference and data loading are excluded.
Independent full-cohort reruns reproduced every case-level Dice/IoU value and
source-file SHA-256 exactly; another run's decoder times were 0.184/1.105 ms
for Prostate and 0.225/1.117 ms for Spleen (Argmax/RMA).

| Prostate158 foreground | Reader | Argmax Dice | RMA Dice | Δ Dice |
| :--- | :--- | ---: | ---: | ---: |
| Central gland | 1 | 87.732 | 87.764 | +0.032 pp |
| Peripheral zone | 1 | 75.773 | 76.071 | +0.298 pp |
| Central gland | 2 | 87.848 | 87.931 | +0.084 pp |
| Peripheral zone | 2 | 72.218 | 72.693 | +0.475 pp |

## Cohort and checkpoint audit

- Prostate158: the [publisher's test archive](https://zenodo.org/records/6592345)
  contains 19 separately released cases with two anatomy annotations. Its MD5
  is `403b3736cb12013ed7585e07cbda6978`. The [training archive](https://zenodo.org/records/6481141)
  contains 119 train and 20 validation entries in its CSVs. The numeric case
  identifiers restart at `001` in the test archive, so the audit uses
  split-qualified paths and checks the training/test ZIP directories for
  byte-identical T2, ADC and DWI files. Each modality has 139 training files,
  19 test files and zero matching `(uncompressed size, CRC32)` signatures.
  All 57 evaluated T2 and anatomy-label files were also checked against their
  entries in the test ZIP by size and CRC32.
  This excludes byte-identical image overlap; the publisher's separate-test
  declaration supplies the patient-level split provenance. The Bundle
  checkpoint SHA-256 is
  `3c18e4c658bb088f551b1d63219ef8340fe6256d016c75fc140d3da49dda696d`.
- MSD Spleen: the [official Bundle training configuration](https://github.com/Project-MONAI/model-zoo/blob/dev/models/spleen_ct_segmentation/configs/train.json)
  uses sorted `imagesTr/labelsTr` indices `[:-9]` for training and `[-9:]` for
  validation: 32 versus 9, with no case-ID overlap. The 9 cases were used to
  select the saved checkpoint, so this row is **validation**, not independent
  test. The checkpoint SHA-256 is
  `502c312893994ec071a85c4f2e7d83a43f7789969faa73dae62f1291177f50fe`.

Both runs used MONAI 1.4.0, published RankSEG 0.0.7, PyTorch 2.11.0+cu130,
AMP inference, and the Bundle's preprocessing, architecture, checkpoint,
sliding-window ROI and overlap. A deterministic seed of 123 and deterministic
PyTorch algorithms were used. `sw_batch_size=1` bounds GPU memory; the
official configs specify 4. Labels were resampled with nearest-neighbor
interpolation into the Bundle model space. Both decoders were evaluated there
without inverse resampling, largest-component filtering or other
postprocessing. RMA used `metric="dice"`, `output_mode="multiclass"`,
`smooth=0`, `pruning_prob=0.5`, and `safe_screening=True`. These numbers are
not directly the Bundles' published native-space, postprocessed Dice values.

The complete per-case JSON outputs remain local in
`artifacts/monai_matched_prostate_test_screened_v140.json` and
`artifacts/monai_matched_spleen_validation_screened_v140.json`; checkpoints and medical
images remain in ignored `artifacts/` directories.

Pancreas DiNTS and BraTS are not included here: their official Bundle
documentation describes internal test splits, but the downloaded Pancreas
Bundle does not contain a frozen case list bound to its released checkpoint,
and the BraTS data/checkpoint split was not independently audited in this run.

## Re-run

Install `rankseg==0.0.7` and `monai==1.4.0`, then download the named Model Zoo
Bundles and the linked datasets under their own terms. With the Bundle parent
at `./artifacts/monai_bundles`, run:

```bash
python -m rankseg_benchmark.monai.matched_official \
  --task prostate158_test \
  --dataset-root ./artifacts/prostate158_test_dataset/prostate158_test \
  --training-list-root ./artifacts/prostate158_train_lists/prostate158_train \
  --test-archive ./artifacts/prostate158_test.zip \
  --bundle-dir ./artifacts/monai_bundles \
  --safe-screening \
  --output ./artifacts/monai_matched_prostate_test_screened_v140.json

python -m rankseg_benchmark.monai.matched_official \
  --task msd_spleen_validation \
  --dataset-root /data/Task09_Spleen \
  --bundle-dir ./artifacts/monai_bundles \
  --safe-screening \
  --output ./artifacts/monai_matched_spleen_validation_screened_v140.json
```

For Prostate158, `--training-list-root` holds only `train.csv` and `valid.csv`
extracted from the official training ZIP; the evaluator checks those CSVs
against the remote official ZIP before running.
