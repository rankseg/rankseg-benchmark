# TIAToolbox paired-decoder results

TIAToolbox 1.6.0; RankSEG 0.0.8 decoder implementation, Dice/RMA, `safe_screening=True`, default `pruning_prob=0.5`. Each pair uses the same saved model probabilities. No retraining, probability scaling, or threshold tuning. Scores are calculated per image or ROI, then averaged.

## Primary single-mask comparisons

| Checkpoint / cohort | Endpoint | n | Argmax | RankSEG | Change (pp) | Paired 95% bootstrap interval (pp) | Improved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `micronet-consep` / CoNSeP Test | Nucleus Dice | 14 | 0.7376 | 0.7915 | +5.392 | +4.739 to +6.068 | 14/14 |
| `micronet-consep` / Kumar test | Nucleus Dice | 14 | 0.5924 | 0.6472 | +5.485 | +4.493 to +6.535 | 14/14 |
| `hovernet_original-consep` / CoNSeP Test | Nucleus Dice | 14 | 0.8388 | 0.8394 | +0.055 | -0.016 to +0.120 | 9/14 |
| `hovernet_original-kumar` / Kumar test | Nucleus Dice | 14 | 0.8276 | 0.8295 | +0.196 | +0.044 to +0.352 | 11/14 |
| `fcn_resnet50_unet-bcss` / BCSS test | Five-class ROI macro Dice | 45 | 0.5928 | 0.5918 | -0.097 | -0.664 to +0.448 | 23/45 |
| ↳ BCSS | Tumor | 45 | 0.8267 | 0.8260 | -0.068 | — | 21/45 |
| ↳ BCSS | Stroma | 45 | 0.7571 | 0.7487 | -0.836 | — | 14/45 |
| ↳ BCSS | Inflammatory | 45 | 0.6752 | 0.6995 | +2.422 | — | 25/45 |
| ↳ BCSS | Necrosis | 45 | 0.3724 | 0.3739 | +0.151 | — | 11/45 |
| ↳ BCSS | Other | 45 | 0.3326 | 0.3111 | -2.152 | — | 13/45 |

For nucleus models, Dice measures binary `inst_map > 0` after identical official instance postprocessing. MicroNet uses the nucleus foreground channel with `output_mode="multilabel"` to produce one binary mask; HoVer-Net uses the nucleus probability with the same output mode while keeping its HV/type maps unchanged. BCSS uses `output_mode="multiclass"` for one five-class mask. BCSS raw labels 0 and 7 are ignored for both methods; both-empty class Dice is 1 and an absent ground-truth class with predicted pixels scores 0. The BCSS per-class rows each average over all 45 ROIs, including class-absent ROIs. Bootstrap intervals resample paired images/ROIs 10,000 times.

## Secondary diagnostics

| Checkpoint / cohort | Endpoint | n | Argmax | RankSEG | Change (pp) | Paired 95% bootstrap interval (pp) | Improved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `micronet-consep` / Kumar test | AJI (exploratory) | 14 | 0.3941 | 0.4411 | +4.695 | — | 14/14 |
| `micronet-consep` / Kumar test | PQ (exploratory) | 14 | 0.3352 | 0.3672 | +3.200 | — | 14/14 |
| `fcn_resnet50_unet-bcss` / BCSS test | Independent-class multilabel Dice | 45 | 0.5928 | 0.5974 | +0.462 | -0.057 to +0.962 | 24/45 |

AJI/PQ use the same TIAToolbox MicroNet instance maps, scored by a local implementation of the HoVer-Net matching definitions; they are secondary checks. BCSS multilabel masks may overlap or leave pixels unassigned, so that row is not a valid single-label BCSS output.

## Interpretation limits

- The MicroNet CoNSeP argmax baseline (0.7376) does not reproduce the approximately 0.80 Dice quoted in [TIAToolbox's MicroNet documentation](https://tia-toolbox.readthedocs.io/en/latest/_autosummary/tiatoolbox.models.architecture.micronet.MicroNet.html). The cause is unresolved. CoNSeP was obtained from a Kaggle mirror whose byte identity with the authors' archive was not verified.
- The [TIAToolbox 1.6.0 BCSS table](https://github.com/TissueImageAnalytics/tiatoolbox/blob/v1.6.0/tiatoolbox/models/engine/semantic_segmentor.py) reports 0.763 overall; its exact aggregation is not reproduced here. Our ROI-wise argmax macro is 0.5928, while our pooled-pixel class macro is 0.7444. These are distinct metrics. The BCSS multiclass paired result shows no overall gain under our ROI-wise rule.
- The same CoNSeP images are used for both CoNSeP checkpoints, and the same Kumar images for both Kumar rows. MicroNet/Kumar is a cross-dataset evaluation of the CoNSeP checkpoint. The 14-image nucleus cohorts are small.

Regenerate this file with `python -m rankseg_benchmark.tiatoolbox.table --output rankseg_benchmark/tiatoolbox/RESULTS.md`. Use `--from-artifacts --check-artifacts` after full inference/evaluation to verify the tracked per-case evidence against the ignored run outputs.
