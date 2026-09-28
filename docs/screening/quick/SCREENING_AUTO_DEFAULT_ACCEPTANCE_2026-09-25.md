# Auto-default acceptance — 2026-09-25

**Historical experiment:** after these measurements, the release decision was
changed to keep `safe_screening=False` as the default. Auto remains opt-in;
the results below describe the preserved auto-default snapshot, not the current
default. Recorded results and raw artifacts have not been rewritten.

## Completed mini-benchmark

All **6,012 cached inputs** completed without OOM: VOC / Cityscapes / ADE20K
use the first 100 cached images each; KiTS uses all 5,712 slices, 210 cases and
five folds. The natural-image sets are **subsets**, not full published datasets.
The [nnU-Net representative-volume acceptance](../nnunet/SCREENING_AUTO_DEFAULT_ACCEPTANCE_2026-09-25.md)
is reported separately (41 volumes, including the Liver regression); this run
does not include MONAI probability caches, new inference or checkpoint changes.

The local candidate defaults to `safe_screening="auto"` for RMA Dice, smooth=0.
CPU screens; CUDA screens with Triton and at least **1,280,000 total probability
values B*C*D**, otherwise it uses optimized full sort. `False` retains the
original path; `True` forces screening without size or candidate-density retries.
This is a working-tree candidate, **not a claim about the published 0.0.6 wheel**.

### Protocol

- Frozen RankSEG package: `/tmp/rankseg-auto-default-snapshot.9I62kX`, also
  preserved in `artifacts/screening-auto-default-2026-09-25/snapshot/`; source
  and cache SHA-256 hashes are stored with the raw reports.
- RTX 3090, Python 3.10.12, Torch 2.11.0+cu130, Triton 3.6.0; float32, B=1,
  four CPU threads. RMA Dice, smooth=0, pruning=0.5, max-score assignment.
- Two warm-ups, seven synchronized measurements in rotating method order;
  reported latency is the mean of per-input medians. GPU jobs run serially.
- Timing includes benchmark channel routing and output conversion, but excludes
  loading/transfers, compilation, metrics, numerical oracles and diagnostics.
  Auxiliary dispatch-control setup and teardown are **outside timing**.
- Memory is mean incremental peak PyTorch allocated GPU memory, including
  routing temporaries and output, excluding resident original probabilities,
  reserved memory and CUDA context. It is not whole-model memory or inference
  speed. Routing copies explain part of the difference from core-only memory
  measurements in the real-scale report.
- Natural-image scores average over GT-present classes per image, then images.
  KiTS uses the repository's foreground slice → case → fold averaging and empty
  slice convention, **not 3D volume Dice**. RankSEG receives foreground only;
  argmax receives both original channels. IoU is measured, not separately optimized.

### Quality, latency and memory

Scores are percentages; times are ms/input and memory is MiB. `Auto` is the new
default, `Off` is explicit False, and `Forced` is explicit True. Optimized full
sort is a separate auxiliary reference, not a replacement for Off.

| Dataset | Method | Dice | IoU | Mean ms | Mean peak MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| VOC | Argmax | 88.9594 | 85.5419 | 0.0544 | 2.00 |
| VOC | Off | 89.3309 | 85.9592 | 3.1697 | 256.76 |
| VOC | Optimized full | 89.3312 | 85.9597 | 2.1505 | 256.76 |
| VOC | Auto | 89.3311 | 85.9595 | 0.6201 | 29.50 |
| VOC | Forced | 89.3311 | 85.9595 | 0.6028 | 29.50 |
| Cityscapes | Argmax | 81.3147 | 74.4120 | 0.2281 | 16.00 |
| Cityscapes | Off | 81.9421 | 74.9180 | 19.6212 | 1688.00 |
| Cityscapes | Optimized full | 81.9433 | 74.9197 | 12.8345 | 928.00 |
| Cityscapes | Auto | 81.9418 | 74.9181 | 1.9817 | 208.00 |
| Cityscapes | Forced | 81.9418 | 74.9181 | 1.9633 | 208.00 |
| ADE20K | Argmax | 64.3637 | 57.1359 | 0.2470 | 2.40 |
| ADE20K | Off | 65.4416 | 58.0087 | 23.2850 | 1981.53 |
| ADE20K | Optimized full | 65.4419 | 58.0090 | 15.2676 | 1619.30 |
| ADE20K | Auto | 65.4419 | 58.0088 | 1.8323 | 230.88 |
| ADE20K | Forced | 65.4419 | 58.0088 | 1.8163 | 230.88 |
| KiTS | Argmax | 61.1566 | 54.1836 | 0.0212 | 1.125 |
| KiTS | Off | 63.5350 | 56.2096 | 0.3038 | 7.313 |
| KiTS | Optimized full | 63.5350 | 56.2096 | 0.2351 | 4.680 |
| KiTS | Auto | 63.5350 | 56.2096 | 0.2268 | 4.680 |
| KiTS | Forced | 63.5350 | 56.2095 | 0.3783 | 1.829 |

| Dataset | Auto speedup vs Off / optimized full | Peak reduction vs Off / optimized full | Potential H+L / active | Actual auto sort avoidance / active |
| --- | ---: | ---: | ---: | ---: |
| VOC | 5.11× / 3.47× | 88.51% / 88.51% | 99.2785% | 99.2785% |
| Cityscapes | 9.90× / 6.48× | 87.68% / 77.59% | 99.7628% | 99.7628% |
| ADE20K | 12.71× / 8.33× | 88.35% / 85.74% | 98.7293% | 98.7293% |
| KiTS | 1.34× / 1.04× | 36.01% / 0.00% | 99.9135% | 0.0000% |

All 5,712 KiTS inputs bypass screening in auto mode. Its gain over Off is from
optimized full sort, not screening. Auto and optimized-full are the same route
here; their small timing difference is not a distinct algorithmic speedup.
Forcing screening lowers memory further but is materially slower.
H/L percentages count class-probability entries in active classes, not unique
pixels, total memory saved, or already-pruned classes. There were no candidate
count/fraction retries.

### Correctness and compatibility

- **24,048 / 24,048** binary-objective checks passed against independent
  exhaustive float64 full-prefix oracles, using the unchanged four-float32-eps
  regression budget. The maximum was **2.2564 eps** (KiTS Off/optimized/auto).
- Every default prediction was compared bitwise with explicit auto: all 6,012
  matched. Shared probability tensors were checked for non-mutation.
- Independent NumPy confusion counts, repository accumulators, metric
  aggregation, timing medians, memory aggregates and numerical status all passed.
- Case/image identities and input/label hashes match the earlier complete-cache
  acceptance. **Argmax and explicit-Off confusion counts match that earlier
  run for every input**, not just their dataset averages.
- Auto is **not** generally bitwise equal to Off. The largest absolute dataset
  mean Dice change is **0.000355 percentage points**. The worst per-image Dice
  decline is **0.0897 pp**, on Cityscapes. KiTS auto masks are exactly equal to Off.
- Epsilon-scale binary-objective differences do not bound final multiclass
  label differences or ground-truth metric changes. These checks are empirical
  acceptance, not a proof for all inputs or a universal 0.8× speed guarantee.

### Artifacts and reproduction

Raw results, per-input journals and `independent-audit.json` are in the ignored
local directory `artifacts/screening-auto-default-2026-09-25/`. Old benchmark
artifacts and published tables were not overwritten.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ../env/bin/python -B \
  scripts/benchmark_screening_acceptance.py \
  --rankseg-path artifacts/screening-auto-default-2026-09-25/snapshot \
  --screening-mode auto --warmup 2 --repeats 7 \
  --output-dir artifacts/screening-auto-default-rerun
```

Use a fresh output directory when rerunning. No checkpoint download or inference
is required. Raw artifacts/snapshots are local and ignored by Git.

## Independent real-image CUDA scale check

Rechecked the fixed **1,280,000-entry** rule without retuning it. Used five
sources per dataset, six spatial sizes each: D = 8,192 / 12,288 / 32,768 /
65,536 / 98,304 / 131,072. These are nested uniform pixel subsets of real cached
probabilities, not synthesized probabilities, interpolation or new inference.
Natural-image source rows 10/30/50/70/90 are disjoint from the earlier
threshold-fitting rows 5/15/…/95. KiTS uses one source slice per fold. Selection
uses seed 20260926, not outcomes. These configurations test scaling, not image
quality estimates; the complete-image quality results remain above.

All **120 configurations / 480 objective checks** passed (maximum 1.00145 eps).
Production auto masks exactly match the route actually selected, in every
round. Input/output hashes, dispatch, candidate partitions, timing medians and
complete configuration coverage passed the independent report audit.

Three rounds, two warm-ups, five timings per method per round; shuffled job and
rotating method order. For each configuration, use the median of round timing
medians and maximum observed incremental peak. Table speedups are ratios of
means over the indicated configurations. Unlike the main table, these are
**core-only** decoder measurements without benchmark channel-routing copies.

| Dataset | Auto route | Configurations | Speedup vs optimized full | Slowest configuration speedup | Peak reduction vs optimized full |
| --- | --- | ---: | ---: | ---: | ---: |
| VOC | Screening | 15 | 1.669× | 1.020× | 96.81% |
| VOC | Optimized full | 15 | 0.983× | 0.965× | 0.00% |
| Cityscapes | Screening | 10 | 1.077× | 0.844× | 96.71% |
| Cityscapes | Optimized full | 20 | 0.988× | 0.957× | 0.00% |
| ADE20K | Screening | 25 | 3.636× | 0.899× | 96.46% |
| ADE20K | Optimized full | 5 | 0.995× | 0.990× | 0.00% |
| KiTS | Optimized full | 30 | 0.955× | 0.910× | 0.00% |

No configuration fell below the agreed **0.8×** speed tradeoff in this run;
the slowest screened configuration was **0.844×**. Some screened inputs are
slower: this is a memory-oriented empirical policy, not a universal speed
guarantee. Bypassed rows run the same optimized-full algorithm; small timing
differences include public auto dispatch and measurement variation.

Artifacts: `artifacts/screening-auto-default-scales-2026-09-25/{results,summary}.json`
and `timings.jsonl`. Core sources match the frozen acceptance package.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ../env/bin/python -B \
  scripts/calibrate_screening_cuda.py --production-screening auto \
  --samples-per-dataset 5 --sampling-seed 20260926 \
  --dims 8192 12288 32768 65536 98304 131072 \
  --rounds 3 --repeats 5 --warmup 2 \
  --output-dir artifacts/screening-auto-default-scales-rerun
```

The scale runner reads the current adjacent RankSEG checkout; verify its source
hashes against the frozen snapshot before comparing a rerun.

## CPU auto-default check

Intel Core i9-12900K, four threads, Torch 2.11.0, one warm-up and three timings
per input. Used the first ten cached images from each natural dataset and two
uniformly spaced slices per KiTS fold (**40 inputs** total). These are subset
results, not the 6,012-input quality averages above. KiTS quality here is a
sampled-slice mean, not the complete-cache case/fold aggregate. Eight of its ten
selected foreground rows are active; whole-class-pruned rows remain included.

CPU auto uses screening for every active input, with no scale cutoff. Both
methods are normal public decoders, without private dispatch overrides; timing
includes channel routing. Memory was **not measured** in this CPU run.

| Dataset | Inputs | Argmax ms | Off → Auto ms | Auto speedup | Auto Dice (%) | Auto IoU (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VOC | 10 | 17.8722 | 113.6775 → 45.3344 | 2.51× | 96.8213 | 94.0889 |
| Cityscapes | 10 | 144.1628 | 901.6514 → 512.9002 | 1.76× | 82.2751 | 75.9067 |
| ADE20K | 10 | 35.0065 | 884.8628 → 357.4886 | 2.48× | 61.3551 | 53.7316 |
| KiTS | 10 | 9.2581 | 5.3220 → 0.3274 | 16.25× | 63.6764 | 56.1066 |

All **80 objective checks** passed, maximum **1.11372 eps**. Independent
recomputation verified unique input coverage, every timing median, overlap
scores and dataset/screening aggregates. Inputs were checked for non-mutation.
Artifacts retain both methods' quality and per-input mask differences:
`artifacts/screening-auto-default-cpu-2026-09-25/{summary.json,records.jsonl}`.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ../env/bin/python -B \
  scripts/benchmark_screening_cpu.py \
  --rankseg-path artifacts/screening-auto-default-2026-09-25/snapshot \
  --screening-mode auto --threads 4 --natural-count 10 --kits-per-fold 2 \
  --warmup 1 --repeats 3 \
  --output-dir artifacts/screening-auto-default-cpu-rerun
```

## Test and distribution acceptance

- Core CPU suite: **1,082 passed, 1,525 skipped**; CPU coverage **92.65%**.
  CUDA-only and optional-data tests are not counted as CPU passes.
- Core CUDA suite, Torch 2.11.0 / Triton 3.6.0, including opt-in large-index
  regressions: **2,599 passed, 8 skipped**; coverage **85.99%**.
- Additional Torch 2.8.0 / Triton 3.4.0 CUDA algorithm and screening suites,
  including opt-in large-index regressions: **2,059 passed, 6 skipped**.
  This environment lacks `torchmetrics`, so the attempted whole-suite run
  stopped at collection of two test modules. The successful rerun explicitly
  selects `test_rankseg_algo.py` and `test_screening*.py`; it is **not** a second
  whole-suite pass. No dependency installation or test weakening was used.
- Mini-benchmark harness, including explicit auto CPU routing and CUDA scale
  end-to-end tests: **242 passed**.
- nnU-Net screening/acceptance/audit/subset harness, Torch 2.8.0 / Triton 3.4.0:
  **207 passed**.
- Sphinx HTML build with warnings as errors: passed.
- Local wheel and sdist build and distribution-content validation: passed.
  Isolated installed-wheel smoke outside the checkout: **27 prediction checks
  on CPU and 27 on CUDA**, passed. CUDA smoke observed actual Triton statistics,
  packing, gather, scoring, scatter and multiclass-assignment calls, including
  the 1,280,000-entry auto boundary.
- The omitted default and explicit auto have regression coverage across direct,
  functional and module APIs, CPU/CUDA, layouts/dtypes and dispatch routes.
  IoU, positive smooth and other solver paths retain their existing behavior;
  `safe_screening=False` remains the original-path compatibility escape hatch.

These local runs use **Python 3.10.12**, not all supported Python versions;
the remaining version matrix still belongs in CI. The build uses the current
working-tree version metadata **0.0.6** only for local artifact testing. Choose
and bump the next release version before publishing; no upload, commit or push
was performed by this acceptance.
