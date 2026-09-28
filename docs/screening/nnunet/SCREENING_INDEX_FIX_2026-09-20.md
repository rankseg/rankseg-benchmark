# Very-wide CUDA row-indexing regression

The first full Liver screening run stopped at `liver_115`, whose probability tensor
has shape `[1, 3, 846, 512, 512]`: **221,773,824 voxels per class**. The error was
`CUDA error: invalid configuration argument` while compacting active class rows,
not a CUDA out-of-memory exception or a failed objective check.

## Reproduction and change

On RTX 3090 / PyTorch 2.8.0+cu128, the error was reproduced without RankSEG or any
patient data: allocate a float32 `[3, 221773824]` CUDA tensor and read rows `[0, 2]`
with advanced indexing. Each operation was checked in a fresh process with
`CUDA_LAUNCH_BLOCKING=1`. The advanced read failed; `index_select(0, indices)`
succeeded and copied every value correctly. Both advanced write and `index_copy_`
passed in that isolated test. The precise internal PyTorch launch-limit mechanism
has not been established; this is not evidence of probability corruption or OOM.

The RankSEG change is restricted to row copying:

- Use `index_select` to gather active probability rows.
- Use `index_copy_` to restore compacted masks in both single-row and multi-row paths.
- Use the same gather/scatter operations for the partial full-sort fallback.

Indices come from `nonzero` and are unique and ascending. Row order and values are
preserved. All-active fast paths, original probabilities, screening certificates,
score arithmetic, pruning, tie rules and multiclass assignment are unchanged.
The default `safe_screening=False` path is unchanged.

## Checks after the fix

| Check | Result |
| :--- | :--- |
| New row-indexing tests, PyTorch 2.8 / CUDA | 42 passed, including two full-size synthetic cases |
| All RankSEG tests, PyTorch 2.11 / CUDA | 1,083 passed, 10 skipped; 92.83% coverage |
| Screening suites, PyTorch 2.8 / CUDA | 263 passed, 2 skipped (large tests are opt-in and were run separately) |
| nnU-Net benchmark suite, CUDA | 73 passed |
| Before/after CPU/CUDA comparisons | 192 exact output matches; input unchanged |

New tests cover float32/float64, strided inputs, one/several/all active rows,
multilabel/multiclass output and partial fallback. Small tests reject advanced
whole-row reads/writes explicitly, so the regression is covered in CPU-only CI too.
Large-memory tests additionally execute the actual failing row width. From the
RankSEG repository, on a machine with sufficient free CUDA memory:

```bash
RANKSEG_TEST_LARGE_CUDA=1 python -m pytest -q --no-cov \
  tests/test_screening_row_indexing.py
```

### Real failure-case validation

Both original whole-volume caches completed through optimized full sort and screened
binary/multiclass decoding. Binary masks were checked against an independent float64
full-prefix oracle, with the unchanged four-float32-eps regression budget.

| Case | Voxels per class | Optimized-full regret (eps) | Screened regret (eps) |
| :--- | ---: | ---: | ---: |
| `liver_115` (original failure) | 221,773,824 | 1.5731 | 0.4712 |
| `liver_127` (largest in cohort) | 258,736,128 | 2.1166 | 0.3277 |

These checks are not a full-cohort accuracy report. Numerical objective agreement
also does not mean screened and unscreened masks are identical. The **indexing fix**
is an exact-copy change; screening itself remains a numerically different evaluation
of the same objective.

### Before/after performance

Same GPU / PyTorch 2.8; whole multiclass calls, five warmups and 31 synchronized
timed repeats, alternating old/new order. Fixed synthetic inputs have three channels,
with one, two or three active classes. Timings are medians, in milliseconds.

| Voxels/class | Active classes | Before | After |
| ---: | ---: | ---: | ---: |
| 262,145 | 1 | 0.527 | 0.527 |
| 262,145 | 2 | 0.760 | 0.764 |
| 262,145 | 3 | 0.731 | 0.732 |
| 1,048,576 | 1 | 0.902 | 0.902 |
| 1,048,576 | 2 | 1.166 | 1.170 |
| 1,048,576 | 3 | 1.173 | 1.173 |
| 16,777,216 | 1 | 7.942 | 7.923 |
| 16,777,216 | 2 | 10.284 | 10.243 |
| 16,777,216 | 3 | 10.717 | 10.716 |

Observed changes are within approximately ±0.5%, not evidence of a universal
performance bound. The old operation fails at the Liver-sized width, so no meaningful
before/after latency ratio exists for that failed case.

Completed diagnostic JSONs, scripts and pre-fix source snapshots are retained locally
in the Git-ignored `outputs/screening-index-fix/` directory. The scripts retain this
workspace's absolute paths. The failed three-case Liver report is preserved at
`outputs/screening-full-cohorts/liver.json`; the fresh 131-case rerun writes to
`outputs/screening-full-cohorts-indexfix/liver.json`. A partial report must never be
presented as a completed benchmark. No checkpoint, cached probability, label or
published Full-16 result was modified.
