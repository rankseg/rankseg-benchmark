"""CPU benchmark selection, timing, routing and objective regression checks."""
import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("cpu_bench", ROOT / "scripts/benchmark_screening_cpu.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


@pytest.fixture(scope="module")
def helpers():
    return bench.acceptance.load_helpers(Path(__import__("rankseg").__file__).resolve().parents[1], ROOT)


@pytest.mark.parametrize("total,count", [(1, 1), (5, 1), (5, 8), (100, 10), (1237, 10)])
@pytest.mark.parametrize("spaced", [False, True])
def test_selection_is_unique_bounded_and_deterministic(total, count, spaced):
    rows = bench.selected_rows(total, count, spaced)
    assert rows == sorted(set(rows))
    assert len(rows) == min(total, count)
    assert rows[0] == 0 and rows[-1] < total
    if spaced and len(rows) > 1:
        assert rows[-1] == total - 1
        steps = [b - a for a, b in zip(rows, rows[1:])]
        assert max(steps) - min(steps) <= 1
    elif not spaced:
        assert rows == list(range(min(total, count)))


@pytest.mark.parametrize("total,count", [(0, 1), (1, 0), (-1, 3)])
def test_bad_counts_rejected(total, count):
    with pytest.raises(ValueError):
        bench.selected_rows(total, count)


def test_timing_rotates_order_and_excludes_output_collection(monkeypatch):
    seen = []
    ticks = iter(range(100))
    monkeypatch.setattr(bench.time, "perf_counter", lambda: next(ticks))
    calls = {name: lambda name=name: seen.append(name) or torch.tensor(1) for name in ("a", "b", "c")}
    values, predictions = bench.measure(calls, warmup=1, repeats=3, index=0)
    assert seen == list("abc" + "abc" + "bca" + "cab" + "abc")
    assert set(predictions) == set(calls)
    assert all(v == {"samples_ms": [1000., 1000., 1000.], "median_ms": 1000.} for v in values.values())


@pytest.mark.parametrize("name", ["kits", "pascal_voc", "cityscapes", "ade20k"])
@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("screening_mode", [True, "auto"])
def test_routing_and_both_objectives(helpers, name, threads, screening_mode):
    oracle, counts = helpers
    from rankseg_benchmark.datasets import REGISTRY
    from rankseg_benchmark.runner import _rankseg_channels, _rankseg_output_mode, _rankseg_predict
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        spec = REGISTRY[name]
        mode = _rankseg_output_mode(spec, "RMA")
        channels = _rankseg_channels(spec, mode)
        p = torch.rand((1, spec.num_classes, 9, 11), generator=torch.Generator().manual_seed(23))
        p = p.pow(8)
        p /= p.sum(1, keepdim=True)
        before = bench.tensor_hash(p)
        selected = p[:, channels]
        optimum = oracle.bounded_binary_oracle(selected)
        for method in ("full", "screened"):
            decoder = bench.acceptance.make_decoder(method, mode, oracle, screening_mode)
            binary = bench.acceptance.make_decoder(method, "multilabel", oracle, screening_mode)
            mask = binary.predict(selected)
            oracle.objective_regret(selected, mask, optimum)
            pred = _rankseg_predict(decoder, p, spec=spec, rankseg_output_mode=mode)
            assert pred.shape == (1, 9, 11) and pred.dtype == torch.int64
            if name == "kits":
                assert channels == (1,) and mode == "multilabel"
                assert torch.equal(pred, mask[:, 0].long())
        assert bench.tensor_hash(p) == before
        rows = counts.screening_counts(selected, channels)["rows"]
        assert sum(r["total_entries"] for r in rows) == selected.numel()
    finally:
        torch.set_num_threads(previous)


def test_objective_oracle_rejects_bad_mask(helpers):
    oracle, _ = helpers
    p = torch.tensor([[[.99, .8, .7, .001]]])
    with pytest.raises(oracle.ObjectiveBudgetExceeded):
        oracle.objective_regret(p, torch.zeros_like(p, dtype=torch.bool), oracle.bounded_binary_oracle(p))


@pytest.mark.parametrize("scenario", ["sparse", "uniform", "dense_candidates", "all_pruned"])
@pytest.mark.parametrize("channels", [1, 3])
def test_synthetic_profiles_have_expected_partition(helpers, scenario, channels):
    oracle, counts = helpers
    profile_spec = importlib.util.spec_from_file_location(
        "cpu_profiles", ROOT / "scripts/benchmark_screening_cpu_synthetic.py")
    profiles = importlib.util.module_from_spec(profile_spec)
    profile_spec.loader.exec_module(profiles)
    p = profiles.make_probabilities(scenario, channels, 4096)
    assert torch.equal(p, profiles.make_probabilities(scenario, channels, 4096))
    measured = counts.summarize([counts.screening_counts(p, list(range(channels)))])
    if scenario == "dense_candidates":
        assert measured["fallback_rows"] == 0
        assert measured["actual_sort_entries"] == measured["undecided"]
        assert measured["actual_sort_entries"] == p.numel() - channels
    elif scenario == "all_pruned":
        assert measured["active_rows"] == 0
    elif scenario == "sparse":
        assert measured["undecided"] == 0
        assert measured["screening_fraction_active"] == 1.
    optimum = oracle.bounded_binary_oracle(p)
    for method in ("full", "screened"):
        decoder = bench.acceptance.make_decoder(method, "multilabel", oracle)
        oracle.objective_regret(p, decoder.predict(p), optimum)


def test_streaming_selection_decodes_only_requested_rows(helpers, monkeypatch):
    from types import SimpleNamespace
    import pyarrow as pa
    from rankseg_benchmark import datasets
    calls = []
    class FakeParquet:
        def __init__(self, path):
            pass
        def iter_batches(self, batch_size):
            for index in range(10):
                yield pa.RecordBatch.from_pylist([{"probs": index, "label": index}])
    def decode(value, **kwargs):
        calls.append(value)
        return torch.full((2, 3, 5), float(value))
    monkeypatch.setattr(bench.pq, "ParquetFile", FakeParquet)
    monkeypatch.setattr(datasets, "_decode_probs", decode)
    monkeypatch.setattr(datasets, "_decode_label", lambda value, **kwargs: torch.zeros((3, 5), dtype=torch.long))
    dataset = SimpleNamespace(num_classes=2, output_mode="multiclass", case_id_key=None, eval_unit="image")
    source = {"path": "unused", "rows": 10, "fold": None}
    rows = list(bench.selected_samples(dataset, source, [0, 4, 9]))
    assert calls == [0, 4, 9]
    assert [row[2]["source_row"] for row in rows] == calls
    with pytest.raises(ValueError, match="Invalid sample"):
        list(bench.selected_samples(dataset, source, [4, 4]))
    with pytest.raises(ValueError, match="Cache ended"):
        list(bench.selected_samples(dataset, {**source, "rows": 12}, [11]))


def test_summary_uses_ratio_of_means_and_preserves_failures():
    from types import SimpleNamespace
    rows = []
    for index, (full_ms, screened_ms) in enumerate([(4., 2.), (20., 10.)]):
        methods = {}
        for method, ms in [("argmax", 1.), ("full", full_ms), ("screened", screened_ms)]:
            methods[method] = dict(median_ms=ms, scores={"dice": .5, "iou": 1/3},
                                   objective={"status": "failed" if index == 1 else "passed",
                                              "max_regret_eps": 5. if index == 1 else .1})
        rows.append(dict(methods=methods, different_pixels=index, prediction_pixels=100, screening_counts={}))
    # Different per-sample speedups: mean ratios 5.1, ratio of means 22 / 12.
    rows[0]["methods"]["full"]["median_ms"] = 20.
    rows[1]["methods"]["full"]["median_ms"] = 2.
    value = bench.summarize(rows, SimpleNamespace(summarize=lambda _: {}))
    assert value["speedup"] == pytest.approx(22 / 12)
    assert value["methods"]["screened"]["objective_passed"] == 1
    assert value["methods"]["screened"]["max_regret_eps"] == 5.
    assert value["different_pixels"] == 1 and value["total_pixels"] == 200
