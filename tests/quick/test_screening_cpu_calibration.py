"""Calibration must preserve probabilities and not manufacture a crossover."""
import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("cpu_calibration", ROOT / "scripts/calibrate_screening_cpu.py")
calibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibration)


@pytest.mark.parametrize("batch,channels", [(1, 1), (1, 3), (4, 3), (1, 16)])
@pytest.mark.parametrize("dim", [1, 2, 7, 24])
def test_subsampling_preserves_exact_probabilities(batch, channels, dim):
    p = torch.rand((batch, channels, 4, 6), generator=torch.Generator().manual_seed(9))
    before = p.clone()
    actual = calibration.spatial_subset(p, dim)
    indices = [i * 23 // (dim - 1) for i in range(dim)] if dim > 1 else [0]
    assert len(indices) == len(set(indices))
    assert actual.shape == (batch, channels, dim)
    assert torch.equal(actual, p.flatten(2)[..., indices])
    assert torch.equal(before, p)
    assert actual.is_contiguous()


@pytest.mark.parametrize("dim", [0, -1, 25])
def test_invalid_subset_size_rejected(dim):
    with pytest.raises(ValueError):
        calibration.spatial_subset(torch.ones(1, 3, 24), dim)


def rounds(ratios):
    return [{"full": {"median_ms": ratio}, "screened": {"median_ms": 1.}} for ratio in ratios]


@pytest.mark.parametrize("ratios,expected", [
    ([1.1, 1.2, 1.3], "win"), ([.5, .9, .7], "loss"),
    ([.8, 1.3, 1.2], "mixed_or_near"), ([1., 1., 1.], "mixed_or_near"),
    ([1.05, 1.2, 1.3], "mixed_or_near"), ([.95, .8, .8], "mixed_or_near"),
])
def test_classification_requires_agreement_across_rounds(ratios, expected):
    value = calibration.timing_summary(rounds(ratios))
    assert value["classification"] == expected
    assert value["round_speedups"] == ratios


def test_summary_reports_absolute_time_as_well_as_ratio():
    value = calibration.timing_summary(rounds([.8, 1.5, 1.2]))
    assert value["speedup"] == 1.2
    assert value["saved_us"] == pytest.approx(200.)


def test_crossover_requires_every_larger_tested_size_to_win():
    rows = [{"dim": dim, "timing": {"classification": label}}
            for dim, label in [(512, "win"), (1024, "loss"), (2048, "win"), (4096, "win")]]
    assert calibration.stable_suffix(rows, [512, 1024, 2048, 4096]) == 2048
    rows.append({"dim": 4096, "timing": {"classification": "loss"}})
    assert calibration.stable_suffix(rows, [512, 1024, 2048, 4096]) is None
    with pytest.raises(ValueError, match="Incomplete"):
        calibration.stable_suffix(rows, [512, 1024, 2048, 4096, 8192])


@pytest.mark.parametrize("shape", calibration.SHAPES)
@pytest.mark.parametrize("scenario", calibration.profiles.SCENARIOS)
def test_synthetic_inputs_and_both_objectives(shape, scenario):
    oracle, counts = calibration.bench.acceptance.load_helpers(Path(__import__("rankseg").__file__).resolve().parents[1], ROOT)
    b, c = shape
    profile = dict(kind="synthetic", batch=b, channels=c, scenario=scenario, seed=3401)
    p = calibration.make_input(profile, 512)
    assert p.shape == (b, c, 512)
    assert torch.equal(p, calibration.make_input(profile, 512))
    optimum = oracle.bounded_binary_oracle(p)
    for method in ("full", "screened"):
        decoder = calibration.bench.acceptance.make_decoder(method, "multilabel", oracle)
        oracle.objective_regret(p, decoder.predict(p), optimum)
    measured = counts.summarize([counts.screening_counts(p[i:i+1], list(range(c))) for i in range(b)])
    assert measured["total_entries"] == p.numel()
    if scenario == "dense_candidates":
        assert measured["fallback_rows"] == 0
        assert measured["undecided"] > .75 * measured["active_entries"]
        assert measured["actual_sort_entries"] == measured["undecided"]


@pytest.mark.parametrize("total,count", [(100, 10), (1142, 2), (9, 9), (7, 1)])
def test_stratified_cache_selection_is_unique_and_inside_each_stratum(total, count):
    rows = calibration.stratum_centers(total, count)
    assert len(rows) == len(set(rows)) == count
    for i, row in enumerate(rows):
        assert i * total / count <= row < (i + 1) * total / count
    assert rows == sorted(rows)
    assert calibration.stratum_centers(100, 10) == list(range(5, 100, 10))


def test_real_only_source_selection_covers_every_kits_fold_and_no_synthetic_data():
    natural = [{"rows": 100, "fold": None}]
    selected = calibration.selected_source_rows("ade20k", natural, True, 10)
    assert selected == [(natural[0], list(range(5, 100, 10)))]
    kits = [{"rows": 1100 + i, "fold": i} for i in range(5)]
    selected = calibration.selected_source_rows("kits", kits, True, 10)
    assert [s["fold"] for s, _ in selected] == list(range(5))
    assert all(len(indices) == 2 for _, indices in selected)
    with pytest.raises(ValueError, match="divisible"):
        calibration.selected_source_rows("kits", kits, True, 11)
    with pytest.raises(ValueError, match="Not enough"):
        calibration.selected_source_rows("ade20k", natural, True, 101)


def test_uniform_sampling_is_reproducible_nested_and_channel_aligned():
    p = torch.arange(3 * 101, dtype=torch.float32).reshape(1, 3, 101) / 303
    order, seed = calibration.sampling_order("sample-1", 101, 20260925)
    repeated, repeated_seed = calibration.sampling_order("sample-1", 101, 20260925)
    different, _ = calibration.sampling_order("sample-2", 101, 20260925)
    assert torch.equal(order, repeated) and seed == repeated_seed
    assert not torch.equal(order, different)
    assert torch.equal(order.sort().values, torch.arange(101))
    profile = dict(kind="real_derived", probs=p, draw_order=order)
    before = p.clone()
    for dim in (1, 7, 23, 101):
        selected = calibration.make_input(profile, dim)
        assert torch.equal(selected, p[..., order[:dim].sort().values])
        assert selected.is_contiguous()
    assert set(order[:7].tolist()) < set(order[:23].tolist())
    assert torch.equal(calibration.make_input(profile, 101), p)
    assert torch.equal(p, before)
    assert "draw_order" not in calibration.profile_metadata(profile)
    assert "probs" not in calibration.profile_metadata(profile)
    with pytest.raises(ValueError):
        calibration.make_input(profile, 102)


def test_real_only_loader_never_calls_synthetic_generator(monkeypatch):
    from types import SimpleNamespace
    calibration.bench.acceptance.load_helpers(Path(__import__("rankseg").__file__).resolve().parents[1], ROOT)
    def sources(dataset, *args):
        entries = [{"rows": 100, "fold": i if dataset == "kits" else None}
                   for i in range(5 if dataset == "kits" else 1)]
        return {}, entries
    def samples(spec, source, indices):
        for row in indices:
            p = torch.full((1, spec.num_classes, 4, 6), .3)
            yield p, torch.zeros((1, 4, 6)), dict(source_row=row, fold=source["fold"], case_id=f"case-{row}")
    monkeypatch.setattr(calibration.bench.acceptance, "cache_sources", sources)
    monkeypatch.setattr(calibration.bench, "selected_samples", samples)
    monkeypatch.setattr(calibration.profiles, "make_probabilities", lambda *a: pytest.fail("Synthetic data generated"))
    args = SimpleNamespace(real_only=True, samples_per_dataset=10, sampling="uniform", sampling_seed=20260925,
                           natural_cache=Path("unused"), kits_root=Path("unused"))
    inputs, _ = calibration.load_profiles(args)
    assert len(inputs) == 40 and all(p["kind"] == "real_derived" for p in inputs)
    assert len({p["id"] for p in inputs}) == 40
    for p in inputs:
        assert calibration.make_input(p, 8).shape == (1, p["channels"], 8)
        if p["dataset"] == "kits":
            assert p["channels"] == 1 and p["mode"] == "multilabel"


def test_run_records_unsupported_sizes_without_upsampling(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace
    calibration.bench.acceptance.load_helpers(Path(__import__("rankseg").__file__).resolve().parents[1], ROOT)
    p = torch.rand((1, 3, 24), generator=torch.Generator().manual_seed(91))
    profile = dict(id="test-source", kind="real_derived", dataset="test", probs=p,
                   batch=1, channels=3, mode="multiclass")
    monkeypatch.setattr(calibration, "load_profiles", lambda args: ([profile], {}))
    args = SimpleNamespace(output_dir=tmp_path / "run", dims=[4, 64], threads=[1], rounds=2,
                           warmup=0, repeats=2, real_only=True, sampling="spaced", sampling_seed=20260925,
                           samples_per_dataset=10)
    old_threads = torch.get_num_threads()
    try:
        calibration.run(args)
    finally:
        torch.set_num_threads(old_threads)
    result = json.loads((args.output_dir / "results.json").read_text())
    assert result["complete"] and len(result["records"]) == 1
    assert result["profiles"][0]["tested_dims"] == [4]
    assert result["profiles"][0]["unavailable_dims"] == [64]
    assert len(result["records"][0]["rounds"]) == 2
    assert all(v["status"] == "passed" for v in result["records"][0]["objectives"].values())
    with pytest.raises(FileExistsError):
        calibration.run(args)
