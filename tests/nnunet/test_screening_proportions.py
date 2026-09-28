"""Untimed screening diagnostics must agree with the actual sorting workspaces."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


spec = importlib.util.spec_from_file_location(
    "screening_proportions", Path(__file__).resolve().parents[2] / "scripts/nnunet/collect_screening_proportions.py")
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))]


def example(device="cpu", dtype=torch.float32):
    return torch.tensor([[[0.9, 0.4, 0.2, 0, 0, 0, 0, 0],
                          [0.5, 0.1, 0, 0, 0, 0, 0, 0],
                          [0.9, 0.45, 0.35, 0.34, 0.2, 0, 0, 0]]], device=device, dtype=dtype)


@pytest.mark.parametrize("device", DEVICES)
def test_known_certificates_and_class_pruning_are_separate(device):
    probs = example(device)
    before = probs.clone()
    result = stats.screening_counts(probs, (0, 2, 7))
    a, pruned, b = result["rows"]
    assert (a["forced_positive"], a["forced_negative"], a["undecided"]) == (1, 6, 1)
    assert (b["forced_positive"], b["forced_negative"], b["undecided"]) == (1, 4, 3)
    assert pruned["class_pruned_entries"] == 8
    assert pruned["screening_fraction_active"] is None
    summary = stats.summarize([result])
    assert summary["screening_fraction_active"] == 12 / 16
    assert summary["effective_sort_avoidance_active"] == 12 / 16
    assert summary["class_pruned_fraction_all"] == 8 / 24
    assert summary["actual_sort_fraction_all"] == 4 / 24
    assert set(summary["by_label"]) == {"0", "2", "7"}
    assert torch.equal(probs, before)


@pytest.mark.parametrize("device", DEVICES)
def test_legacy_fallback_reporting_and_padding_is_not_real_candidates(device, monkeypatch):
    from rankseg import _screening
    monkeypatch.setattr(_screening, "_MAX_CANDIDATE_ELEMENTS", 2, raising=False)
    result = stats.screening_counts(example(device), (0, 1, 2))
    summary = stats.summarize([result])
    assert summary["undecided"] == 4
    assert summary["screening_fraction_active"] == 12 / 16
    assert summary["effective_sort_avoidance_active"] == 7 / 16
    assert summary["actual_sort_entries"] == summary["sort_workspace_elements"] == 9
    assert summary["fallback_rows"] == 1
    monkeypatch.setattr(_screening, "_MAX_CANDIDATE_ELEMENTS", 100)
    monkeypatch.setattr(_screening, "_MAX_PADDING_RATIO", 100)
    padded = stats.summarize([stats.screening_counts(example(device), (0, 1, 2))])
    assert padded["actual_sort_entries"] == 4
    assert padded["sort_workspace_elements"] == 6


@pytest.mark.parametrize("device", DEVICES)
def test_all_pruned_has_no_screening_denominator(device):
    result = stats.summarize([stats.screening_counts(torch.zeros(1, 3, 8, device=device), (0, 1, 2))])
    assert result["screening_fraction_active"] is None
    assert result["effective_sort_avoidance_active"] is None
    assert result["actual_sort_fraction_all"] == 0
    assert result["class_pruned_fraction_all"] == 1
    assert result["sort_workspace_elements"] == 0


def test_aggregate_ratios_weight_probability_entries_not_cases():
    small = stats.screening_counts(torch.tensor([[[1., 0.]]]), (0,))
    big = stats.screening_counts(torch.full((1, 1, 8), 0.5).index_fill(2, torch.tensor([0]), 0.6), (0,))
    summary = stats.summarize([small, big])
    removed = sum(r["rows"][0]["forced_positive"] + r["rows"][0]["forced_negative"] for r in (small, big))
    assert summary["screening_fraction_active"] == removed / 10
    assert summary["screening_fraction_active"] != sum(r["rows"][0]["screening_fraction_active"] for r in (small, big)) / 2


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("active_channels", [(0,), (0, 1, 2)])
def test_multiblock_counts_and_fallback_match_helper(device, active_channels):
    from rankseg import _screening
    generator = torch.Generator().manual_seed(98234)
    probs = torch.rand(1, 3, 65537, generator=generator).pow(12).to(device)
    for channel in set(range(3)) - set(active_channels):
        probs[0, channel].mul_(0.1)
    before = probs.clone()
    result = stats.screening_counts(probs, (0, 1, 2))
    assert all(row["fallback_rows"] == 0 for row in result["rows"])
    threshold = 0.5 + _screening._ROUNDING_GUARD * torch.finfo(probs.dtype).eps
    for row in result["rows"]:
        channel = row["channel"]
        if channel in active_channels:
            assert row["forced_positive"] == (probs[0, channel] > threshold).sum().item()
            assert row["forced_positive"] + row["forced_negative"] + row["undecided"] == 65537
        else:
            assert row["class_pruned_entries"] == 65537
    assert torch.equal(probs, before)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("budget", [0, 2, 1048576])
@pytest.mark.parametrize("strided", [False, True])
def test_statistics_match_actual_sort_workspace_without_altering_input(device, dtype, budget, strided, monkeypatch):
    from rankseg import _rankseg_algo as algo
    from rankseg import _screening
    monkeypatch.setattr(_screening, "_MAX_PADDED_ELEMENTS", budget)
    monkeypatch.setattr(_screening, "_MAX_PADDING_RATIO", 100)
    probs = example(device, dtype)
    if strided:
        storage = torch.zeros((1, 6, 16), device=device, dtype=dtype)
        storage[:, ::2, ::2] = probs
        probs = storage[:, ::2, ::2]
    before = probs.clone()
    measured = []
    original_sort, original_method = torch.sort, torch.Tensor.sort

    def sort(values, *args, **kwargs):
        measured.append(values.numel())
        return original_sort(values, *args, **kwargs)

    def method(values, *args, **kwargs):
        measured.append(values.numel())
        return original_method(values, *args, **kwargs)

    monkeypatch.setattr(torch, "sort", sort)
    monkeypatch.setattr(torch.Tensor, "sort", method)
    result = stats.screening_counts(probs, (0, 1, 2))
    assert measured == []  # The collector never reruns full sorts.
    algo.rankseg_rma(probs, safe_screening=True, output_mode="multilabel")
    assert sum(measured) == result["sort_workspace_elements"]
    assert all(row["fallback_rows"] == 0 for row in result["rows"])
    assert torch.equal(probs, before)


@pytest.mark.parametrize("shape,labels", [((2, 3, 8), (0, 1, 2)), ((1, 3, 8), (0, 1)), ((1, 3, 8), (0, 0, 1))])
def test_rejects_ambiguous_volume_or_label_mapping(shape, labels):
    with pytest.raises(ValueError):
        stats.screening_counts(torch.zeros(shape), labels)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_collector_links_completed_report_and_preserves_inputs(tmp_path, monkeypatch):
    import numpy as np
    import rankseg
    import triton
    from rankseg_benchmark.nnunet import config as configs, screening_benchmark as benchmark
    from rankseg_benchmark.nnunet.io import CaseFiles

    probability = tmp_path / "case.npz"
    np.savez(probability, softmax=example().numpy()[0].reshape(3, 2, 2, 2))
    manifest = tmp_path / "config.yaml"
    manifest.write_text("test fixture")
    config = SimpleNamespace(channel_labels=(0, 2, 7), probability_key="softmax")
    monkeypatch.setattr(configs, "load_dataset_config", lambda path: config)
    monkeypatch.setattr(benchmark, "prepare_cases", lambda cfg, count: (
        [CaseFiles("case", probability, tmp_path / "unused-label.npy")], {}, 1))
    root = Path(rankseg.__file__).resolve().parents[1]
    report_path = tmp_path / "benchmark.json"
    report = {"complete": True, "subset": False, "force_screening": True, "dtype": "float32",
              "torch": torch.__version__, "triton": triton.__version__, "device": torch.cuda.get_device_name(),
              "dataset": "test", "rankseg_sources": {"_screening.py": benchmark.sha256(root / "rankseg/_screening.py")},
              "manifest": str(manifest), "manifest_sha256": benchmark.sha256(manifest),
              "selected_case_ids": ["case"], "records": [{"case_id": "case", "fold": 0,
              "shape": [1, 3, 2, 2, 2], "probability_sha256": benchmark.sha256(probability),
              "restoration": {"crop_restored": False, "stored_probability_shape": "[3, 2, 2, 2]",
                              "crop_bbox": None, "properties_file": None}}]}
    report_path.write_text(json.dumps(report))
    args = SimpleNamespace(rankseg_path=root, benchmark_report=report_path, output=tmp_path / "stats.json")
    before = report_path.read_bytes(), probability.read_bytes()
    stats.run(args)
    result = json.loads(args.output.read_text())
    assert result["complete"] and result["summary"]["cases"] == 1
    assert result["benchmark_sha256"] == benchmark.sha256(report_path)
    assert result["summary"]["screening_fraction_active"] == 0.75
    assert result["timed"] is False
    assert before == (report_path.read_bytes(), probability.read_bytes())
    with pytest.raises(FileExistsError):
        stats.run(args)
    report["records"][0]["probability_sha256"] = "wrong"
    report_path.write_text(json.dumps(report))
    args.output = tmp_path / "wrong-cache.json"
    with pytest.raises(ValueError, match="Probability cache changed"):
        stats.run(args)
    assert json.loads(args.output.read_text())["complete"] is False
    # True now forces screening even for small inputs. Historical checkouts
    # still need the legacy bypass guard; never misstate their sort savings.
    report["force_screening"] = False
    report["records"][0]["probability_sha256"] = benchmark.sha256(probability)
    report_path.write_text(json.dumps(report))
    args.output = tmp_path / "production-bypass.json"
    stats.run(args)
    assert json.loads(args.output.read_text())["complete"]
    from rankseg import _rankseg_algo as algo
    monkeypatch.delattr(algo, "_rma_dice_use_screening")
    monkeypatch.setattr(algo, "_RMA_CUDA_SCREENING_MIN_SINGLE_CHANNEL_DIM", 262144, raising=False)
    monkeypatch.setattr(algo, "_RMA_CUDA_SCREENING_MIN_FEW_ROWS_DIM", 65536, raising=False)
    monkeypatch.setattr(algo, "_RMA_CUDA_SCREENING_FEW_ROWS", 16, raising=False)
    args.output = tmp_path / "legacy-production-bypass.json"
    with pytest.raises(ValueError, match="Production small-input bypass"):
        stats.run(args)
