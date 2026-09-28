"""Real scale CUDA harness: fair controls, coverage, and audit failures."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("cuda_scale_test", ROOT / "scripts/calibrate_screening_cuda.py")
cuda = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cuda)


def fixture_report():
    values = {m: dict(median_ms=2. if m == "screened" else 3.,
                      samples_ms=[2. if m == "screened" else 3.], peak_mib=1.) for m in cuda.METHODS}
    rounds = [values, deepcopy(values)]
    row = dict(profile="sample", dataset="real", dim=16, is_full_input=False, production_bypass=True,
               dispatch=dict(full=0, full_optimized=0, screened=1, production=0), rounds=rounds,
               objectives={m: dict(max_regret_eps=0., status="passed") for m in cuda.METHODS},
               counts=dict(active_entries=16, class_pruned_entries=16, total_entries=32,
                           forced_positive=4, forced_negative=4, undecided=8, actual_sort_entries=8,
                           active_rows=1, fallback_rows=0),
               comparisons={m: cuda.pair_summary(rounds, m) for m in cuda.METHODS if m != "screened"})
    return dict(complete=True, device="cuda", real_only=True, sampling="uniform", rounds=2, repeats=1,
                profiles=[dict(id="sample", kind="real_derived", dataset="real", tested_dims=[16])], records=[row])


def test_summary_keeps_optimized_control_separate_and_counts_objectives():
    report = fixture_report()
    result = cuda.summarize(report)
    assert result["objective_checks"] == 4 and result["objective_failures"] == 0
    values = result["results"][0]["active_inputs"]
    assert values["speedup_vs"]["full_optimized"] == 1.5
    assert values["screening_fraction_active"] == .5


@pytest.mark.parametrize("problem", ["incomplete", "duplicate", "missing", "synthetic", "rounds", "median",
                                     "nan", "memory", "objective", "partition", "retry", "dispatch", "comparison"])
def test_audit_rejects_corrupted_results(problem):
    report = fixture_report()
    row = report["records"][0]
    if problem == "incomplete":
        report["complete"] = False
    elif problem == "duplicate":
        report["records"].append(deepcopy(row))
    elif problem == "missing":
        report["records"].clear()
    elif problem == "synthetic":
        report["profiles"][0]["kind"] = "synthetic"
    elif problem == "rounds":
        row["rounds"].pop()
    elif problem == "median":
        row["rounds"][0]["full"]["median_ms"] = 0.
    elif problem == "nan":
        row["rounds"][0]["full"]["samples_ms"][0] = float("nan")
    elif problem == "memory":
        row["rounds"][0]["full"]["peak_mib"] = -1.
    elif problem == "objective":
        row["objectives"]["screened"]["max_regret_eps"] = 5.
    elif problem == "partition":
        row["counts"]["undecided"] += 1
    elif problem == "retry":
        row["counts"]["fallback_rows"] = 1
    elif problem == "dispatch":
        row["dispatch"]["full_optimized"] = 1
    elif problem == "comparison":
        row["comparisons"]["full"]["speedup"] = 17.
    with pytest.raises(ValueError):
        cuda.audit(report)


def test_failed_objective_remains_failed_in_summary():
    report = fixture_report()
    report["records"][0]["objectives"]["screened"] = dict(max_regret_eps=5., status="failed")
    assert cuda.summarize(report)["objective_failures"] == 1


def test_scales_include_full_without_duplicate_or_upsampling():
    p = dict(probs=torch.ones(1, 3, 128))
    assert cuda.planned_dims(p, [64, 128, 256], True) == [64, 128]
    assert cuda.planned_dims(p, [64, 256], True) == [64, 128]
    assert cuda.planned_dims(p, [64, 256], False) == [64]


def test_all_pruned_inputs_are_not_mixed_into_active_summary():
    assert cuda.aggregate([]) is None
    report = fixture_report()
    row = report["records"][0]
    row["counts"].update(active_rows=0, active_entries=0, class_pruned_entries=32,
                         forced_positive=0, forced_negative=0, undecided=0, actual_sort_entries=0)
    row["dispatch"]["screened"] = 0
    summary = cuda.summarize(report)["results"][0]
    assert summary["active_inputs"] is None
    assert summary["all_inputs"]["screening_fraction_active"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("screening_mode", ["true", "auto"])
def test_gpu_harness_end_to_end_and_restores_dispatch(tmp_path, monkeypatch, screening_mode):
    oracle, _ = cuda.cpu.bench.acceptance.load_helpers(Path(__import__("rankseg").__file__).resolve().parents[1], ROOT)
    from rankseg import _rankseg_algo as algo, _screening
    if _screening._cuda_backend() is None:
        pytest.skip("Triton unavailable")
    saved = {k: getattr(algo, k, None) for k in oracle.CUTOFFS}
    dispatch = algo._rma_dice_use_screening
    generator = torch.Generator().manual_seed(4)
    def profiles(args):
        result = []
        for i, channels in enumerate((1, 21)):
            p = torch.rand(1, channels, 129, generator=generator).pow(10)
            if i:
                p.zero_()
            result.append(dict(id=f"fixture{i}", kind="real_derived", dataset="fixture", probs=p,
                               channels=channels, mode="multilabel" if channels == 1 else "multiclass",
                               draw_order=torch.randperm(129, generator=generator)))
        return result, {}
    monkeypatch.setattr(cuda.cpu, "load_profiles", profiles)
    args = SimpleNamespace(output_dir=tmp_path / "run", dims=[64, 128], include_full=True,
                           samples_per_dataset=10, sampling_seed=123, threads=1, rounds=2, repeats=2, warmup=1,
                           production_screening=screening_mode)
    cuda.run(args)
    report = json.loads((args.output_dir / "results.json").read_text())
    summary = cuda.summarize(report)
    assert summary["configurations"] == 6 and summary["objective_checks"] == 24
    assert summary["objective_failures"] == 0
    assert report["production_screening"] == ("auto" if screening_mode == "auto" else True)
    assert all(r["production_bypass"] == (screening_mode == "auto") for r in report["records"])
    assert {k: getattr(algo, k, None) for k in oracle.CUTOFFS} == saved
    assert algo._rma_dice_use_screening is dispatch
    with pytest.raises(FileExistsError):
        cuda.run(args)
    with pytest.raises(RuntimeError):
        with cuda.control("screened", oracle):
            raise RuntimeError("intentional")
    assert {k: getattr(algo, k, None) for k in oracle.CUTOFFS} == saved
    assert algo._rma_dice_use_screening is dispatch
