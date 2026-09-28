import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rankseg_benchmark.nnunet.config import DatasetConfig
from rankseg_benchmark.nnunet.oof import (
    default_nnunet_v1_splits,
    prepare_oof_ensemble_v1,
    prepare_oof_v1,
)
from rankseg_benchmark.nnunet.screening_benchmark import (
    CUTOFFS,
    METHODS,
    ObjectiveBudgetExceeded,
    ScreeningDecoder,
    _probability_fold,
    audit_oof_root,
    binary_oracle,
    check_method_objectives,
    decoder_control,
    metric_summary,
    objective_regret,
    prepare_cases,
    resolve_legacy_link,
    safe_json,
)


def _fixture(tmp_path, model="3d_fullres"):
    images = tmp_path / "imagesTr"
    labels = tmp_path / "labelsTr"
    images.mkdir(exist_ok=True)
    labels.mkdir(exist_ok=True)
    ids = [f"case_{i:03d}" for i in range(10)]
    for case in ids:
        (images / f"{case}_0000.nii.gz").write_bytes(b"fixture")
        np.save(labels / f"{case}.npy", np.zeros((2, 2, 2), dtype=np.uint8))
    root = tmp_path / model
    prepare_oof_v1(task="Task999", images_dir=images, output_dir=root, model=model)
    assignment = {case: fold for fold, split in enumerate(default_nnunet_v1_splits(ids)) for case in split["val"]}
    for case, fold in assignment.items():
        np.savez(root / "fold_predictions" / f"fold_{fold}" / f"{case}.npz",
                 softmax=np.tile(np.array([0.8, 0.2])[:, None, None, None], (1, 2, 2, 2)))
    config = DatasetConfig("Task999", "fixture", {0: "bg", 1: "fg"}, (0, 1),
                           root / "fold_predictions", labels, tmp_path / "output", (1,),
                           probability_glob="fold_*/*.npz", label_extension=".npy")
    return config, assignment


def test_audited_selection_is_deterministic_and_covers_folds(tmp_path):
    config, assignment = _fixture(tmp_path)
    cases, _, available = prepare_cases(config, 1)
    assert available == 10 and len(cases) == 5
    assert [c.case_id for c in cases] == [min(c for c, f in assignment.items() if f == i) for i in range(5)]
    assert len(prepare_cases(config, 0)[0]) == 10
    with pytest.raises(ValueError, match="fixed unsmoothed"):
        prepare_cases(replace(config, smooth=1), 1)
    with pytest.raises(ValueError, match="case list"):
        prepare_cases(replace(config, case_ids=(cases[0].case_id,)), 1)
    with pytest.raises(ValueError, match="complete held-out cohort"):
        prepare_cases(replace(config, probability_glob="fold_0/*.npz"), 0)


def test_nested_raw_export_layout_and_duplicate_rejection(tmp_path):
    config, assignments = _fixture(tmp_path)
    for path in config.probabilities_dir.glob("fold_*/*.npz"):
        destination = path.parent / "not_postprocessed" / path.name
        destination.parent.mkdir(exist_ok=True)
        path.rename(destination)
    nested = replace(config, probability_glob="fold_*/not_postprocessed/*.npz")
    cases, _, total = prepare_cases(nested, 0)
    assert len(cases) == total == 10
    assert all(_probability_fold(c.probabilities) == assignments[c.case_id] for c in cases)
    duplicate = cases[0].probabilities.parent.parent / cases[0].probabilities.name
    duplicate.write_bytes(b"duplicate raw export")
    with pytest.raises(ValueError, match="Duplicate OOF case"):
        prepare_cases(nested, 0)


@pytest.mark.parametrize("path", ["fold_predictions/fold_7/x.npz",
                                  "fold_predictions/fold_0/arbitrary/x.npz",
                                  "other/fold_0/not_postprocessed/x.npz"])
def test_rejects_unexpected_probability_layout(path):
    with pytest.raises(ValueError):
        _probability_fold(path)


@pytest.mark.parametrize("mutation", ["multi_fold", "wrong_model", "wrong_input"])
def test_rejects_invalid_inference_lineage(tmp_path, mutation):
    config, assignments = _fixture(tmp_path)
    script = config.probabilities_dir.parent / "run_oof_inference.sh"
    text = script.read_text()
    if mutation == "multi_fold":
        text = text.replace("-f 0", "-f 0 1")
    elif mutation == "wrong_model":
        text = text.replace("-m 3d_fullres", "-m 2d")
    else:
        text = text.replace("fold_inputs/fold_0", "fold_inputs/fold_1")
    script.write_text(text)
    with pytest.raises(ValueError):
        audit_oof_root(config.probabilities_dir.parent, "Task999", assignments)


def test_rejects_cross_fold_ensemble(tmp_path):
    first, assignment = _fixture(tmp_path, "2d")
    second, _ = _fixture(tmp_path, "3d_fullres")
    root = tmp_path / "ensemble"
    prepare_oof_ensemble_v1(first_oof_dir=first.probabilities_dir.parent,
                           second_oof_dir=second.probabilities_dir.parent, output_dir=root)
    for case, fold in assignment.items():
        (root / "fold_predictions" / f"fold_{fold}" / f"{case}.npz").write_bytes(b"fixture")
    assert len(audit_oof_root(root, "Task999", assignment)["components"]) == 2
    script = root / "run_oof_ensemble.sh"
    script.write_text(script.read_text().replace("3d_fullres/fold_predictions/fold_0", "3d_fullres/fold_predictions/fold_1"))
    with pytest.raises(ValueError, match="SAME held-out fold"):
        audit_oof_root(root, "Task999", assignment)


def test_moved_symlinks_are_read_only_and_restricted(tmp_path):
    checkout = tmp_path / "rankseg-nnunet-benchmark"
    checkout.mkdir()
    target = checkout / "value.npy"
    target.write_bytes(b"label")
    link = checkout / "old.npy"
    link.symlink_to(tmp_path / "nnUNet_bench" / "value.npy")
    original = link.readlink()
    assert resolve_legacy_link(link, checkout) == target
    assert link.readlink() == original
    wrong = checkout / "missing.npy"
    wrong.symlink_to(tmp_path / "unrelated" / "value.npy")
    with pytest.raises(FileNotFoundError):
        resolve_legacy_link(wrong, checkout)


def test_dispatch_controls_restore_on_error():
    from rankseg import _rankseg_algo as algo
    if not hasattr(algo, CUTOFFS[0]) and not hasattr(algo, "_rma_dice_use_screening"):
        pytest.skip("requires local screening implementation")
    from rankseg import _screening
    original = [getattr(algo, k, None) for k in CUTOFFS]
    dispatch = getattr(algo, "_rma_dice_use_screening", None)
    backend = _screening._cuda_backend
    for method in METHODS + ("screened_torch",):
        with pytest.raises(RuntimeError):
            with decoder_control(method):
                if dispatch is not None and method in ("full_optimized", "screened", "screened_torch"):
                    assert algo._rma_dice_use_screening(None, True) == (method != "full_optimized")
                elif method.startswith("screened"):
                    assert getattr(algo, CUTOFFS[0]) == 0
                raise RuntimeError("test")
        assert original == [getattr(algo, k, None) for k in CUTOFFS]
        assert getattr(algo, "_rma_dice_use_screening", None) is dispatch
        assert _screening._cuda_backend is backend


def test_summary_uses_label_macro_and_paired_complete_cases():
    records = []
    for i in range(2):
        methods = {m: {"status": "ok", "median_ms": 2, "peak_incremental_mib": 4,
                       "metrics": [{"label": 1, "dice": 1., "iou": 1.},
                                   {"label": 2, "dice": 0. if i == 0 else math.nan,
                                    "iou": 0. if i == 0 else math.nan}]} for m in METHODS}
        records.append({"case_id": str(i), "methods": methods})
    records.append({"case_id": "OOM", "methods": {m: {"status": "cuda_oom"} for m in METHODS}})
    result = metric_summary(records)
    assert result["paired_complete_cases"] == 2
    assert result["methods"]["full"]["dice"] == 0.5  # not per-case mean 0.75
    assert result["attempted_cases"] == 3
    json.dumps(safe_json(records), allow_nan=False)


def test_oracle_rejects_wrong_mask_and_class_pruning():
    probs = torch.tensor([[[0.9, 0.8, 0.0], [0.1, 0.0, 0.0]]])
    optimum = binary_oracle(probs)
    valid = torch.tensor([[[True, True, False], [False, False, False]]])
    assert objective_regret(probs, valid, optimum)["max_regret"] == 0
    with pytest.raises(AssertionError, match="Objective regret"):
        objective_regret(probs, torch.zeros_like(valid), optimum)
    valid[0, 1, 0] = True
    with pytest.raises(AssertionError, match="Class-pruned"):
        objective_regret(probs, valid, optimum)


@pytest.mark.parametrize("policy", ["stop", "record"])
def test_objective_failure_policy_retains_failure_and_checks_other_methods(policy):
    probs = torch.tensor([[[0.9, 0.8, 0.0]]])
    valid = torch.tensor([[[True, True, False]]])
    called = []

    def predict(method, mask):
        called.append(method)
        return mask

    decoders = {"full": SimpleNamespace(predict=lambda p: predict("full", torch.zeros_like(valid))),
                "screened": SimpleNamespace(predict=lambda p: predict("screened", valid))}
    record = {"methods": {m: {"status": "ok"} for m in decoders}}
    if policy == "stop":
        with pytest.raises(ObjectiveBudgetExceeded):
            check_method_objectives(record, probs, binary_oracle(probs), decoders, list(decoders), policy)
        assert called == ["full"]
        assert "objective" not in record["methods"]["screened"]
    else:
        check_method_objectives(record, probs, binary_oracle(probs), decoders, list(decoders), policy)
        assert called == ["full", "screened"]
        assert record["methods"]["screened"]["objective"]["status"] == "passed"
    assert record["objective_check"] == "failed"
    assert record["methods"]["full"]["objective"]["status"] == "failed"
    assert record["methods"]["full"]["objective"]["max_regret_eps"] > 4


@pytest.mark.parametrize("bad", ["pruned", "nonfinite", "decoder"])
def test_record_policy_does_not_swallow_structural_or_unexpected_errors(bad):
    probs = torch.tensor([[[0.9, 0.8, 0.0], [0.1, 0, 0]]])
    masks = torch.tensor([[[True, True, False], [False, False, False]]])
    optimum = binary_oracle(probs)
    if bad == "pruned":
        masks[0, 1, 0] = True
    if bad == "nonfinite":
        optimum[0] = float("nan")

    def predict(values):
        if bad == "decoder":
            raise RuntimeError("inference error")
        return masks

    record = {"methods": {"screened": {"status": "ok"}}}
    with pytest.raises((AssertionError, RuntimeError)):
        check_method_objectives(record, probs, optimum, {"screened": SimpleNamespace(predict=predict)},
                                ["screened"], "record")
    assert record.get("objective_check") != "passed"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_all_controls_decode_same_easy_case_on_cuda():
    from rankseg import _rankseg_algo as algo
    if not hasattr(algo, CUTOFFS[0]) and not hasattr(algo, "_rma_dice_use_screening"):
        pytest.skip("requires local screening implementation")
    probs = torch.zeros(1, 2, 8, 8, 8, device="cuda")
    probs[:, 0, :4] = 1
    probs[:, 1, 4:] = 1
    expected = probs.argmax(1)
    for method in METHODS + ("screened_torch",):
        assert torch.equal(ScreeningDecoder(method).predict(probs), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_end_to_end_screening_report_and_no_overwrite(tmp_path, monkeypatch):
    import rankseg
    import yaml
    from rankseg import _rankseg_algo as algo

    from rankseg_benchmark.nnunet.screening_benchmark import run
    if not hasattr(algo, CUTOFFS[0]) and not hasattr(algo, "_rma_dice_use_screening"):
        pytest.skip("requires local screening implementation")
    config, _ = _fixture(tmp_path)
    channels = np.arange(8).reshape(2, 2, 2) % 3
    labels = np.array([0, 1, 4])[channels].astype(np.uint8)
    probs = np.eye(3, dtype=np.float32)[channels].transpose(3, 0, 1, 2)
    for path in config.probabilities_dir.glob("fold_*/*.npz"):
        np.savez(path, softmax=probs)
        np.save(config.labels_dir / (path.stem + ".npy"), labels)
    manifest = tmp_path / "test.yaml"
    manifest.write_text(yaml.safe_dump({
        "dataset": {"id": "Task999", "labels": {0: "bg", 1: "one", 4: "four"}},
        "inputs": {"probabilities_dir": str(config.probabilities_dir), "labels_dir": str(config.labels_dir),
                   "probability_glob": "fold_*/*.npz", "label_extension": ".npy"},
        "outputs": {"dir": str(config.output_dir)},
    }))
    args = SimpleNamespace(rankseg_path=Path(rankseg.__file__).resolve().parents[1],
                           manifest=manifest, output=tmp_path / "screening.json",
                           cases_per_fold=1, repeats=2, warmup=1, threads=2)
    run(args)
    report = json.loads(args.output.read_text())
    assert report["complete"] and report["subset"]
    assert report["summary"]["paired_complete_cases"] == 5
    assert all(r["objective_check"] == "passed" for r in report["records"])
    assert all(report["summary"]["methods"][m]["dice"] == 1 for m in METHODS)
    assert all(report["summary"]["methods"][m]["iou"] == 1 for m in METHODS)
    before = args.output.read_bytes()
    with pytest.raises(FileExistsError):
        run(args)
    assert args.output.read_bytes() == before

    # A numerical failure must remain failed while later methods/cases run.
    import rankseg_benchmark.nnunet.screening_benchmark as benchmark
    original_regret = benchmark.objective_regret
    calls = []

    def baseline_failure(*values):
        calls.append(True)
        if len(calls) % 3 == 1:
            eps = torch.finfo(torch.float32).eps
            raise ObjectiveBudgetExceeded({"max_regret": 5 * eps, "max_regret_eps": 5.0}, 4 * eps)
        return original_regret(*values)

    with monkeypatch.context() as patch:
        patch.setattr(benchmark, "objective_regret", baseline_failure)
        args.objective_failure_policy = "record"
        args.output = tmp_path / "record_failures.json"
        run(args)
        report = json.loads(args.output.read_text())
        assert report["complete"] and len(report["records"]) == 5
        assert report["objective_budget_eps"] == 4
        assert report["summary"]["objective_check_counts"] == {"failed": 5}
        assert all(r["methods"]["screened"]["objective"]["status"] == "passed" for r in report["records"])
        assert all(r["methods"]["full"]["objective"]["status"] == "failed" for r in report["records"])
        assert all(report["summary"]["methods"][m]["dice"] == 1 for m in METHODS)
        calls.clear()
        args.objective_failure_policy = "stop"
        args.output = tmp_path / "stop_failure.json"
        with pytest.raises(ObjectiveBudgetExceeded):
            run(args)
        partial = json.loads(args.output.read_text())
        assert not partial["complete"] and len(partial["records"]) == 1
        assert partial["summary"]["objective_check_counts"] == {"failed": 1}
        assert partial["records"][0]["methods"]["full"]["objective"]["max_regret_eps"] == 5.0

    # A completed run with only argmax surviving is NOT a passed RankSEG oracle.
    original_predict = ScreeningDecoder.predict

    def oom_for_rankseg(self, values):
        if self.method == "argmax":
            return original_predict(self, values)
        raise torch.OutOfMemoryError("simulated RankSEG allocation failure")

    def no_oracle_expected(values):
        raise AssertionError("no successful RankSEG prediction to verify")

    monkeypatch.setattr(ScreeningDecoder, "predict", oom_for_rankseg)
    monkeypatch.setattr("rankseg_benchmark.nnunet.screening_benchmark.binary_oracle", no_oracle_expected)
    args.output = tmp_path / "oom.json"
    run(args)
    report = json.loads(args.output.read_text())
    assert report["complete"] and report["summary"]["paired_complete_cases"] == 0
    assert all(r["objective_check"] == "no_rankseg_method_verified" for r in report["records"])
    assert all(r["methods"][m]["status"] == "cuda_oom"
               for r in report["records"] for m in METHODS if m != "argmax")
