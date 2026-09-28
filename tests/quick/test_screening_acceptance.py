"""Independent metrics, dispatch, provenance and CUDA timing acceptance checks."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("mini_acceptance", ROOT / "scripts/benchmark_screening_acceptance.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


@pytest.fixture(scope="module")
def helpers():
    return bench.load_helpers(Path(__import__("rankseg").__file__).resolve().parents[1], ROOT)


@pytest.mark.parametrize("classes", [2, 19, 21, 150])
@pytest.mark.parametrize("seed", range(5))
def test_independent_image_metrics(classes, seed):
    from rankseg_benchmark.metrics import ConfusionAccumulator
    g = torch.Generator().manual_seed(seed)
    pred = torch.randint(classes, (3, 9, 17), generator=g)
    label = torch.randint(classes, pred.shape, generator=g)
    label[0] = 0
    label[1, 0] = 255
    label[1, 1] = classes
    label[1, 2] = -1
    label[2] = 255
    acc = ConfusionAccumulator(classes, "multiclass", 255)
    acc.update(pred, label)
    rows = [{"methods": {"x": {"counts": bench.confusion(p, y, classes)}}} for p, y in zip(pred, label)]
    actual = bench.independent_metrics(rows, "x")
    expected = acc.summary()
    assert actual["dice"] == pytest.approx(expected["mDice"], abs=1e-14)
    assert actual["iou"] == pytest.approx(expected["mIoU"], abs=1e-14)


@pytest.mark.parametrize("seed", range(10))
def test_independent_kits_slice_case_fold_aggregation(seed):
    from rankseg_benchmark.metrics import MedicalCaseAccumulator, FoldMeanAccumulator
    g = torch.Generator().manual_seed(seed)
    folded = FoldMeanAccumulator(2)
    rows = []
    for fold in range(5):
        acc = MedicalCaseAccumulator(2, 255, True)
        for case in range(1 + fold):
            for image in range(case + 1):
                pred = torch.randint(2, (5, 9), generator=g)
                label = torch.randint(2, pred.shape, generator=g)
                if case == 0:
                    pred.fill_(0)
                    label.fill_(0 if fold % 2 else 255)
                elif case == 1:
                    pred.fill_(1)
                    label.fill_(1)
                acc.update_slice(case, pred, label)
                rows.append({"fold": fold, "case_id": case, "methods": {"x": {
                    "counts": bench.confusion(pred, label, 2, medical=True)}}})
        folded.add_fold(fold, acc)
    actual = bench.independent_metrics(rows, "x", True)
    expected = folded.summary()
    assert actual["dice"] == pytest.approx(expected["mDice"], abs=1e-14)
    assert actual["iou"] == pytest.approx(expected["mIoU"], abs=1e-14)


@pytest.mark.parametrize("pred,label", [([2], [0]), ([-1], [0]), ([0, 1], [0])])
def test_invalid_confusion_input_rejected(pred, label):
    with pytest.raises(ValueError):
        bench.confusion(pred, label, 2)


def test_natural_void_labels_match_existing_metric():
    p = torch.tensor([[0, 1, 20, 0, 20]])
    y = torch.tensor([[0, 1, 21, 255, -1]])
    c = bench.confusion(p, y, 21)
    assert sum(c["tp"]) == 2 and sum(c["fp"]) == sum(c["fn"]) == 0
    assert bench.unit_scores(c) == {"dice": 1., "iou": 1.}


def test_medical_label_clamping_matches_existing_metric():
    from rankseg_benchmark.metrics import MedicalCaseAccumulator
    p = torch.tensor([0, 1, 0, 1])
    y = torch.tensor([-1, 2, 255, 0])
    acc = MedicalCaseAccumulator(2, 255, True)
    acc.update_slice(0, p, y)
    actual = bench.unit_scores(bench.confusion(p, y, 2, medical=True), True)
    assert actual["dice"] == acc.summary()["mDice"]
    assert actual["iou"] == acc.summary()["mIoU"]


@pytest.mark.parametrize("medical", [False, True])
def test_all_ignored_is_not_perfect_empty(medical):
    c = bench.confusion([0, 1], [255, 255], 2, medical=medical)
    assert bench.unit_scores(c, medical) is None


@pytest.mark.parametrize("bad", [None, "revision", "dataset", "metadata_rows", "parquet_rows"])
def test_cache_provenance(tmp_path, bad):
    path = tmp_path / "pascal_voc-first100.parquet"
    pq.write_table(pa.table({"x": list(range(99 if bad == "parquet_rows" else 100))}), path)
    meta = {"dataset": "pascal_voc", "revision": bench.REVISION, "rows": 100}
    if bad == "revision":
        meta["revision"] = "main"
    elif bad == "dataset":
        meta["dataset"] = "ade20k"
    elif bad == "metadata_rows":
        meta["rows"] = 99
    path.with_suffix(".source.json").write_text(json.dumps(meta))
    if bad:
        with pytest.raises(ValueError):
            bench.cache_sources("pascal_voc", tmp_path, tmp_path)
    else:
        metadata, sources = bench.cache_sources("pascal_voc", tmp_path, tmp_path)
        assert not metadata["complete_source"]
        assert sources[0]["rows"] == 100
        assert sources[0]["sha256"] == bench.sha256(path)


def test_kits_requires_all_five_folds_and_pinned_revision(tmp_path):
    with pytest.raises(ValueError, match="revision"):
        bench.cache_sources("kits", tmp_path, tmp_path)
    pinned = tmp_path / bench.REVISION
    pinned.mkdir()
    with pytest.raises(FileNotFoundError):
        bench.cache_sources("kits", tmp_path, pinned)


def test_public_on_off_do_not_override_dispatch(helpers, monkeypatch):
    oracle, _ = helpers
    def reject(*args, **kwargs):
        raise AssertionError("Normal public paths must not patch dispatch")
    monkeypatch.setattr(oracle, "decoder_control", reject)
    p = torch.tensor([[[.8, .2, .03]]])
    from rankseg import RankSEG
    for name in ("full", "screened"):
        actual = bench.make_decoder(name, "multilabel", oracle).predict(p)
        expected = RankSEG(output_mode="multilabel", safe_screening=name == "screened").predict(p)
        assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("shape,bypass", [((1, 1, 32, 32), False), ((1, 1, 512, 512), False),
                                        ((1, 1, 513, 512), False), ((1, 3, 256, 256), False),
                                        ((1, 3, 257, 256), False), ((1, 19, 16, 16), False)])
@pytest.mark.parametrize("pruned", [False, True])
def test_cuda_production_dispatch_and_objective(helpers, shape, bypass, pruned):
    oracle, _ = helpers
    g = torch.Generator().manual_seed(812)
    p = torch.rand(shape, generator=g).pow(8).cuda()
    if pruned:
        p.mul_(.2)
    original = p.clone()
    assert bench.small_input_bypass(p) is bypass
    for name in ("full", "full_optimized", "screened", "screened_forced"):
        decoder = bench.make_decoder(name, "multilabel", oracle)
        def call():
            return decoder.predict(p)
        expected_calls = int(not pruned and (name == "screened_forced" or (name == "screened" and not bypass)))
        assert bench.observed_screening_call(call) == expected_calls
        diag = oracle.objective_regret(p, call(), oracle.bounded_binary_oracle(p))
        assert diag["max_regret_eps"] <= 4
    assert torch.equal(p, original)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_foreground_routing_and_timing(helpers):
    from rankseg_benchmark.datasets import get_spec
    from rankseg_benchmark.runner import _rankseg_predict
    oracle, _ = helpers
    p = torch.tensor([[[[.9, .1, .5]], [[.1, .9, .5]]]], device="cuda")
    dataset = get_spec("kits")
    operations = {"argmax": lambda: p.argmax(1)}
    for name in bench.METHODS[1:]:
        decoder = bench.make_decoder(name, "multilabel", oracle)
        operations[name] = lambda d=decoder: _rankseg_predict(d, p, spec=dataset, rankseg_output_mode="multilabel")
    stats, preds = bench.measure(operations, 1, 3, 1)
    assert preds["argmax"].tolist() == [[[0, 1, 0]]]
    for name in bench.METHODS:
        assert preds[name].shape == (1, 1, 3) and preds[name].device.type == "cpu"
        assert len(stats[name]["samples_ms"]) == 3
        assert stats[name]["median_ms"] > 0 and stats[name]["peak_mib"] > 0


def summary_fixture():
    counts = {"tp": [2, 1], "fp": [0, 0], "fn": [0, 0]}
    per_row = {"total_entries": 3, "active_entries": 3, "class_pruned_entries": 0,
               "forced_positive": 1, "forced_negative": 1, "undecided": 1,
               "actual_sort_entries": 1, "fallback_rows": 0, "active_rows": 1, "total_rows": 1,
               "label": 1}
    record = {"fold": None, "source_row": 0, "case_id": None, "small_input_bypass": True,
              "observed_screening_calls": 0, "different_pixels": {m: 0 for m in bench.METHODS},
              "screening_counts": {"rows": [per_row], "sort_workspace_elements": 1},
              "methods": {m: {"counts": deepcopy(counts), "median_ms": 1., "peak_mib": 2.,
                              "objective": {"status": "passed", "max_regret_eps": 0.}} for m in bench.METHODS}}
    return {"dataset": "pascal_voc", "sources": [{"rows": 1}], "records": [record],
            "repository_metrics": {m: {"mDice": 1., "mIoU": 1.} for m in bench.METHODS}}


def test_bypass_does_not_claim_forced_screening_savings(helpers):
    _, counts = helpers
    report = summary_fixture()
    actual = bench.summarize(report, counts)
    assert actual["forced_screening_certificate"]["screening_fraction_active"] == pytest.approx(2 / 3)
    assert actual["production_effective_sort_avoidance_active"] == 0
    assert actual["production_small_input_bypass_samples"] == 1


def test_failure_and_incomplete_reports_not_passed(helpers):
    _, counts = helpers
    report = summary_fixture()
    report["records"][0]["methods"]["screened"]["objective"] = {"status": "failed", "max_regret_eps": 5.}
    assert not bench.summarize(report, counts)["screened_checks_passed"]
    report["sources"][0]["rows"] = 2
    with pytest.raises(ValueError, match="Incomplete"):
        bench.summarize(report, counts)
    report["records"].append(deepcopy(report["records"][0]))
    with pytest.raises(ValueError, match="Duplicated"):
        bench.summarize(report, counts)


def test_metric_mismatch_not_accepted(helpers):
    _, counts = helpers
    report = summary_fixture()
    report["repository_metrics"]["screened"]["mDice"] = .1
    with pytest.raises(AssertionError, match="Independent metric"):
        bench.summarize(report, counts)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("screening_mode", ["true", "auto"])
def test_end_to_end_kits_report(helpers, tmp_path, screening_mode):
    oracle, counts = helpers
    pinned = tmp_path / bench.REVISION
    for fold in range(5):
        folder = pinned / f"kits/fold{fold}"
        folder.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist([{
            "probs": [[[.9, .1, .5]], [[.1, .9, .5]]],
            "label": [[0, 1, 0]], "case_id": fold,
        }]), folder / "test-00000.parquet")
    output = tmp_path / "output"
    output.mkdir()
    args = SimpleNamespace(kits_root=pinned, natural_cache=tmp_path, output_dir=output,
                           warmup=1, repeats=2, screening_mode=screening_mode)
    summary = bench.run_dataset(args, "kits", oracle, counts, {})
    report = json.loads((output / "kits.json").read_text())
    assert report["complete"] and len(report["records"]) == 5
    assert summary["case_count"] == 5 and summary["samples"] == 5
    assert report["screening_mode"] == ("auto" if screening_mode == "auto" else True)
    assert all(row["default_matches_false"] is True for row in report["records"])
    assert report["timing_controls"] == "outside timed intervals"
    assert summary["production_small_input_bypass_samples"] == (5 if screening_mode == "auto" else 0)
    assert summary["production_effective_sort_avoidance_active"] == pytest.approx(0 if screening_mode == "auto" else 2 / 3)
    assert summary["screened_checks_passed"] and summary["forced_screening_checks_passed"]
    assert len((output / "kits.records.jsonl").read_text().splitlines()) == 5
