# SPDX-License-Identifier: Apache-2.0
# Relocated from rankseg-nnunet-benchmark; see LICENSES/Apache-2.0.txt.
"""Separate engineering benchmark of RMA screening on audited v1 OOF caches.

This does not modify or replace the fixed Full-16 scientific evaluation.
Run serially: decoder controls temporarily override private RankSEG dispatch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import statistics
import sys
import time
from collections import Counter
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch

METHODS = ("argmax", "full", "full_optimized", "screened")
CUTOFFS = (
    "_RMA_CUDA_SCREENING_MIN_SINGLE_CHANNEL_DIM",
    "_RMA_CUDA_SCREENING_MIN_FEW_ROWS_DIM",
    "_RMA_CUDA_SCREENING_FEW_ROWS",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def decoder_control(method):
    """Force matched full sort or screening; always restore production settings."""
    from rankseg import _rankseg_algo as algo
    from rankseg import _screening as screening

    if method not in (*METHODS, "screened_torch"):
        raise ValueError(f"Unknown decoder: {method}")
    dispatch = getattr(algo, "_rma_dice_use_screening", None)
    saved = {key: getattr(algo, key) for key in CUTOFFS if hasattr(algo, key)}
    backend = screening._cuda_backend
    try:
        if dispatch is not None and method in ("full_optimized", "screened", "screened_torch"):
            algo._rma_dice_use_screening = lambda probs, mode: method != "full_optimized"
        elif method == "full_optimized":
            for key in CUTOFFS:
                setattr(algo, key, sys.maxsize)
        elif method.startswith("screened"):
            for key in CUTOFFS[:2]:
                setattr(algo, key, 0)
        if method == "screened_torch":
            screening._cuda_backend = lambda: None
        yield
    finally:
        for key, value in saved.items():
            setattr(algo, key, value)
        if dispatch is not None:
            algo._rma_dice_use_screening = dispatch
        screening._cuda_backend = backend


class ScreeningDecoder:
    """Use the same public wrapper and parameters for all RankSEG controls."""

    def __init__(self, method, output_mode="multiclass", *, production_dispatch=False, screening_mode=True):
        from rankseg import RankSEG

        if method not in (*METHODS, "screened_torch"):
            raise ValueError(f"Unknown decoder: {method}")
        self.method = method
        self.output_mode = output_mode
        self.production_dispatch = production_dispatch
        self.decoder = RankSEG(metric="dice", solver="RMA", smooth=0, pruning_prob=0.5,
                               output_mode=output_mode,
                               safe_screening=(screening_mode if method == "screened" else method != "full"))

    def control(self):
        if self.method == "argmax" or (self.method == "screened" and self.production_dispatch):
            return nullcontext()
        return decoder_control(self.method)

    def predict_uncontrolled(self, probs):
        if self.method == "argmax":
            return probs.argmax(1) if self.output_mode == "multiclass" else probs > 0.5
        # This matched optimized-full control is calibrated for CUDA only.
        if self.method == "full_optimized" and not probs.is_cuda:
            raise ValueError("full_optimized control requires CUDA")
        return self.decoder.predict(probs)

    def predict(self, probs):
        with self.control():
            return self.predict_uncontrolled(probs)


def _commands(path, executable):
    commands = []
    for line in path.read_text().splitlines():
        tokens = shlex.split(line, comments=True)
        if tokens and Path(tokens[0]).name == executable:
            commands.append(tokens)
    if len(commands) != 5:
        raise ValueError(f"Expected five recorded {executable} fold commands: {path}")
    return commands


def _fold(path):
    name = Path(path).name
    if name not in {f"fold_{i}" for i in range(5)}:
        raise ValueError(f"Invalid OOF fold directory: {path}")
    return int(name[-1])


def _probability_fold(path):
    """Accept the two official export layouts, not arbitrary nested folders."""
    parent = Path(path).parent
    if parent.name == "not_postprocessed":
        parent = parent.parent
    if parent.parent.name != "fold_predictions":
        raise ValueError(f"Unexpected OOF probability layout: {path}")
    return _fold(parent)


def audit_oof_root(root, task, assignments, seen=None):
    """Verify cached prediction placement and recorded single-fold lineage.

    Recorded commands are parsed, NEVER executed. Legacy absolute roots may
    have moved; only component directory names under this OOF parent resolve.
    This audits cache provenance, not a fresh checkpoint training/inference run.
    """
    root = Path(root).resolve()
    seen = set() if seen is None else seen
    if root in seen:
        raise ValueError("Cyclic OOF ensemble lineage")
    seen = seen | {root}
    observed = {}
    predictions = root / "fold_predictions"
    paths = list(predictions.glob("fold_*/*.npz"))
    paths.extend(predictions.glob("fold_*/not_postprocessed/*.npz"))
    for path in sorted(paths):
        if path.stem in observed:
            raise ValueError(f"Duplicate OOF case: {path.stem}")
        observed[path.stem] = _probability_fold(path)
    if observed != assignments:
        raise ValueError(f"Cached cases/folds do not match held-out assignments: {root}")
    single = root / "oof_metadata.json"
    ensemble = root / "oof_ensemble_metadata.json"
    metadata_path = single if single.exists() else ensemble
    metadata = json.loads(metadata_path.read_text())
    if metadata["task"] != task or metadata["cases"] != len(assignments):
        raise ValueError(f"OOF task/case count mismatch: {root}")
    evidence = {"root": str(root), "metadata_sha256": sha256(metadata_path), "metadata": metadata}
    if single.exists():
        if metadata["split_source"] != "nnU-Net v1 default KFold(5, shuffle=True, random_state=12345)":
            raise ValueError("This audit supports only the declared default v1 splits")
        csv_path = root / "fold_assignments.csv"
        with csv_path.open() as handle:
            rows = list(csv.DictReader(handle))
        actual = {r["case_id"]: int(r["fold"]) for r in rows}
        if actual != assignments or len(rows) != len(actual):
            raise ValueError(f"Incorrect or duplicate fold assignments: {root}")
        script = root / "run_oof_inference.sh"
        folds = set()
        for tokens in _commands(script, "nnUNet_predict"):
            fold = _fold(tokens[tokens.index("-o") + 1])
            f_start = tokens.index("-f") + 1
            f_end = next((i for i in range(f_start, len(tokens)) if tokens[i].startswith("-")), len(tokens))
            if tokens[f_start:f_end] != [str(fold)] or fold in folds:
                raise ValueError("OOF inference must use exactly the matching held-out fold")
            if (_fold(tokens[tokens.index("-i") + 1]) != fold
                    or tokens[tokens.index("-t") + 1] != task
                    or tokens[tokens.index("-m") + 1] != metadata["model"]):
                raise ValueError("Recorded inference task/model/input-fold mismatch")
            folds.add(fold)
        evidence.update(assignments_sha256=sha256(csv_path), commands_sha256=sha256(script))
    else:
        if metadata.get("fold_assignments_identical") is not True:
            raise ValueError("Ensemble lacks identical-fold provenance")
        script = root / "run_oof_ensemble.sh"
        components, folds = None, set()
        for tokens in _commands(script, "nnUNet_ensemble"):
            start, end = tokens.index("-f") + 1, tokens.index("-o")
            sources = [Path(value) for value in tokens[start:end]]
            fold = _fold(tokens[end + 1])
            if len(sources) != 2 or any(_fold(p) != fold for p in sources) or fold in folds:
                raise ValueError("Ensemble must combine two models from the SAME held-out fold")
            if any(p.parent.name != "fold_predictions" for p in sources):
                raise ValueError("Unexpected ensemble source layout")
            current = tuple(root.parent / p.parent.parent.name for p in sources)
            if components is not None and components != current:
                raise ValueError("Ensemble components differ between folds")
            components, folds = current, folds | {fold}
        evidence["components"] = [audit_oof_root(p, task, assignments, seen) for p in components]
        models = [item["metadata"].get("model") for item in evidence["components"]]
        if models != [metadata["first_model"], metadata["second_model"]]:
            raise ValueError("Ensemble model identities differ from metadata")
        evidence["commands_sha256"] = sha256(script)
    return evidence


def prepare_cases(config, per_fold):
    from .io import CaseFiles
    from .oof import default_nnunet_v1_splits

    if per_fold < 0:
        raise ValueError("cases per fold must be nonnegative (0 means all)")
    if (config.rankseg_metric, config.rankseg_solver, config.rankseg_output_mode,
        config.pruning_prob, config.smooth, config.unassigned_policy) != (
            "dice", "RMA", "multiclass", 0.5, 0.0, "max_score"):
        raise ValueError("This engineering benchmark requires the fixed unsmoothed Dice RMA configuration")
    # Native hard-mask comparison belongs to the existing scientific audit.
    # This experiment compares decoders of the exact same cached probabilities.
    all_cases, found = [], set()
    for path in sorted(config.probabilities_dir.glob(config.probability_glob)):
        if path.stem in found:
            raise ValueError(f"Duplicate probability case: {path.stem}")
        label = resolve_legacy_link(config.labels_dir / f"{path.stem}{config.label_extension}")
        all_cases.append(CaseFiles(path.stem, path, label))
        found.add(path.stem)
    if not all_cases:
        raise FileNotFoundError("No cached probabilities found")
    if config.case_ids is not None:
        raise ValueError("Use cases-per-fold, not an outcome-selected case list")
    label_ids = sorted(p.name[:-len(config.label_extension)] for p in
                       config.labels_dir.glob(f"*{config.label_extension}"))
    assignments = {case: fold for fold, split in enumerate(default_nnunet_v1_splits(label_ids))
                   for case in split["val"]}
    if found != set(assignments):
        raise ValueError("Manifest probability selection does not cover the complete held-out cohort")
    audit = audit_oof_root(config.probabilities_dir.parent, config.dataset_id, assignments)
    corrections = config.labels_dir / "corrections.json"
    if corrections.is_file():
        for item in json.loads(corrections.read_text())["corrections"]:
            label = resolve_legacy_link(config.labels_dir / f"{item['case_id']}{config.label_extension}")
            if sha256(label) != item["corrected_sha256"]:
                raise ValueError("Release-corrected label checksum mismatch")
        audit["label_corrections_sha256"] = sha256(corrections)
    selected = []
    for fold in range(5):
        cases = sorted((c for c in all_cases if assignments[c.case_id] == fold), key=lambda c: c.case_id)
        selected.extend(cases[:per_fold] if per_fold else cases)
    return selected, audit, len(all_cases)


def resolve_legacy_link(path, checkout_root=None):
    """Read a moved checkout's old absolute symlink without rewriting it."""
    path = Path(path)
    if path.is_file():
        return path
    from .paths import workspace_root
    root = Path(checkout_root) if checkout_root else workspace_root()
    if path.is_symlink():
        target = path.readlink()
        old_root = root.parent / "nnUNet_bench"
        if target.is_absolute() and target.is_relative_to(old_root):
            relocated = root / target.relative_to(old_root)
            if relocated.is_file() and relocated.resolve().is_relative_to(root.resolve()):
                return relocated
    raise FileNotFoundError(f"Missing input (no verified relocation): {path}")


@torch.no_grad()
def binary_oracle(probs):
    """Full-prefix float64 reference, bounded to one class workspace at a time."""
    flat = probs.flatten(2).reshape(-1, probs.flatten(2).shape[-1])
    result = []
    volumes = torch.arange(1, flat.shape[-1] + 1, device=probs.device, dtype=torch.float64)
    for row in flat:
        values = row.double()
        mean = values.sum()
        prefix = values.sort(descending=True).values.cumsum(0)
        result.append((2 * prefix / (mean + volumes + 1)).max())
    return torch.stack(result)


@torch.no_grad()
def bounded_binary_oracle(probs, chunk_size=1_048_576):
    """Same exhaustive float64 objective with bounded score/volume workspace.

    Sort in the input dtype: converting finite float32 values to float64
    cannot change their order. Accumulate every sorted value in float64 and
    examine every prefix, without approximations, screening or early exits.
    This is an untimed benchmark oracle, not a change to RankSEG's solver.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    flat = probs.flatten(2).reshape(-1, probs.flatten(2).shape[-1])
    result = []
    for row in flat:
        values = row.double()
        mean = values.sum()
        del values
        values, indices = row.sort(descending=True)
        del indices
        prefix = values.double()
        del values
        prefix.cumsum_(0)
        best = prefix.new_zeros(())
        for start in range(0, row.numel(), chunk_size):
            end = min(start + chunk_size, row.numel())
            volumes = torch.arange(start + 1, end + 1, device=probs.device, dtype=torch.float64)
            scores = 2 * prefix[start:end] / (mean + volumes + 1)
            best = torch.maximum(best, scores.max())
            del volumes, scores
        result.append(best)
        del prefix
    return torch.stack(result)


class ObjectiveBudgetExceeded(AssertionError):
    """A measured finite regret above the fixed budget, not an inference error."""

    def __init__(self, diagnostics, limit):
        self.diagnostics = diagnostics
        super().__init__(f"Objective regret {diagnostics['max_regret']} exceeds regression budget {limit}")


@torch.no_grad()
def objective_regret(probs, masks, optimum):
    flat = probs.flatten(2).reshape(-1, probs.flatten(2).shape[-1])
    masks = masks.reshape_as(flat)
    losses = []
    for i, row in enumerate(flat):
        active = row.max().item() > 0.5
        if not active:
            if masks[i].any():
                raise AssertionError("Class-pruned row is nonempty")
            continue
        values = row.double()
        score = 2 * values[masks[i]].sum() / (values.sum() + masks[i].sum() + 1)
        regret = (optimum[i] - score).item()
        if not math.isfinite(regret):
            raise AssertionError("Nonfinite objective regret")
        losses.append(max(0.0, regret))
    maximum = max(losses, default=0.0)
    limit = 4 * torch.finfo(torch.float32).eps
    diagnostics = {"max_regret": maximum, "max_regret_eps": maximum / torch.finfo(torch.float32).eps}
    if maximum > limit:
        raise ObjectiveBudgetExceeded(diagnostics, limit)
    return diagnostics


def check_method_objectives(record, tensor, optimum, decoders, methods, failure_policy):
    """Record each method independently; only finite budget excesses may continue."""
    if failure_policy not in ("stop", "record"):
        raise ValueError("Unknown objective failure policy")
    failed = False
    for method in methods:
        mask = decoders[method].predict(tensor)
        try:
            diagnostics = objective_regret(tensor, mask, optimum)
        except ObjectiveBudgetExceeded as error:
            record["methods"][method]["objective"] = {
                **error.diagnostics, "status": "failed", "error": str(error),
            }
            record["objective_check"] = "failed"
            failed = True
            if failure_policy == "stop":
                raise
        else:
            record["methods"][method]["objective"] = {**diagnostics, "status": "passed"}
        finally:
            del mask
    record["objective_check"] = "failed" if failed else "passed"


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def metric_summary(records, methods=METHODS):
    """Label-then-case means as in nnU-Net; paired-complete cohort for comparisons."""
    complete = [r for r in records if all(r["methods"].get(m, {}).get("status") == "ok" for m in methods)]
    result = {"paired_complete_cases": len(complete), "attempted_cases": len(records), "methods": {}}
    result["objective_check_counts"] = dict(Counter(r.get("objective_check", "not_checked") for r in records))
    for method in methods:
        rows = [row for r in complete for row in r["methods"][method]["metrics"]]
        per_label = {}
        for label in sorted({r["label"] for r in rows}):
            per_label[label] = {}
            for metric in ("dice", "iou"):
                numbers = [r[metric] for r in rows if r["label"] == label and math.isfinite(r[metric])]
                per_label[label][metric] = statistics.mean(numbers) if numbers else math.nan
        summary = {"per_label": per_label}
        for metric in ("dice", "iou"):
            numbers = [r[metric] for r in per_label.values() if math.isfinite(r[metric])]
            summary[metric] = statistics.mean(numbers) if numbers else math.nan
        times = [r["methods"][method]["median_ms"] for r in complete]
        peaks = [r["methods"][method]["peak_incremental_mib"] for r in complete]
        summary.update(mean_ms=statistics.mean(times) if times else None,
                       median_ms=statistics.median(times) if times else None,
                       max_incremental_mib=max(peaks, default=None))
        result["methods"][method] = summary
    # Describe per-case foreground-macro and per-label changes separately.
    result["worst_changes_vs_full"] = {}
    for method in methods:
        if method == "full":
            continue
        for metric in ("dice", "iou"):
            case_changes, label_changes = [], []
            for record in complete:
                base = {r["label"]: r[metric] for r in record["methods"]["full"]["metrics"]}
                new = {r["label"]: r[metric] for r in record["methods"][method]["metrics"]}
                finite = [k for k in base if math.isfinite(base[k]) and math.isfinite(new[k])]
                if finite:
                    case_changes.append({"case_id": record["case_id"],
                                         "delta_pp": 100 * statistics.mean(new[k] - base[k] for k in finite)})
                    label_changes.extend({"case_id": record["case_id"], "label": k,
                                          "delta_pp": 100 * (new[k] - base[k])} for k in finite)
            result["worst_changes_vs_full"][f"{method}_{metric}"] = {
                "case": min(case_changes, key=lambda r: r["delta_pp"], default=None),
                "case_label": min(label_changes, key=lambda r: r["delta_pp"], default=None),
                "note": "paired finite labels only; empty-label counts remain in raw records",
            }
    return result


def run(args):
    # Do this before importing any module that imports RankSEG.
    failure_policy = getattr(args, "objective_failure_policy", "stop")
    if failure_policy not in ("stop", "record"):
        raise ValueError("Unknown objective failure policy")
    rankseg_root = args.rankseg_path.resolve()
    if not (rankseg_root / "rankseg" / "__init__.py").is_file():
        raise ValueError("--rankseg-path must name the RankSEG repository root")
    sys.path.insert(0, str(rankseg_root))
    import rankseg
    from rankseg import _screening

    from .config import load_dataset_config
    from .io import (
        load_probabilities,
        load_segmentation_with_voxel_volume,
        restore_cropped_probabilities,
        validate_case,
    )
    from .metrics import overlap_metrics, overlap_region_metrics
    from .postprocessing import OfficialNnUNetPostprocessor

    if Path(rankseg.__file__).resolve().parent != rankseg_root / "rankseg":
        raise RuntimeError("A different RankSEG was already imported")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; never silently mix CPU timings with CUDA")
    if _screening._cuda_backend() is None:
        raise RuntimeError("This fused-screening comparison requires optional Triton")
    import triton

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark output: {args.output}")
    config = load_dataset_config(args.manifest)
    cases, audit, available = prepare_cases(config, args.cases_per_fold)
    torch.set_num_threads(args.threads)
    postprocessor = OfficialNnUNetPostprocessor(config.postprocessing_file) if config.postprocessing_file else None
    production_dispatch = getattr(args, "production_dispatch", False)
    oracle_name = getattr(args, "oracle", "original")
    if oracle_name not in ("original", "bounded"):
        raise ValueError("Unknown numerical oracle")
    oracle = bounded_binary_oracle if oracle_name == "bounded" else binary_oracle
    screening_mode = "auto" if getattr(args, "screening_mode", "true") == "auto" else True
    decoders = {m: ScreeningDecoder(m, production_dispatch=production_dispatch, screening_mode=screening_mode)
                for m in METHODS}
    binary_decoders = {m: ScreeningDecoder(m, "multilabel", production_dispatch=production_dispatch,
                                         screening_mode=screening_mode)
                       for m in METHODS if m != "argmax"}
    report = {
        "kind": "screening_engineering_not_full16", "dataset": config.dataset_id,
        "selection": ("all cases in all five held-out folds" if args.cases_per_fold == 0 else
                      "first lexicographic case IDs within each of five held-out folds"),
        "subset": len(cases) != available, "available_cases": available,
        "selected_case_ids": [c.case_id for c in cases], "cases_per_fold": args.cases_per_fold,
        "manifest": str(args.manifest.resolve()), "manifest_sha256": sha256(args.manifest),
        "cache_oof_audit": audit, "source_provenance": config.provenance,
        "native_hard_mask_audit": "not repeated; compare argmax of identical cached probabilities",
        "rankseg_path": rankseg.__file__, "torch": torch.__version__, "triton": triton.__version__,
        "device": torch.cuda.get_device_name(), "dtype": "float32", "evaluation_unit": "whole 3D volume",
        "rankseg_sources": {p.name: sha256(p) for p in sorted((rankseg_root / "rankseg").glob("*.py"))},
        "benchmark_sha256": sha256(__file__),
        "postprocessing": postprocessor.summary() if postprocessor else None,
        "timing": "paired rotating order; median synchronized wall time; excludes loading, transfers, compilation, metrics, postprocessing and objective oracle",
        "warmup": args.warmup, "repeats": args.repeats,
        "objective_failure_policy": failure_policy, "objective_budget_eps": 4,
        "complete_definition": "all selected cases processed; numerical failures remain failures",
        "force_screening": not production_dispatch, "oracle": oracle_name,
        "screening_mode": screening_mode, "timing_controls": "outside timed intervals",
        "oom_policy": "record CUDA OOM, no CPU retry; comparisons use paired-complete cases",
        "records": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve a new file now, but keep it a valid, explicitly partial report
    # after every completed case. No probability/checkpoint files are written.
    with args.output.open("x") as handle:
        json.dump(safe_json({**report, "complete": False}), handle, indent=2, allow_nan=False)
    for index, case in enumerate(cases):
        print(f"[{index + 1}/{len(cases)}] {config.dataset_id} {case.case_id}", flush=True)
        probabilities, key = load_probabilities(case.probabilities, config.probability_key)
        target, voxel_volume = load_segmentation_with_voxel_volume(case.label, reader=config.segmentation_reader)
        probabilities, restoration = restore_cropped_probabilities(
            probabilities, probability_path=case.probabilities, target_shape=target.shape,
            background_channel=config.channel_labels.index(0),
        )
        diagnostics = validate_case(probabilities, target, channel_labels=config.channel_labels,
                                    ignore_label=config.ignore_label, probability_key=key)
        tensor = torch.from_numpy(np.ascontiguousarray(probabilities, dtype=np.float32)).unsqueeze(0).cuda()
        record = {"case_id": case.case_id, "fold": _probability_fold(case.probabilities),
                  "shape": list(tensor.shape), "probability_sha256": sha256(case.probabilities),
                  "label_path_used": str(case.label), "label_sha256": sha256(case.label), "restoration": restoration,
                  "probability_diagnostics": diagnostics.__dict__, "methods": {}}
        cpu_predictions = {}
        for method in METHODS:
            try:
                torch.cuda.synchronize()
                baseline_memory = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                prediction = decoders[method].predict(tensor)
                torch.cuda.synchronize()
                peak = (torch.cuda.max_memory_allocated() - baseline_memory) / 2**20
                cpu_predictions[method] = prediction.squeeze(0).cpu().numpy()
                del prediction
                record["methods"][method] = {"status": "ok", "peak_incremental_mib": peak}
            except torch.OutOfMemoryError:
                record["methods"][method] = {"status": "cuda_oom"}
                torch.cuda.empty_cache()
        active_methods = list(cpu_predictions)
        for _ in range(args.warmup):
            for method in list(active_methods):
                try:
                    prediction = decoders[method].predict(tensor)
                    del prediction
                except torch.OutOfMemoryError:
                    record["methods"][method] = {"status": "cuda_oom", "phase": "warmup"}
                    active_methods.remove(method)
                    torch.cuda.empty_cache()
        samples = {method: [] for method in active_methods}
        for repeat in range(args.repeats):
            if not active_methods:
                break
            shift = (index + repeat) % len(active_methods)
            for method in active_methods[shift:] + active_methods[:shift]:
                try:
                    with decoders[method].control():
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        prediction = decoders[method].predict_uncontrolled(tensor)
                        torch.cuda.synchronize()
                        samples[method].append((time.perf_counter() - start) * 1000)
                        del prediction
                except torch.OutOfMemoryError:
                    record["methods"][method] = {"status": "cuda_oom", "phase": "timing"}
                    active_methods.remove(method)
                    torch.cuda.empty_cache()
        for method in active_methods:
            raw = cpu_predictions[method]
            result = record["methods"][method]
            result.update(samples_ms=samples[method], median_ms=statistics.median(samples[method]))
            if "full" in cpu_predictions:
                result["raw_differing_voxels_vs_full"] = int(np.count_nonzero(raw != cpu_predictions["full"]))
            prediction = np.asarray(config.channel_labels)[raw]
            start = time.perf_counter()
            if postprocessor is not None:
                prediction = postprocessor(prediction, volume_per_voxel=voxel_volume)
            result["postprocessing_ms"] = (time.perf_counter() - start) * 1000
            rows = (overlap_region_metrics(prediction, target, config.regions, ignore_label=config.ignore_label)
                    if config.regions else overlap_metrics(prediction, target, config.foreground_labels,
                                                          ignore_label=config.ignore_label))
            result["metrics"] = rows
        del cpu_predictions, probabilities, target
        if active_methods:
            del prediction, raw
        optimum, mask = None, None
        try:
            checked_methods = [method for method in active_methods if method != "argmax"]
            if checked_methods:
                optimum = oracle(tensor)
                check_method_objectives(record, tensor, optimum, binary_decoders, checked_methods, failure_policy)
            else:
                record["objective_check"] = "no_rankseg_method_verified"
        except torch.OutOfMemoryError:
            record["objective_oom"] = True
            if record.get("objective_check") != "failed":
                record["objective_check"] = "cuda_oom_not_verified"
        except AssertionError as error:
            record["objective_check"] = "failed"
            record["objective_error"] = str(error)
            report["records"].append(record)
            report["complete"] = False
            report["summary"] = metric_summary(report["records"])
            args.output.write_text(json.dumps(safe_json(report), indent=2, allow_nan=False) + "\n")
            raise
        finally:
            del optimum, mask
            torch.cuda.empty_cache()
        del tensor
        report["records"].append(record)
        report["summary"] = metric_summary(report["records"])
        report["complete"] = index == len(cases) - 1
        args.output.write_text(json.dumps(safe_json(report), indent=2, allow_nan=False) + "\n")
        print({m: round(r["median_ms"], 3) for m, r in record["methods"].items() if r["status"] == "ok"}, flush=True)
    print(json.dumps(safe_json(report["summary"]), indent=2, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rankseg-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases-per-fold", type=int, default=2, help="0 means all; selection does not use outcomes")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--screening-mode", choices=("true", "auto"), default="true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--production-dispatch", action="store_true",
                        help="Use normal safe_screening=True dispatch, without forcing small-input screening")
    parser.add_argument("--oracle", choices=("original", "bounded"), default="original",
                        help="Untimed exhaustive float64 oracle; bounded reduces temporary score memory")
    parser.add_argument("--objective-failure-policy", choices=("stop", "record"), default="stop",
                        help="record continues after finite objective-budget failures; never relaxes the check")
    args = parser.parse_args()
    if args.cases_per_fold < 0 or args.repeats < 1 or args.warmup < 0 or args.threads < 1:
        parser.error("invalid benchmark counts")
    run(args)


if __name__ == "__main__":
    main()
