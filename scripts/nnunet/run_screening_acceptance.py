"""Serial full-cohort acceptance on frozen RankSEG sources; no inference.

Benchmark process logs stay in the artifact directory. Numerical failures are
recorded without relaxing checks, and never relabeled as successful validation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from rankseg_benchmark.nnunet.paths import workspace_root
import subprocess
import sys


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rankseg-path", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    artifacts = args.output_dir.resolve()
    snapshot = args.rankseg_path.resolve()
    if (artifacts / "run_status.json").exists():
        raise FileExistsError(artifacts / "run_status.json")
    artifacts.mkdir(parents=True, exist_ok=True)
    sources = [*snapshot.rglob("*.py"), *root.joinpath("rankseg_benchmark/nnunet").glob("*.py"),
               root / "scripts/nnunet/collect_screening_proportions.py", Path(__file__).resolve()]
    hashes = {str(p): sha256(p) for p in sources}
    datasets = [("liver", "Task003_Liver", 131), ("pancreas", "Task007_Pancreas", 281),
                ("hepaticvessel", "Task008_HepaticVessel", 303), ("lung", "Task006_Lung", 63)]
    status = dict(started_utc=now(), complete=False, source_hashes=hashes, datasets={})
    def write_status():
        (artifacts / "run_status.json").write_text(json.dumps(status, indent=2) + "\n")
    def check_sources():
        for name, digest in hashes.items():
            if sha256(Path(name)) != digest:
                raise RuntimeError(f"Frozen source changed: {name}")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(root), str(snapshot)))
    def execute(label, arguments):
        check_sources()
        with (artifacts / (label + ".log")).open("x") as log:
            result = subprocess.run([sys.executable, "-B", *map(str, arguments)],
                                    cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        check_sources()
        return result.returncode
    write_status()
    for short, task, count in datasets:
        record = dict(state="running", expected_cases=count, started_utc=now())
        status["datasets"][short] = record
        write_status()
        report = artifacts / (short + ".json")
        result = execute(short, ["-m", "rankseg_benchmark.nnunet.screening_benchmark",
            "--manifest", root / f"configs/nnunet/{task}_ensemble_oof.yaml", "--rankseg-path", snapshot,
            "--cases-per-fold", "0", "--warmup", "2", "--repeats", "5", "--threads", "4",
            "--production-dispatch", "--oracle", "bounded", "--objective-failure-policy", "record",
            "--output", report])
        record.update(benchmark_exit=result, benchmark_finished_utc=now())
        if result:
            record["state"] = "benchmark_failed"
            write_status()
            print(f"{short}: benchmark failed", flush=True)
            continue
        data = json.loads(report.read_text())
        if not data["complete"] or len(data["records"]) != count:
            raise RuntimeError(f"Incomplete cohort: {short}")
        result = execute(short + ".audit", ["-m", "rankseg_benchmark.nnunet.screening_audit", report,
            "--reference-csv", workspace_root(root) / f"outputs/{task}_ensemble_oof/case_label_metrics.csv",
            "--published-summary", root / f"evidence/nnunet/datasets/{task}/summary.json",
            "--methods", "argmax", "full_optimized", "screened", "--report-objective-failures"])
        record["audit_exit"] = result
        if not result:
            audit = json.loads((artifacts / (short + ".audit.log")).read_text())
            (artifacts / (short + ".audit.json")).write_text(json.dumps(audit, indent=2) + "\n")
            record.update(validation=audit["audit"], objective_failures=audit["objective_failures"])
        record["state"] = "measurements_complete" if not result else "audit_failed"
        write_status()
        print(f"{short}: complete; audit exit {result}", flush=True)
    # Untimed second pass: keep it separate from all decoder timings.
    for short, task, count in datasets:
        record = status["datasets"][short]
        if record.get("benchmark_exit"):
            continue
        result = execute(short + ".proportions", [root / "scripts/nnunet/collect_screening_proportions.py",
            "--benchmark-report", artifacts / (short + ".json"), "--rankseg-path", snapshot,
            "--output", artifacts / (short + ".proportions.json")])
        record["proportions_exit"] = result
        write_status()
        print(f"{short}: screening counts exit {result}", flush=True)
    status.update(complete=True, finished_utc=now())
    write_status()


if __name__ == "__main__":
    main()
