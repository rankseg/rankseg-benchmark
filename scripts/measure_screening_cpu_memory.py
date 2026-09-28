"""Sample incremental CPU RSS in a fresh subprocess per method and repetition.

This is NOT PyTorch allocated memory or a precise allocator peak. Inputs and
imports are resident before baseline; outputs are held until the final sample.
Run separately from the latency benchmark so sampling cannot affect its times.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path
from rankseg_benchmark.common.paths import installed_rankseg_root
import statistics
import subprocess
import sys
import threading
import time

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cpu_memory_helpers", ROOT / "scripts/benchmark_screening_cpu.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


@torch.no_grad()
def worker(args):
    oracle, _ = bench.acceptance.load_helpers(installed_rankseg_root(), ROOT)
    from rankseg_benchmark.datasets import REGISTRY
    from rankseg_benchmark.runner import _rankseg_output_mode, _rankseg_predict
    torch.set_num_threads(args.threads)
    # Initialize CPU workers on a small tensor, not on the measured decoder.
    _ = torch.ones(65536).sum().item()
    data = torch.load(args.worker, map_location="cpu", weights_only=True)
    p = data["probs"]
    dataset = REGISTRY[data["dataset"]]
    mode = _rankseg_output_mode(dataset, "RMA")
    decoder = bench.acceptance.make_decoder(args.method, mode, oracle)
    fingerprint = bench.tensor_hash(p)
    gc.collect()
    process = psutil.Process()
    print(json.dumps({"baseline_rss": process.memory_info().rss, "input_sha256": fingerprint,
                      "shape": list(p.shape), "identity": data["identity"]}), flush=True)
    if sys.stdin.readline().strip() != "start":
        raise RuntimeError("Expected parent start signal")
    prediction = _rankseg_predict(decoder, p, spec=dataset, rankseg_output_mode=mode)
    print(json.dumps({"output_rss": process.memory_info().rss, "output_elements": prediction.numel()}), flush=True)
    if sys.stdin.readline().strip() != "done":
        raise RuntimeError("Expected parent completion signal")
    if bench.tensor_hash(p) != fingerprint:
        raise AssertionError("Input changed")
    assert not torch.cuda.is_initialized(), "CPU benchmark unexpectedly initialized CUDA"


def measure_one(path, method, threads):
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(path),
                                "--method", method, "--threads", str(threads)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    stopped = threading.Event()
    samples = []
    sampling_errors = []
    def sample():
        try:
            monitor = psutil.Process(process.pid)
            while not stopped.is_set():
                samples.append((time.perf_counter(), monitor.memory_info().rss))
                stopped.wait(.001)
        except Exception as error:
            sampling_errors.append(error)
    monitor_thread = None
    try:
        baseline = json.loads(process.stdout.readline())
        monitor_thread = threading.Thread(target=sample)
        monitor_thread.start()
        process.stdin.write("start\n")
        process.stdin.flush()
        final = json.loads(process.stdout.readline())
        stopped.set()
        monitor_thread.join()
        if sampling_errors:
            raise RuntimeError("RSS sampler failed") from sampling_errors[0]
        process.stdin.write("done\n")
        process.stdin.flush()
        if process.wait(timeout=60) != 0:
            raise RuntimeError("Memory worker failed")
        peak = max([baseline["baseline_rss"], final["output_rss"]] + [rss for _, rss in samples])
        return {**baseline, **final, "method": method, "peak_observed_rss": peak,
                "incremental_peak_mib": (peak - baseline["baseline_rss"]) / 2**20,
                "rss_samples": len(samples),
                "largest_sampling_gap_ms": max([1000 * (b[0] - a[0]) for a, b in zip(samples, samples[1:])],
                                               default=0.)}
    finally:
        stopped.set()
        if monitor_thread is not None:
            monitor_thread.join()
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=30)
        process.stdin.close()
        process.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=("full", "screened"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 1:
        parser.error("Threads and repeats must be positive")
    if args.worker:
        if args.method is None:
            parser.error("Worker needs method")
        worker(args)
        return
    if args.input_dir is None:
        parser.error("Provide input-dir")
    destination = args.input_dir / "memory.json"
    if destination.exists():
        raise FileExistsError(destination)
    timing = json.loads((args.input_dir / "summary.json").read_text())
    frozen = bench.source_hashes(installed_rankseg_root())
    if not timing["complete"] or frozen != timing["source_hashes"] or timing["torch"] != torch.__version__:
        raise ValueError("Timing run incomplete or source/environment mismatch")
    report = dict(complete=False, threads=args.threads, repeats=args.repeats, source_hashes=frozen,
                  torch=torch.__version__, harness_sha256=bench.acceptance.sha256(__file__),
                  measurement="1ms RSS sampling, fresh process, resident input; incremental observed peak",
                  selection="first timed sample per dataset, not cohort-wide memory average", results=[])
    for dataset in dict.fromkeys(item["dataset"] for item in timing["results"]):
        path = args.input_dir / f"{dataset}-memory-input.pt"
        rows = []
        for repeat in range(args.repeats):
            for method in (("full", "screened") if repeat % 2 == 0 else ("screened", "full")):
                rows.append(measure_one(path, method, args.threads))
        if len({row["input_sha256"] for row in rows}) != 1:
            raise AssertionError("Memory comparison inputs differ")
        summary = {method: statistics.median(r["incremental_peak_mib"] for r in rows if r["method"] == method)
                   for method in ("full", "screened")}
        report["results"].append(dict(dataset=dataset, records=rows, median_incremental_peak_mib=summary))
    if bench.source_hashes(installed_rankseg_root()) != frozen:
        raise AssertionError("RankSEG source changed")
    report["complete"] = True
    with destination.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
