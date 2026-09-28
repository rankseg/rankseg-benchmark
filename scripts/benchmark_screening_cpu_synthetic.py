"""CPU boundary cases complementing the real-cache screening benchmark."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from rankseg_benchmark.common.paths import installed_rankseg_root

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("synthetic_cpu_helpers", ROOT / "scripts/benchmark_screening_cpu.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
SCENARIOS = ("sparse", "uniform", "dense_candidates", "all_pruned")


def make_probabilities(scenario, channels, dim, seed=3401):
    p = torch.rand((1, channels, dim), generator=torch.Generator().manual_seed(seed))
    if scenario == "sparse":
        return torch.where(p > .99, .99, 1e-5)
    if scenario == "uniform":
        return p
    if scenario == "dense_candidates":
        p.fill_(.49)
        p[..., 0] = .51  # Active class but almost every entry remains a candidate.
        return p
    if scenario == "all_pruned":
        return p * .4
    raise ValueError("Unknown scenario")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--dims", type=int, nargs="+", default=[64, 4096, 65536, 262144, 1048576])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.threads + args.dims + [args.repeats]) < 1 or args.warmup < 0:
        parser.error("Invalid counts")
    oracle, counts = bench.acceptance.load_helpers(installed_rankseg_root(), ROOT)
    frozen = bench.source_hashes(installed_rankseg_root())
    report = dict(complete=False, device="cpu", torch=torch.__version__, source_hashes=frozen,
                  harness_sha256=bench.acceptance.sha256(__file__), warmup=args.warmup,
                  repeats=args.repeats, records=[])
    for threads in args.threads:
        torch.set_num_threads(threads)
        for channels in (1, 3):
            mode = "multilabel" if channels == 1 else "multiclass"
            decoders = {m: bench.acceptance.make_decoder(m, mode, oracle) for m in ("full", "screened")}
            binary = {m: bench.acceptance.make_decoder(m, "multilabel", oracle) for m in decoders}
            for dim in args.dims:
                for scenario in SCENARIOS:
                    p = make_probabilities(scenario, channels, dim)
                    before = bench.tensor_hash(p)
                    operations = {m: lambda d=d: d.predict(p) for m, d in decoders.items()}
                    values, predictions = bench.measure(operations, args.warmup, args.repeats, len(report["records"]))
                    row = dict(threads=threads, channels=channels, dim=dim, scenario=scenario, output_mode=mode,
                               methods=values, different_pixels=int((predictions["full"] != predictions["screened"]).sum()),
                               speedup=values["full"]["median_ms"] / values["screened"]["median_ms"])
                    del predictions
                    row["screening_counts"] = counts.summarize([counts.screening_counts(p, list(range(channels)))])
                    optimum = oracle.bounded_binary_oracle(p)
                    for method, decoder in binary.items():
                        mask = decoder.predict(p)
                        try:
                            diag = oracle.objective_regret(p, mask, optimum)
                            values[method]["objective"] = {**diag, "status": "passed"}
                        except oracle.ObjectiveBudgetExceeded as error:
                            values[method]["objective"] = {**error.diagnostics, "status": "failed"}
                        del mask
                    if bench.tensor_hash(p) != before:
                        raise AssertionError("Input modified")
                    report["records"].append(row)
    if bench.source_hashes(installed_rankseg_root()) != frozen:
        raise AssertionError("RankSEG source changed")
    report["complete"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
