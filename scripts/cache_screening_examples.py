"""Cache a fixed prefix of pinned public probabilities without outcome selection."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REVISION = "1884f0766268cdc62730e696f24dcc913d551b35"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["pascal_voc", "cityscapes", "ade20k"],
                        choices=["pascal_voc", "cityscapes", "ade20k"])
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    filesystem = HfFileSystem()
    for dataset in args.datasets:
        destination = args.output_dir / f"{dataset}-first{args.count}.parquet"
        partial = destination.with_suffix(".partial.parquet")
        metadata = destination.with_suffix(".source.json")
        if destination.exists() or partial.exists() or metadata.exists():
            raise FileExistsError(f"Refusing to overwrite existing cache: {destination}")
        source = f"datasets/ZixunWang/rankseg-benchmark@{REVISION}/{dataset}/test-00000.parquet"
        count = 0
        writer = None
        try:
            with filesystem.open(source, "rb", block_size=1024 * 1024) as handle:
                parquet = pq.ParquetFile(handle)
                for group in range(parquet.metadata.num_row_groups):
                    table = parquet.read_row_group(group).slice(0, args.count - count)
                    if writer is None:
                        writer = pq.ParquetWriter(partial, table.schema, compression="zstd")
                    writer.write_table(table)
                    count += table.num_rows
                    print(f"{dataset}: {count}/{args.count}", flush=True)
                    if count == args.count:
                        break
        finally:
            if writer is not None:
                writer.close()
        if count != args.count:
            raise ValueError(f"Only {count} rows available; partial cache retained, not marked complete")
        partial.rename(destination)
        metadata.write_text(json.dumps({"dataset": dataset, "revision": REVISION, "source": source,
                                        "rows": count, "selection": "first stored rows; no quality/performance selection"},
                                       indent=2) + "\n")


if __name__ == "__main__":
    main()
