"""Lazy suite dispatcher, retaining the original flag-only command syntax."""
from __future__ import annotations

import argparse
import sys


def build_parser():
    parser = argparse.ArgumentParser(
        prog="rankseg-bench",
        description="Compare RankSEG and argmax using Quick, MONAI, or nnU-Net experiments.",
        epilog="Legacy usage remains supported: rankseg-bench --dataset pascal_voc --limit 5",
    )
    parser.add_argument("suite", choices=("quick", "monai", "nnunet"), nargs="?")
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv in (["--help"], ["-h"]):
        build_parser().print_help()
        return 0
    if argv[0] == "quick":
        from .quick.cli import main as run
        return run(argv[1:], source_type="huggingface")
    if argv[0] == "monai":
        from .monai.cli import main as run
        return run(argv[1:])
    if argv[0] == "nnunet":
        from .nnunet.cli import main as run
        return run(argv[1:])
    if argv[0].startswith("-"):
        from .quick.cli import main as run
        return run(argv)
    build_parser().error(f"unknown suite: {argv[0]}")


if __name__ == "__main__":
    raise SystemExit(main())
