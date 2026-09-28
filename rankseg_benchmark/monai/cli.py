"""MONAI cache generation, evaluation, and demo commands."""
import argparse
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="rankseg-bench monai")
    parser.add_argument("command", choices=("cache", "evaluate", "demo"))
    if not argv or argv in (["--help"], ["-h"]):
        parser.print_help()
        return 0
    command = parser.parse_args(argv[:1]).command
    if command == "cache":
        from .cache import main as run
    elif command == "demo":
        from .demo import main as run
    else:
        from ..quick.cli import main as run
        return run(argv[1:], source_type="local_artifacts")
    return run(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
