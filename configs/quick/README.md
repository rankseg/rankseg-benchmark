# Quick dataset configuration

The existing typed dataset registry is in
[`rankseg_benchmark/quick/datasets.py`](../../rankseg_benchmark/quick/datasets.py).
Use `rankseg-bench quick --list-datasets` to list hosted targets. It is not
duplicated into YAML: preserving one source of truth avoids changing the
established dataset/channel routing during this reorganization.
