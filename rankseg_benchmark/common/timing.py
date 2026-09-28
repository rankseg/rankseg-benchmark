"""Timing utilities with CUDA sync + warmup."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


@dataclass
class Timer:
    """Accumulates wall-clock time across many samples, CUDA-aware."""

    use_cuda: bool = False
    total_seconds: float = 0.0
    n_calls: int = 0
    n_batches: int = 0
    per_call: list[float] = field(default_factory=list)

    @contextmanager
    def measure(self, units: int = 1):
        if units < 1:
            raise ValueError(f"Timer units must be >= 1, got {units}")
        if self.use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.use_cuda:
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            self.total_seconds += dt
            self.n_calls += units
            self.n_batches += 1
            self.per_call.extend([dt / units] * units)

    @property
    def mean_ms(self) -> float:
        return 1000.0 * self.total_seconds / max(self.n_calls, 1)

    @property
    def median_ms(self) -> float:
        if not self.per_call:
            return 0.0
        s = sorted(self.per_call)
        return 1000.0 * s[len(s) // 2]
