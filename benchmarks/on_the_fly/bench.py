"""Benchmark the core against the on-the-fly prototype (issue #41).

Each run happens in a fresh process, so that the peak memory (RSS) of one
run does not affect another. The data are standard normal with 8 features.

Usage (from the repository root)::

    PYTHONPATH=src python benchmarks/on_the_fly/bench.py cpu > cpu.jsonl
    PYTHONPATH=src python benchmarks/on_the_fly/bench.py gpu > gpu.jsonl

Modes:
- single: n_jobs=None and one BLAS thread for both implementations (the
  core does not limit BLAS threads itself, so it is limited here).
- parallel: n_jobs=-1 for both implementations.
- gpu: device="mps" or "cuda".
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
N_FEATURES = 8
SIZES_BOTH = (1000, 5000, 10000, 20000)
SIZES_PROTOTYPE_ONLY = (40000,)
# On a GPU the prototype launches a small kernel and copies back one row per
# step, which is very slow (37 s for 2,000 samples on MPS), so it is only
# run up to this size there.
MAX_GPU_PROTOTYPE_SIZE = 10000
GROUPS = (1, 5)


def _peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # bytes on macOS, kilobytes on Linux
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def _child(impl: str, n_samples: int, n_groups: int, mode: str) -> None:
    import logging

    import numpy as np
    from threadpoolctl import threadpool_limits

    sys.path.insert(0, str(HERE))
    from _prototype import kennard_stone_on_the_fly

    from kennard_stone._core._core import _KennardStone

    logging.disable(logging.CRITICAL)
    device = "cpu"
    if mode == "gpu":
        import torch

        device = "cuda" if torch.cuda.is_available() else "mps"
    n_jobs = -1 if mode == "parallel" else None

    def run(X):
        if impl == "core":
            ks = _KennardStone(n_groups=n_groups, n_jobs=n_jobs, device=device)
            if mode == "single":
                with threadpool_limits(limits=1):
                    return ks.get_indexes(X)
            return ks.get_indexes(X)
        return kennard_stone_on_the_fly(
            X, n_groups=n_groups, n_jobs=n_jobs, device=device
        )

    rng = np.random.RandomState(0)
    run(rng.standard_normal((200, N_FEATURES)))  # warm up imports/devices
    X = rng.standard_normal((n_samples, N_FEATURES))
    baseline = _peak_rss_mb()

    start = time.perf_counter()
    groups = run(X)
    elapsed = time.perf_counter() - start

    digest = hashlib.sha1(
        json.dumps([[int(i) for i in g] for g in groups]).encode()
    ).hexdigest()[:12]
    print(
        json.dumps(
            {
                "impl": impl,
                "mode": mode,
                "device": device,
                "n_samples": n_samples,
                "n_groups": n_groups,
                "seconds": round(elapsed, 2),
                "peak_rss_mb": round(_peak_rss_mb(), 1),
                "rss_increase_mb": round(_peak_rss_mb() - baseline, 1),
                "result": digest,
            }
        ),
        flush=True,
    )


def _driver(kind: str) -> None:
    modes = ("single", "parallel") if kind == "cpu" else ("gpu",)
    configs = [
        (impl, n, g, mode)
        for mode in modes
        for g in GROUPS
        for n in SIZES_BOTH + SIZES_PROTOTYPE_ONLY
        for impl in ("core", "prototype")
        if not (impl == "core" and n in SIZES_PROTOTYPE_ONLY)
        if not (
            mode == "gpu"
            and impl == "prototype"
            and n > MAX_GPU_PROTOTYPE_SIZE
        )
    ]
    for impl, n, g, mode in configs:
        result = subprocess.run(
            [sys.executable, __file__, "child", impl, str(n), str(g), mode],
            capture_output=True,
            text=True,
            env=os.environ,
        )
        if result.returncode != 0:
            print(
                json.dumps(
                    {
                        "impl": impl,
                        "mode": mode,
                        "n_samples": n,
                        "n_groups": g,
                        "error": result.stderr[-500:],
                    }
                ),
                flush=True,
            )
        else:
            print(result.stdout.strip().splitlines()[-1], flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "child":
        _child(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
    else:
        _driver(sys.argv[1])
