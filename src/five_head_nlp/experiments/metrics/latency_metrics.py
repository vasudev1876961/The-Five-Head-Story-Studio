"""Latency and generation speed benchmarking metrics."""

from __future__ import annotations

import statistics
from typing import Any, Dict, Sequence


def calculate_latency_metrics(
    generation_time_sec: float,
    token_latencies_ms: Sequence[float],
) -> Dict[str, Any]:
    """Calculate generation duration, token latency distribution, and throughput.

    Benchmarking uses time.perf_counter() around each token prediction event,
    recording isolated per-token evaluation latency in milliseconds.

    Parameters
    ----------
    generation_time_sec : float
        Total story generation duration in seconds.
    token_latencies_ms : Sequence[float]
        Per-token prediction latencies in milliseconds.

    Returns
    -------
    Dict[str, Any]
        Dictionary of latency metrics.
    """
    n_tokens = len(token_latencies_ms)
    if n_tokens == 0:
        return {
            "generation_time_sec": round(generation_time_sec, 4),
            "tokens_measured": 0,
            "mean_token_latency_ms": 0.0,
            "median_token_latency_ms": 0.0,
            "std_token_latency_ms": 0.0,
            "p95_token_latency_ms": 0.0,
            "tokens_per_second": 0.0,
        }

    sorted_lats = sorted(token_latencies_ms)
    mean_lat = float(statistics.mean(token_latencies_ms))
    median_lat = float(statistics.median(token_latencies_ms))
    std_lat = float(statistics.stdev(token_latencies_ms)) if n_tokens > 1 else 0.0

    p95_idx = int(0.95 * n_tokens)
    p95_lat = float(sorted_lats[min(p95_idx, n_tokens - 1)])

    tps = (n_tokens / generation_time_sec) if generation_time_sec > 0 else 0.0

    return {
        "generation_time_sec": round(generation_time_sec, 4),
        "tokens_measured": n_tokens,
        "mean_token_latency_ms": round(mean_lat, 4),
        "median_token_latency_ms": round(median_lat, 4),
        "std_token_latency_ms": round(std_lat, 4),
        "p95_token_latency_ms": round(p95_lat, 4),
        "tokens_per_second": round(tps, 2),
    }
