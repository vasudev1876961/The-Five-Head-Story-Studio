"""Safe Experiment Reproducibility Harness for Phase 5 Monte Carlo Experiments.

Usage:
    python -m five_head_nlp.experiments.reproduce_all [options]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from five_head_nlp.experiments.aggregation import (
    aggregate_config_metrics,
    export_aggregate_metrics_csv,
    export_per_story_metrics_csv,
    flatten_story_metrics,
)
from five_head_nlp.experiments.experiment_config import ExperimentConfig
from five_head_nlp.experiments.experiment_registry import (
    get_experiment_config,
    list_all_experiments,
)
from five_head_nlp.experiments.generation_harness import GenerationHarness
from five_head_nlp.experiments.story_generator import GeneratedStory

# ---------------------------------------------------------------------------
# Canonical Configuration & Pre-Registered Statistical Equivalence Constants
# ---------------------------------------------------------------------------

CANONICAL_ORDER: Tuple[str, ...] = ("E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8")
CANONICAL_INDEX_MAP: Dict[str, int] = {cid: idx for idx, cid in enumerate(CANONICAL_ORDER)}

# Verified by source inspection (greedy argmax + seeded starter selection, zero decoder RNG):
DETERMINISTIC_CONFIGS: frozenset[str] = frozenset({"E1", "E2", "E3", "E4", "E5", "E7"})
# Verified by source inspection (TopKDecoder k=5 and TemperatureDecoder T=0.8 use weighted RNG sampling):
STOCHASTIC_CONFIGS: frozenset[str] = frozenset({"E6", "E8"})

DEFAULT_BASE_SEED = 42
DEFAULT_STOCHASTIC_SEED = 1042  # Disjoint seed yielding independent Monte Carlo trajectories
DEFAULT_STORIES_PER_CONFIG = 200
MIN_STATISTICAL_SAMPLE_SIZE = 15
FAMILY_ALPHA = 0.05


@dataclass(frozen=True)
class EquivalenceMarginSpec:
    """Pre-specified practical equivalence bound (SESOI) for a metric."""

    metric: str
    margin: float
    rationale: str


# Margins defined strictly BEFORE examining any reproduced stochastic samples,
# anchored to domain scale and canonical Phase 5 N=200 dispersion:
PREDEFINED_EQUIVALENCE_MARGINS: Tuple[EquivalenceMarginSpec, ...] = (
    EquivalenceMarginSpec(
        metric="length_total_tokens",
        margin=4.0,
        rationale="10% of the [80, 120] token target window (canonical E6/E8 SD ~ 11.1-11.2 tokens).",
    ),
    EquivalenceMarginSpec(
        metric="repetition_repeated_bigram_rate",
        margin=0.025,
        rationale="2.5 percentage-point tolerance across [0, 1] scale (canonical E1-E8 span 0.006-0.660).",
    ),
    EquivalenceMarginSpec(
        metric="diversity_ttr",
        margin=0.020,
        rationale="0.020 TTR units (small vs canonical E5=0.221, E6=0.289, E8=0.678).",
    ),
    EquivalenceMarginSpec(
        metric="setting_setting_consistency_ratio",
        margin=0.060,
        rationale="6.0 percentage-point tolerance calibrated for canonical E6 SD=0.202 at N=200.",
    ),
    EquivalenceMarginSpec(
        metric="action_dynamic_action_ratio",
        margin=0.040,
        rationale="4.0 percentage-point tolerance calibrated for canonical E8 SD=0.103 at N=200.",
    ),
    EquivalenceMarginSpec(
        metric="character_character_continuity_ratio",
        margin=0.090,
        rationale=(
            "9.0 percentage-point tolerance (<0.29 SD of canonical E6 SD=0.312 at N=200; "
            "power limitation noted for high-variance continuity events)."
        ),
    ),
)


def find_repo_root() -> Path:
    """Locate project root containing pyproject.toml and canonical data."""
    return Path(__file__).resolve().parents[3]


def sha256_file(path: Path) -> str:
    """Compute uppercase hex SHA256 digest of a file without modifying it."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _normalize_config_dict(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize config dictionary for hash comparison (sorting set-derived lexicon lists)."""
    out = dict(cfg_dict)
    sc = out.get("scoring_config")
    if isinstance(sc, dict):
        sc_norm = dict(sc)
        for lex_key in ("character_lexicon", "setting_lexicon", "action_lexicon"):
            if isinstance(sc_norm.get(lex_key), list):
                sc_norm[lex_key] = sorted(sc_norm[lex_key])
        out["scoring_config"] = sc_norm
    return out


def canonical_config_hash(configs: Dict[str, Dict[str, Any]]) -> str:
    """Compute order-independent SHA256 digest of normalized experimental configs."""
    normalized = {k: _normalize_config_dict(configs[k]) for k in sorted(configs.keys())}
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def validate_safe_output_dir(output_dir: Path, repo_root: Path) -> Path:
    """Ensure output directory is strictly isolated from canonical protected locations."""
    resolved_target = output_dir.resolve()
    resolved_root = repo_root.resolve()

    if resolved_target == resolved_root:
        raise ValueError(f"Refusing to write directly to repository root: {resolved_target}")

    protected_dirs = [
        (resolved_root / "experiments" / "results").resolve(),
        (resolved_root / "experiments" / "models").resolve(),
        (resolved_root / "data").resolve(),
        (resolved_root / "phase_7").resolve(),
        (resolved_root / "src").resolve(),
        (resolved_root / "tests").resolve(),
        (resolved_root / ".git").resolve(),
    ]

    for pdir in protected_dirs:
        if resolved_target == pdir or pdir in resolved_target.parents or resolved_target in pdir.parents:
            raise ValueError(
                f"Unsafe output directory {resolved_target}: overlaps protected path {pdir}"
            )

    return resolved_target


# ---------------------------------------------------------------------------
# Pure-Python Student-t & Statistical Equivalence Engine (TOST + Holm)
# ---------------------------------------------------------------------------


def _betacf(a: float, b: float, x: float) -> float:
    """Evaluate continued fraction for incomplete beta function (Lentz's method)."""
    max_iter = 200
    eps = 3.0e-14
    fpmin = 1.0e-300

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break

    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Compute regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    bt = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def student_t_cdf(t_val: float, df: float) -> float:
    """Compute exact CDF P(T <= t_val) for Student's t-distribution with df > 0."""
    if df <= 0.0:
        raise ValueError(f"Degrees of freedom must be positive, got {df}")
    if math.isinf(t_val):
        return 1.0 if t_val > 0 else 0.0
    if t_val == 0.0:
        return 0.5
    x = df / (df + t_val * t_val)
    ib = regularized_incomplete_beta(0.5 * df, 0.5, x)
    if t_val > 0.0:
        return 1.0 - 0.5 * ib
    return 0.5 * ib


def student_t_ppf(prob: float, df: float) -> float:
    """Compute inverse CDF (quantile) of Student's t-distribution via bisection."""
    if not (0.0 < prob < 1.0):
        raise ValueError(f"Probability must be in (0, 1), got {prob}")
    lo, hi = -100.0, 100.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if student_t_cdf(mid, df) < prob:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def evaluate_tost_metric(
    config_id: str,
    spec: EquivalenceMarginSpec,
    canon_values: Sequence[float],
    repro_values: Sequence[float],
    alpha: float = FAMILY_ALPHA,
) -> Dict[str, Any]:
    """Run Welch TOST equivalence test on a single metric without Holm adjustment."""
    n1 = len(canon_values)
    n2 = len(repro_values)

    if n1 < 2 or n2 < 2:
        return {
            "config_id": config_id,
            "metric": spec.metric,
            "n_canonical": n1,
            "n_reproduced": n2,
            "margin": spec.margin,
            "rationale": spec.rationale,
            "status": "INCONCLUSIVE",
            "reason": "Sample size < 2 prevents variance estimation.",
        }

    mean1 = statistics.mean(canon_values)
    mean2 = statistics.mean(repro_values)
    sd1 = statistics.stdev(canon_values)
    sd2 = statistics.stdev(repro_values)
    diff = mean2 - mean1

    se_sq = (sd1 * sd1) / n1 + (sd2 * sd2) / n2
    se = math.sqrt(se_sq)

    # Pooled SD & Hedges' g (bias-corrected standardized mean difference)
    pooled_var = (((n1 - 1) * sd1 * sd1) + ((n2 - 1) * sd2 * sd2)) / (n1 + n2 - 2)
    pooled_sd = math.sqrt(pooled_var)
    if pooled_sd > 0.0:
        d_cohen = diff / pooled_sd
        j_corr = 1.0 - 3.0 / (4.0 * (n1 + n2 - 2) - 1.0)
        hedges_g = j_corr * d_cohen
    else:
        hedges_g = 0.0

    if se == 0.0:
        return {
            "config_id": config_id,
            "metric": spec.metric,
            "n_canonical": n1,
            "n_reproduced": n2,
            "canonical_mean": round(mean1, 6),
            "canonical_sd": round(sd1, 6),
            "reproduced_mean": round(mean2, 6),
            "reproduced_sd": round(sd2, 6),
            "mean_diff": round(diff, 6),
            "se": 0.0,
            "hedges_g": round(hedges_g, 6),
            "margin": spec.margin,
            "rationale": spec.rationale,
            "status": "INCONCLUSIVE",
            "reason": "Zero standard error across compared samples.",
        }

    # Welch-Satterthwaite degrees of freedom
    denom_df = ((sd1 * sd1 / n1) ** 2) / (n1 - 1) + ((sd2 * sd2 / n2) ** 2) / (n2 - 1)
    df = (se_sq * se_sq) / denom_df

    # Two-sided Welch difference test H0: mu_repro == mu_canon
    t_diff = diff / se
    p_diff = 2.0 * (1.0 - student_t_cdf(abs(t_diff), df))

    # TOST one-sided statistics against [-margin, +margin]
    t_lower = (diff + spec.margin) / se  # H01: diff <= -margin
    t_upper = (diff - spec.margin) / se  # H02: diff >= +margin
    p_lower = 1.0 - student_t_cdf(t_lower, df)
    p_upper = student_t_cdf(t_upper, df)
    p_tost = max(p_lower, p_upper)

    # Nominal 90% Welch CI corresponding to alpha=0.05 TOST
    t_crit_90 = student_t_ppf(1.0 - alpha, df)
    ci90_lo = diff - t_crit_90 * se
    ci90_hi = diff + t_crit_90 * se

    status = "PENDING_CORRECTION"
    reason = ""
    if n2 < MIN_STATISTICAL_SAMPLE_SIZE:
        status = "INCONCLUSIVE"
        reason = f"Reproduced sample size {n2} < minimum {MIN_STATISTICAL_SAMPLE_SIZE} for equivalence power."

    return {
        "config_id": config_id,
        "metric": spec.metric,
        "n_canonical": n1,
        "n_reproduced": n2,
        "canonical_mean": round(mean1, 6),
        "canonical_sd": round(sd1, 6),
        "reproduced_mean": round(mean2, 6),
        "reproduced_sd": round(sd2, 6),
        "mean_diff": round(diff, 6),
        "se": round(se, 6),
        "welch_df": round(df, 2),
        "hedges_g": round(hedges_g, 6),
        "t_diff": round(t_diff, 4),
        "p_diff": p_diff,
        "margin": spec.margin,
        "t_lower": round(t_lower, 4),
        "p_lower": p_lower,
        "t_upper": round(t_upper, 4),
        "p_upper": p_upper,
        "p_tost": p_tost,
        "ci_90_nominal": [round(ci90_lo, 6), round(ci90_hi, 6)],
        "rationale": spec.rationale,
        "status": status,
        "reason": reason,
    }


def apply_holm_bonferroni(
    metric_results: List[Dict[str, Any]],
    alpha: float = FAMILY_ALPHA,
) -> List[Dict[str, Any]]:
    """Apply Holm-Bonferroni correction across all testable equivalence comparisons."""
    testable = [r for r in metric_results if "p_tost" in r and r["status"] != "INCONCLUSIVE"]
    m = len(testable)
    if m == 0:
        return metric_results

    # Order ascending by raw TOST p-value
    indexed = sorted(enumerate(testable), key=lambda x: x[1]["p_tost"])
    running_max = 0.0
    for rank, (_, item) in enumerate(indexed):
        factor = m - rank
        raw_adj = min(1.0, item["p_tost"] * factor)
        running_max = max(running_max, raw_adj)
        item["p_tost_holm"] = running_max

        # Simultaneous Bonferroni-conservative CI (1 - 2*alpha/m) for visual inspection
        t_crit_fam = student_t_ppf(1.0 - (alpha / m), item["welch_df"])
        item["ci_family_bonferroni"] = [
            round(item["mean_diff"] - t_crit_fam * item["se"], 6),
            round(item["mean_diff"] + t_crit_fam * item["se"], 6),
        ]
        item["family_ci_level_pct"] = round((1.0 - 2.0 * (alpha / m)) * 100.0, 2)

        margin = item["margin"]
        diff = item["mean_diff"]
        if running_max < alpha:
            item["status"] = "STATISTICAL_EQUIVALENCE"
            item["reason"] = (
                f"Holm-adjusted TOST p={running_max:.4g} < {alpha}; mean difference "
                f"bounded strictly within [-{margin}, +{margin}]."
            )
        elif abs(diff) >= margin and item["p_diff"] < alpha:
            item["status"] = "FAILED_REPRODUCTION"
            item["reason"] = (
                f"Difference {diff:+.4f} meets/exceeds margin +/-{margin} with p_diff={item['p_diff']:.4g}."
            )
        else:
            item["status"] = "SIMILAR_INSUFFICIENT_EQUIVALENCE"
            if abs(diff) < margin:
                item["reason"] = (
                    f"Point estimate inside margin ({diff:+.4f} vs +/-{margin}), "
                    f"but Holm TOST p={running_max:.4g} >= {alpha}; equivalence is NOT established."
                )
            else:
                item["reason"] = (
                    f"Mean difference ({diff:+.4f}) lies at/outside margin +/-{margin} with "
                    f"p_diff={item['p_diff']:.4g} >= {alpha}; equivalence fails."
                )

    return metric_results


# ---------------------------------------------------------------------------
# Canonical Audit & Comparison Loaders
# ---------------------------------------------------------------------------


def audit_canonical_artifacts(repo_root: Path) -> Dict[str, Any]:
    """Read and validate canonical Phase 5 corpus, configs, metadata, and hashes."""
    mc_dir = repo_root / "experiments" / "results" / "monte_carlo"
    corpus_path = repo_root / "data" / "raw" / "story_corpus.txt"
    configs_path = mc_dir / "configurations.json"
    stories_path = mc_dir / "generated_stories.jsonl"
    per_story_path = mc_dir / "per_story_metrics.csv"
    agg_path = mc_dir / "aggregate_metrics.csv"
    meta_path = mc_dir / "run_metadata.json"

    required = [corpus_path, configs_path, stories_path, per_story_path, agg_path, meta_path]
    for p in required:
        if not p.is_file():
            raise FileNotFoundError(f"Canonical Phase 5 artifact missing: {p}")

    with open(configs_path, "r", encoding="utf-8") as f:
        canon_configs: Dict[str, Dict[str, Any]] = json.load(f)

    with open(meta_path, "r", encoding="utf-8") as f:
        canon_meta: Dict[str, Any] = json.load(f)

    # Validate registry alignment (including E6 decoder k=5)
    reg_configs = {c.config_id: c.to_dict() for c in list_all_experiments()}
    mismatches: List[str] = []

    if set(canon_configs.keys()) != set(reg_configs.keys()):
        mismatches.append(
            f"Config ID mismatch: file={sorted(canon_configs.keys())} registry={sorted(reg_configs.keys())}"
        )

    for cid in CANONICAL_ORDER:
        if cid in canon_configs and cid in reg_configs:
            c_norm = _normalize_config_dict(canon_configs[cid])
            r_norm = _normalize_config_dict(reg_configs[cid])
            if c_norm != r_norm:
                mismatches.append(f"Config {cid} in configurations.json differs from EXPERIMENT_REGISTRY")

    if canon_configs.get("E6", {}).get("decoder_params", {}).get("k") != 5:
        mismatches.append(
            f"Unexpected canonical E6 parameter: {canon_configs.get('E6', {}).get('decoder_params')}"
        )

    if mismatches:
        raise ValueError("Canonical configuration audit failed: " + "; ".join(mismatches))

    return {
        "canonical_hashes": {
            "story_corpus.txt": sha256_file(corpus_path),
            "configurations.json": sha256_file(configs_path),
            "generated_stories.jsonl": sha256_file(stories_path),
            "per_story_metrics.csv": sha256_file(per_story_path),
            "aggregate_metrics.csv": sha256_file(agg_path),
            "run_metadata.json": sha256_file(meta_path),
        },
        "canonical_config_semantic_hash": canonical_config_hash(canon_configs),
        "registry_config_semantic_hash": canonical_config_hash(reg_configs),
        "canonical_metadata": canon_meta,
        "canonical_configs": canon_configs,
        "e6_verified_k": canon_configs["E6"]["decoder_params"]["k"],
    }


def load_canonical_stories(stories_path: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load canonical generated_stories.jsonl indexed by config_id -> story_id."""
    by_config: Dict[str, Dict[str, Dict[str, Any]]] = {}
    with open(stories_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_config.setdefault(rec["config_id"], {})[rec["story_id"]] = rec
    return by_config


def load_canonical_per_story_csv(csv_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load canonical per_story_metrics.csv rows grouped by config_id."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            grouped.setdefault(row["config_id"], []).append(row)
    return grouped


# ---------------------------------------------------------------------------
# Deterministic Story & Metric Exact Comparison
# ---------------------------------------------------------------------------


def _compare_metric_trees(
    canon_metrics: Dict[str, Any],
    repro_metrics: Dict[str, Any],
    tol: float = 1e-4,
) -> List[str]:
    """Compare non-timing story metrics dictionaries within numeric tolerance."""
    diffs: List[str] = []
    for cat, c_cat in canon_metrics.items():
        if cat == "latency":
            continue  # Hardware perf_counter timing excluded from deterministic equality
        if cat not in repro_metrics:
            diffs.append(f"Missing metric category {cat}")
            continue
        r_cat = repro_metrics[cat]
        if isinstance(c_cat, dict) and isinstance(r_cat, dict):
            for k, c_val in c_cat.items():
                r_val = r_cat.get(k)
                if isinstance(c_val, float) and isinstance(r_val, (int, float)):
                    if abs(c_val - float(r_val)) > tol:
                        diffs.append(f"{cat}.{k}: canon={c_val} vs repro={r_val}")
                elif c_val != r_val:
                    diffs.append(f"{cat}.{k}: canon={c_val!r} vs repro={r_val!r}")
    return diffs


def evaluate_deterministic_condition(
    config_id: str,
    repro_stories: Sequence[GeneratedStory],
    repro_failure_count: int,
    canon_story_map: Dict[str, Dict[str, Any]],
    expected_base_seed: int,
) -> Dict[str, Any]:
    """Verify exact token sequence, seed, and non-timing metric equality."""
    total = len(repro_stories)
    exact_token_matches = 0
    exact_text_matches = 0
    exact_metric_matches = 0
    mismatches: List[Dict[str, Any]] = []

    for s in repro_stories:
        canon = canon_story_map.get(s.story_id)
        if canon is None:
            mismatches.append({"story_id": s.story_id, "issue": "Missing in canonical JSONL"})
            continue

        story_diffs: List[str] = []
        if s.seed != canon["seed"]:
            story_diffs.append(f"seed canon={canon['seed']} vs repro={s.seed}")
        if s.tokens == canon["tokens"]:
            exact_token_matches += 1
        else:
            story_diffs.append("token sequence mismatch")

        if s.text == canon["text"]:
            exact_text_matches += 1
        else:
            story_diffs.append("text mismatch")

        if s.sentence_count != canon["sentence_count"]:
            story_diffs.append(f"sentence_count canon={canon['sentence_count']} vs repro={s.sentence_count}")
        if s.token_count != canon["token_count"]:
            story_diffs.append(f"token_count canon={canon['token_count']} vs repro={s.token_count}")

        m_diffs = _compare_metric_trees(canon.get("metrics", {}), s.metrics)
        if not m_diffs:
            exact_metric_matches += 1
        else:
            story_diffs.extend(m_diffs[:3])

        if story_diffs and len(mismatches) < 10:
            mismatches.append({"story_id": s.story_id, "issues": story_diffs})

    if expected_base_seed != DEFAULT_BASE_SEED:
        verdict = "FAILED_REPRODUCTION" if mismatches else "INCONCLUSIVE"
        note = f"Non-canonical base seed {expected_base_seed} used (canonical={DEFAULT_BASE_SEED})."
    elif total == 0:
        verdict = "INCONCLUSIVE"
        note = "Zero stories generated."
    elif (
        repro_failure_count == 0
        and exact_token_matches == total
        and exact_text_matches == total
        and exact_metric_matches == total
        and not mismatches
    ):
        verdict = "EXACT_REPRODUCTION"
        note = (
            f"All {total}/{total} stories matched canonical tokens, text, seeds, and "
            "all non-timing metric values identically."
        )
    else:
        verdict = "FAILED_REPRODUCTION"
        note = (
            f"Mismatch detected: tokens={exact_token_matches}/{total}, "
            f"metrics={exact_metric_matches}/{total}, failures={repro_failure_count}."
        )

    return {
        "config_id": config_id,
        "classification": "deterministic",
        "verdict": verdict,
        "stories_compared": total,
        "failure_count": repro_failure_count,
        "exact_token_matches": exact_token_matches,
        "exact_text_matches": exact_text_matches,
        "exact_metric_matches": exact_metric_matches,
        "sample_mismatches": mismatches,
        "note": note,
    }


# ---------------------------------------------------------------------------
# Report Rendering
# ---------------------------------------------------------------------------


def render_markdown_report(summary: Dict[str, Any]) -> str:
    """Render concise human-readable Markdown reproducibility report."""
    env = summary["environment"]
    audit = summary["canonical_audit"]
    lines: List[str] = [
        "# Phase 5 Reproducibility Verification Report",
        "",
        f"- **Run ID:** `{summary['run_id']}`",
        f"- **Overall Status:** **{summary['overall_verdict']}**",
        f"- **Deterministic Base Seed:** `{summary['base_seed']}` (for `{', '.join(summary['deterministic_configs'])}`)",
        f"- **Stochastic Comparison Seed:** `{summary['stochastic_seed']}` (for `{', '.join(summary['stochastic_configs'])}`)",
        f"- **Stories per Condition:** `{summary['stories_per_config']}` | **Total Generated:** `{summary['total_stories_generated']}` | **Failures:** `{summary['total_failures']}`",
        f"- **Wall Time:** `{summary['wall_time_sec']:.2f} s` (`{summary['throughput_stories_per_sec']:.2f} stories/sec`)",
        f"- **Canonical E6 Decoder Verified:** `top_k(k={audit['e6_verified_k']})`",
        "",
        "## 1. Environment & Configuration Hashes",
        "",
        f"- **Python:** `{env['python_version']}` (`{env['platform']}`)",
        f"- **NumPy / Pandas:** `numpy {env['numpy_version']}`, `pandas {env['pandas_version']}`",
        f"- **Canonical Config Hash:** `{audit['canonical_config_semantic_hash']}`",
        f"- **Reproduced Registry Hash:** `{audit['registry_config_semantic_hash']}`",
        "",
        "## 2. Configuration Verdicts",
        "",
        "| Config | Type | Seed Offset Rule | Verdict | Details |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for cid in summary["evaluated_configs"]:
        d_res = summary["deterministic_results"].get(cid)
        s_res = summary["stochastic_results"].get(cid)
        if d_res:
            lines.append(
                f"| **{cid}** | Deterministic | `base + {CANONICAL_INDEX_MAP[cid]}*10000 + i` | "
                f"**{d_res['verdict']}** | {d_res['exact_token_matches']}/{d_res['stories_compared']} exact token matches |"
            )
        elif s_res:
            lines.append(
                f"| **{cid}** | Stochastic | `stoch + {CANONICAL_INDEX_MAP[cid]}*10000 + i` | "
                f"**{s_res['verdict']}** | {s_res['note']} |"
            )

    if summary["skipped_configs"]:
        lines.extend([
            "",
            f"**Skipped Conditions (Partial Run):** `{', '.join(summary['skipped_configs'])}`",
        ])

    lines.extend([
        "",
        "## 3. Stochastic Statistical Equivalence Analysis (TOST + Holm-Bonferroni)",
        "",
        f"Family-wise error rate $\\alpha = {FAMILY_ALPHA}$ across $m = {summary['tost_comparison_count']}$ pre-registered tests.",
        "",
        "| Config | Metric | Canon Mean (SD) | Repro Mean (SD) | Diff | Hedges' g | Margin +/- | Holm TOST p | Verdict |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for item in summary["tost_tests"]:
        if "canonical_mean" not in item:
            lines.append(
                f"| {item['config_id']} | `{item['metric']}` | - | - | - | - | +/-{item['margin']} | - | **{item['status']}** |"
            )
            continue
        holm_p = item.get("p_tost_holm")
        holm_str = f"{holm_p:.4g}" if isinstance(holm_p, float) else "-"
        lines.append(
            f"| {item['config_id']} | `{item['metric']}` | "
            f"{item['canonical_mean']:.4f} ({item['canonical_sd']:.4f}) | "
            f"{item['reproduced_mean']:.4f} ({item['reproduced_sd']:.4f}) | "
            f"{item['mean_diff']:+.4f} | {item['hedges_g']:+.3f} | +/-{item['margin']} | "
            f"{holm_str} | **{item['status']}** |"
        )

    if summary["warnings"]:
        lines.extend(["", "## 4. Warnings & Limitations", ""])
        for w in summary["warnings"]:
            lines.append(f"- {w}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main Harness Runner & CLI
# ---------------------------------------------------------------------------


def run_reproducibility(
    base_seed: int = DEFAULT_BASE_SEED,
    stochastic_seed: int = DEFAULT_STOCHASTIC_SEED,
    stories_per_config: int = DEFAULT_STORIES_PER_CONFIG,
    configs: Optional[Sequence[str]] = None,
    run_id: Optional[str] = None,
    output_root: Optional[Path] = None,
    repo_root: Optional[Path] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Execute safe Phase 5 reproducibility verification and write isolated artifacts."""
    if stories_per_config < 1:
        raise ValueError(f"stories_per_config must be >= 1, got {stories_per_config}")

    root = repo_root.resolve() if repo_root else find_repo_root()
    audit = audit_canonical_artifacts(root)

    if configs:
        selected_configs = [get_experiment_config(cid) for cid in configs]
    else:
        selected_configs = list_all_experiments()

    evaluated_ids = [c.config_id for c in selected_configs]
    skipped_ids = [cid for cid in CANONICAL_ORDER if cid not in evaluated_ids]

    if run_id is None:
        suffix = f"_{'_'.join(evaluated_ids)}" if configs else ""
        run_id = f"run_b{base_seed}_s{stochastic_seed}_n{stories_per_config}{suffix}"

    parent_dir = output_root.resolve() if output_root else (root / "reports" / "reproducibility")
    target_dir = validate_safe_output_dir(parent_dir / run_id, root)

    if dry_run:
        return {
            "status": "DRY_RUN_VALID",
            "run_id": run_id,
            "target_dir": str(target_dir),
            "evaluated_configs": evaluated_ids,
            "skipped_configs": skipped_ids,
            "e6_verified_k": audit["e6_verified_k"],
            "canonical_config_semantic_hash": audit["canonical_config_semantic_hash"],
        }

    target_dir.mkdir(parents=True, exist_ok=True)

    # Record reproduced configurations JSON
    repro_cfg_dicts = {c.config_id: c.to_dict() for c in selected_configs}
    with open(target_dir / "reproduced_configurations.json", "w", encoding="utf-8") as f:
        json.dump(repro_cfg_dicts, f, indent=2)

    det_harness = GenerationHarness(base_seed=base_seed)
    stoch_harness = GenerationHarness(base_seed=stochastic_seed, loader=det_harness.loader, model=det_harness.model)

    canon_stories = load_canonical_stories(
        root / "experiments" / "results" / "monte_carlo" / "generated_stories.jsonl"
    )
    canon_csv = load_canonical_per_story_csv(
        root / "experiments" / "results" / "monte_carlo" / "per_story_metrics.csv"
    )

    all_stories: List[GeneratedStory] = []
    all_failures = []
    det_results: Dict[str, Dict[str, Any]] = {}
    tost_tests: List[Dict[str, Any]] = []
    warnings: List[str] = []

    if stochastic_seed == base_seed:
        warnings.append(
            f"stochastic_seed ({stochastic_seed}) equals base_seed ({base_seed}): "
            "stochastic runs represent deterministic PRNG replay, NOT independent Monte Carlo sampling."
        )

    if skipped_ids:
        warnings.append(f"Partial run executed; skipped canonical configs: {skipped_ids}.")

    t_start = time.perf_counter()

    for cfg in selected_configs:
        cid = cfg.config_id
        c_idx = CANONICAL_INDEX_MAP[cid]
        if cid in DETERMINISTIC_CONFIGS:
            res = det_harness.run_configuration(
                config=cfg,
                num_stories=stories_per_config,
                config_index=c_idx,
            )
            all_stories.extend(res.stories)
            all_failures.extend(res.failures)
            det_results[cid] = evaluate_deterministic_condition(
                config_id=cid,
                repro_stories=res.stories,
                repro_failure_count=len(res.failures),
                canon_story_map=canon_stories.get(cid, {}),
                expected_base_seed=base_seed,
            )
        else:
            res = stoch_harness.run_configuration(
                config=cfg,
                num_stories=stories_per_config,
                config_index=c_idx,
            )
            all_stories.extend(res.stories)
            all_failures.extend(res.failures)

            canon_rows = canon_csv.get(cid, [])
            repro_flat = [flatten_story_metrics(s) for s in res.stories]
            for spec in PREDEFINED_EQUIVALENCE_MARGINS:
                c_vals = [float(r[spec.metric]) for r in canon_rows if spec.metric in r]
                r_vals = [float(r[spec.metric]) for r in repro_flat if spec.metric in r]
                tost_tests.append(evaluate_tost_metric(cid, spec, c_vals, r_vals))

    wall_time = time.perf_counter() - t_start

    # Holm-Bonferroni family correction across all stochastic metric tests
    apply_holm_bonferroni(tost_tests, alpha=FAMILY_ALPHA)

    stoch_results: Dict[str, Dict[str, Any]] = {}
    for cid in evaluated_ids:
        if cid in STOCHASTIC_CONFIGS:
            cid_tests = [t for t in tost_tests if t["config_id"] == cid]
            statuses = {t["status"] for t in cid_tests}
            if "FAILED_REPRODUCTION" in statuses:
                verdict = "FAILED_REPRODUCTION"
                note = "One or more metrics departed significantly beyond predefined equivalence margins."
            elif "INCONCLUSIVE" in statuses:
                verdict = "INCONCLUSIVE"
                note = "Insufficient sample size or degenerate variance for equivalence inference."
            elif statuses == {"STATISTICAL_EQUIVALENCE"}:
                verdict = "STATISTICAL_EQUIVALENCE"
                note = f"All {len(cid_tests)} pre-registered metrics passed Holm-adjusted TOST equivalence."
            else:
                verdict = "SIMILAR_INSUFFICIENT_EQUIVALENCE"
                note = (
                    "Metric means appear similar without significant divergence, but at least one "
                    "metric failed to establish Holm-adjusted TOST equivalence."
                )
            stoch_results[cid] = {
                "config_id": cid,
                "classification": "stochastic",
                "verdict": verdict,
                "metrics_evaluated": len(cid_tests),
                "note": note,
            }

    # Export reproduced artifacts strictly to target_dir
    with open(target_dir / "reproduced_stories.jsonl", "w", encoding="utf-8") as f:
        for s in all_stories:
            f.write(json.dumps(s.to_dict()) + "\n")

    with open(target_dir / "reproduced_failures.jsonl", "w", encoding="utf-8") as f:
        for fl in all_failures:
            f.write(json.dumps(fl.to_dict()) + "\n")

    export_per_story_metrics_csv(all_stories, target_dir / "reproduced_per_story_metrics.csv")
    agg = aggregate_config_metrics(all_stories)
    export_aggregate_metrics_csv(agg, target_dir / "reproduced_aggregate_metrics.csv")

    all_verdicts = [v["verdict"] for v in det_results.values()] + [v["verdict"] for v in stoch_results.values()]
    if "FAILED_REPRODUCTION" in all_verdicts:
        overall = "FAILED_REPRODUCTION"
    elif skipped_ids or "INCONCLUSIVE" in all_verdicts:
        overall = "PARTIAL_OR_INCONCLUSIVE"
    elif "SIMILAR_INSUFFICIENT_EQUIVALENCE" in all_verdicts:
        overall = "SIMILAR_INSUFFICIENT_EQUIVALENCE"
    else:
        overall = "REPRODUCED_EXACT_AND_EQUIVALENT"

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "output_dir": str(target_dir),
        "overall_verdict": overall,
        "base_seed": base_seed,
        "stochastic_seed": stochastic_seed,
        "stories_per_config": stories_per_config,
        "total_stories_generated": len(all_stories),
        "total_failures": len(all_failures),
        "wall_time_sec": round(wall_time, 4),
        "throughput_stories_per_sec": round(len(all_stories) / max(1e-6, wall_time), 2),
        "evaluated_configs": evaluated_ids,
        "skipped_configs": skipped_ids,
        "deterministic_configs": [c for c in evaluated_ids if c in DETERMINISTIC_CONFIGS],
        "stochastic_configs": [c for c in evaluated_ids if c in STOCHASTIC_CONFIGS],
        "canonical_audit": {
            "e6_verified_k": audit["e6_verified_k"],
            "canonical_config_semantic_hash": audit["canonical_config_semantic_hash"],
            "registry_config_semantic_hash": audit["registry_config_semantic_hash"],
            "canonical_hashes": audit["canonical_hashes"],
        },
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
        },
        "equivalence_methodology": {
            "test": "Welch Two One-Sided Tests (TOST)",
            "multiple_comparisons": "Holm-Bonferroni step-down",
            "family_alpha": FAMILY_ALPHA,
            "effect_size": "Hedges' g",
            "min_sample_size": MIN_STATISTICAL_SAMPLE_SIZE,
            "margins": [asdict(m) for m in PREDEFINED_EQUIVALENCE_MARGINS],
        },
        "tost_comparison_count": sum(1 for t in tost_tests if "p_tost" in t and t["status"] != "INCONCLUSIVE"),
        "deterministic_results": det_results,
        "stochastic_results": stoch_results,
        "tost_tests": tost_tests,
        "warnings": warnings,
    }

    with open(target_dir / "reproducibility_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_md = render_markdown_report(summary)
    with open(target_dir / "reproducibility_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    return summary


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for reproduce_all."""
    parser = argparse.ArgumentParser(
        prog="python -m five_head_nlp.experiments.reproduce_all",
        description="Safely reproduce and verify Phase 5 Monte Carlo experiments in an isolated output directory.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=DEFAULT_BASE_SEED,
        help=f"Base seed for deterministic conditions (default: {DEFAULT_BASE_SEED}).",
    )
    parser.add_argument(
        "--stochastic-seed",
        type=int,
        default=DEFAULT_STOCHASTIC_SEED,
        help=f"Independent base seed for stochastic equivalence testing (default: {DEFAULT_STOCHASTIC_SEED}).",
    )
    parser.add_argument(
        "--stories-per-config",
        type=int,
        default=DEFAULT_STORIES_PER_CONFIG,
        help=f"Stories generated per condition (default: {DEFAULT_STORIES_PER_CONFIG}).",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Optional subset of configs to run (e.g. E1 E6 E7).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Deterministic run identifier under reports/reproducibility/<run_id>/.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Optional parent directory overriding reports/reproducibility.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit canonical configuration and path safety without running generation.",
    )
    return parser


def main(args: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for Phase 5 reproducibility verification."""
    parser = build_parser()
    parsed = parser.parse_args(args)
    out_root = Path(parsed.output_root) if parsed.output_root else None

    result = run_reproducibility(
        base_seed=parsed.base_seed,
        stochastic_seed=parsed.stochastic_seed,
        stories_per_config=parsed.stories_per_config,
        configs=parsed.configs,
        run_id=parsed.run_id,
        output_root=out_root,
        dry_run=parsed.dry_run,
    )

    if parsed.dry_run:
        print(json.dumps(result, indent=2))
        return 0

    print(f"Reproducibility run complete: {result['run_id']}")
    print(f"Output directory : {result['output_dir']}")
    print(f"Overall verdict  : {result['overall_verdict']}")
    print(f"Total stories    : {result['total_stories_generated']} (failures: {result['total_failures']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
