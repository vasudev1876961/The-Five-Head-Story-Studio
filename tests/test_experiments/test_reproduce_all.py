"""Tests for the Phase 5 Safe Experiment Reproducibility Harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from five_head_nlp.experiments.experiment_registry import get_experiment_config
from five_head_nlp.experiments.reproduce_all import (
    DEFAULT_BASE_SEED,
    DETERMINISTIC_CONFIGS,
    PREDEFINED_EQUIVALENCE_MARGINS,
    EquivalenceMarginSpec,
    apply_holm_bonferroni,
    audit_canonical_artifacts,
    canonical_config_hash,
    evaluate_deterministic_condition,
    evaluate_tost_metric,
    find_repo_root,
    main,
    run_reproducibility,
    sha256_file,
    student_t_cdf,
    student_t_ppf,
    validate_safe_output_dir,
)
from five_head_nlp.experiments.story_generator import GeneratedStory


REPO_ROOT = find_repo_root()


class TestCanonicalAuditAndSafety:
    """Verify canonical configuration loading, E6 truth, and overwrite protections."""

    def test_canonical_audit_and_e6_k5(self) -> None:
        audit = audit_canonical_artifacts(REPO_ROOT)
        assert audit["e6_verified_k"] == 5
        assert audit["canonical_configs"]["E6"]["decoder_type"] == "top_k"
        assert audit["canonical_configs"]["E6"]["decoder_params"] == {"k": 5}
        assert audit["canonical_config_semantic_hash"] == audit["registry_config_semantic_hash"]
        assert get_experiment_config("E6").decoder_params == {"k": 5}
        assert len(audit["canonical_hashes"]) == 6

    def test_missing_canonical_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            audit_canonical_artifacts(tmp_path)

    @pytest.mark.parametrize(
        "unsafe_rel",
        [
            ".",
            "experiments/results",
            "experiments/results/monte_carlo",
            "experiments/results/monte_carlo/nested",
            "experiments/models",
            "data",
            "data/raw",
            "phase_7",
            "src",
            "tests",
        ],
    )
    def test_protected_paths_rejected(self, unsafe_rel: str) -> None:
        target = REPO_ROOT if unsafe_rel == "." else REPO_ROOT / unsafe_rel
        with pytest.raises(ValueError, match="Refusing|Unsafe output directory"):
            validate_safe_output_dir(target, REPO_ROOT)

    def test_traversal_run_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsafe output directory"):
            run_reproducibility(
                run_id="../../experiments/results/monte_carlo",
                dry_run=True,
                repo_root=REPO_ROOT,
            )

    def test_isolated_output_dir_allowed(self, tmp_path: Path) -> None:
        resolved = validate_safe_output_dir(tmp_path / "repro_run", REPO_ROOT)
        assert resolved == (tmp_path / "repro_run").resolve()


class TestStudentTAndStatisticalEquivalence:
    """Verify exact Student-t computations and TOST + Holm-Bonferroni verdicts."""

    def test_student_t_cdf_and_ppf_accuracy(self) -> None:
        # Textbook two-sided 95% critical values
        assert abs(student_t_ppf(0.975, df=10.0) - 2.22813885) < 1e-6
        assert abs(student_t_ppf(0.975, df=1000.0) - 1.962339) < 1e-5
        assert abs(student_t_cdf(0.0, df=5.0) - 0.5) < 1e-12
        assert abs(student_t_cdf(2.22813885, df=10.0) - 0.975) < 1e-6

    def test_tost_statistical_equivalence_passes_for_tight_distributions(self) -> None:
        spec = EquivalenceMarginSpec(metric="dummy_metric", margin=0.05, rationale="test")
        # Two tight centered samples with difference ~0.002 << margin 0.05
        canon = [0.50 + ((i % 5) - 2) * 0.005 for i in range(200)]
        repro = [0.502 + (((i + 1) % 5) - 2) * 0.005 for i in range(200)]

        res = evaluate_tost_metric("E6", spec, canon, repro)
        apply_holm_bonferroni([res], alpha=0.05)

        assert res["status"] == "STATISTICAL_EQUIVALENCE"
        assert res["p_tost_holm"] < 0.05
        assert -0.05 < res["ci_family_bonferroni"][0]
        assert res["ci_family_bonferroni"][1] < 0.05

    def test_tost_detects_similar_insufficient_equivalence_under_high_noise(self) -> None:
        spec = EquivalenceMarginSpec(metric="noisy_metric", margin=0.05, rationale="test")
        # Difference is +0.02 (< 0.05 margin) AND p_diff non-significant, but SD=0.25 -> CI crosses 0.05
        canon = [0.50 + (0.25 if i % 2 == 0 else -0.25) for i in range(40)]
        repro = [0.52 + (0.25 if i % 2 == 0 else -0.25) for i in range(40)]

        res = evaluate_tost_metric("E6", spec, canon, repro)
        apply_holm_bonferroni([res], alpha=0.05)

        assert res["status"] == "SIMILAR_INSUFFICIENT_EQUIVALENCE"
        assert res["p_diff"] > 0.05
        assert res["p_tost_holm"] >= 0.05

    def test_tost_detects_failed_reproduction_beyond_margin(self) -> None:
        spec = EquivalenceMarginSpec(metric="shifted_metric", margin=0.05, rationale="test")
        canon = [0.50 + (0.01 if i % 2 == 0 else -0.01) for i in range(100)]
        repro = [0.70 + (0.01 if i % 2 == 0 else -0.01) for i in range(100)]

        res = evaluate_tost_metric("E8", spec, canon, repro)
        apply_holm_bonferroni([res], alpha=0.05)

        assert res["status"] == "FAILED_REPRODUCTION"

    def test_tost_small_sample_inconclusive(self) -> None:
        spec = EquivalenceMarginSpec(metric="small_sample", margin=0.05, rationale="test")
        canon = [0.50 + i * 0.01 for i in range(50)]
        repro = [0.50 + i * 0.01 for i in range(5)]

        res = evaluate_tost_metric("E6", spec, canon, repro)
        apply_holm_bonferroni([res], alpha=0.05)

        assert res["status"] == "INCONCLUSIVE"


class TestDeterministicComparisonAndHarness:
    """Verify exact deterministic comparison, seed anchoring, partial runs, and immutability."""

    def _make_story(self, tokens: list[str], seed: int = 42, gen_time: float = 0.001) -> GeneratedStory:
        return GeneratedStory(
            story_id="E1_0001",
            config_id="E1",
            config_name="Baseline MLE",
            seed=seed,
            tokens=tokens,
            text="The cat.",
            sentence_count=1,
            token_count=len(tokens),
            generation_time_sec=gen_time,
            token_latencies_ms=[0.1],
            metrics={
                "length": {"total_tokens": len(tokens)},
                "latency": {"mean_token_latency_ms": 0.1 if gen_time == 0.001 else 99.0},
            },
        )

    def test_deterministic_exact_match_ignores_hardware_latency(self) -> None:
        canon = {
            "E1_0001": {
                "seed": 42,
                "tokens": ["the", "cat", "."],
                "text": "The cat.",
                "sentence_count": 1,
                "token_count": 3,
                "metrics": {
                    "length": {"total_tokens": 3},
                    "latency": {"mean_token_latency_ms": 0.01},
                },
            }
        }
        story = self._make_story(["the", "cat", "."], seed=42, gen_time=1.234)
        res = evaluate_deterministic_condition("E1", [story], 0, canon, DEFAULT_BASE_SEED)
        assert res["verdict"] == "EXACT_REPRODUCTION"
        assert res["exact_token_matches"] == 1
        assert res["exact_metric_matches"] == 1

    def test_deterministic_mismatch_fails(self) -> None:
        canon = {
            "E1_0001": {
                "seed": 42,
                "tokens": ["the", "cat", "."],
                "text": "The cat.",
                "sentence_count": 1,
                "token_count": 3,
                "metrics": {"length": {"total_tokens": 3}},
            }
        }
        story = self._make_story(["the", "dog", "."], seed=42)
        res = evaluate_deterministic_condition("E1", [story], 0, canon, DEFAULT_BASE_SEED)
        assert res["verdict"] == "FAILED_REPRODUCTION"
        assert res["exact_token_matches"] == 0

    def test_partial_run_immutability_and_seed_anchoring(self, tmp_path: Path) -> None:
        model_files = [
            REPO_ROOT / "experiments" / "models" / "unigrammodel_corpus.json",
            REPO_ROOT / "experiments" / "models" / "bigrammodel_corpus.json",
            REPO_ROOT / "experiments" / "models" / "trigrammodel_corpus.json",
            REPO_ROOT / "experiments" / "results" / "monte_carlo" / "generated_stories.jsonl",
        ]
        before = {p.name: sha256_file(p) for p in model_files}

        # Run partial subset E7 (canonical index 6 -> seed 60042!) + E6 (stochastic) with n=2
        summary = run_reproducibility(
            base_seed=42,
            stochastic_seed=42,  # Deliberate equality to test replay warning
            stories_per_config=2,
            configs=["E7", "E6"],
            output_root=tmp_path,
            repo_root=REPO_ROOT,
        )

        after = {p.name: sha256_file(p) for p in model_files}
        assert before == after

        out_dir = Path(summary["output_dir"])
        assert (out_dir / "reproducibility_summary.json").is_file()
        assert (out_dir / "reproducibility_report.md").is_file()
        assert (out_dir / "reproduced_stories.jsonl").is_file()

        assert summary["overall_verdict"] == "PARTIAL_OR_INCONCLUSIVE"
        assert summary["deterministic_results"]["E7"]["verdict"] == "EXACT_REPRODUCTION"
        assert summary["stochastic_results"]["E6"]["verdict"] == "INCONCLUSIVE"
        assert len(summary["skipped_configs"]) == 6
        assert any("deterministic PRNG replay" in w for w in summary["warnings"])

    def test_cli_dry_run_and_help(self, capsys) -> None:
        rc = main(["--dry-run", "--configs", "E1", "E6"])
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["status"] == "DRY_RUN_VALID"
        assert out["e6_verified_k"] == 5

        with pytest.raises(ValueError, match="stories_per_config"):
            run_reproducibility(stories_per_config=0, dry_run=True, repo_root=REPO_ROOT)

    def test_canonical_hash_invariant_to_lexicon_list_order(self) -> None:
        cfg_a = {"E4": {"scoring_config": {"character_lexicon": ["milo", "pip"]}}}
        cfg_b = {"E4": {"scoring_config": {"character_lexicon": ["pip", "milo"]}}}
        assert canonical_config_hash(cfg_a) == canonical_config_hash(cfg_b)
        assert "E7" in DETERMINISTIC_CONFIGS
        assert len(PREDEFINED_EQUIVALENCE_MARGINS) == 6
