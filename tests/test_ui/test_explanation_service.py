"""Unit tests for ExplanationService and Hamilton breakdown."""

import pytest

from five_head_nlp.ui.explanation_service import ExplanationService


class TestExplanationService:
    """Test candidate extraction, explanation retrieval, and Hamilton d10 apportionment."""

    def test_hamilton_d10_breakdown_properties(self):
        weights = {
            "Grammar Scout": 0.40,
            "Character Keeper": 0.15,
            "Setting Painter": 0.15,
            "Action Booster": 0.10,
            "Repeat Blocker": 0.20,
        }
        res = ExplanationService.compute_hamilton_breakdown(weights, total_points=10)

        assert res["total_budget"] == 10
        alloc = res["final_allocation"]
        assert sum(alloc.values()) == 10
        assert alloc["Grammar Scout"] == 4
        assert alloc["Character Keeper"] in {1, 2}
        assert alloc["Repeat Blocker"] == 2
        assert len(res["table"]) == 5

    def test_candidate_breakdown_table_five_head(self):
        step = {
            "token": "castle",
            "mode": "five_head",
            "orchestrator_result": {
                "fused_candidates": [
                    {"token": "castle", "fused_score": 0.85, "rank": 1},
                    {"token": "cave", "fused_score": 0.65, "rank": 2},
                ],
                "individual_head_scores": {
                    "castle": {
                        "Grammar Scout": {"normalized_score": 0.90},
                        "Character Keeper": {"normalized_score": 0.20},
                        "Setting Painter": {"normalized_score": 0.85},
                        "Action Booster": {"normalized_score": 0.20},
                        "Repeat Blocker": {"normalized_score": 0.95},
                    }
                },
            },
        }
        table = ExplanationService.get_candidate_breakdown_table(step)
        assert len(table) == 2
        assert table[0]["Token"] == "castle"
        assert table[0]["Fused Score"] == 0.85
        assert table[0]["Setting Painter"] == 0.85

    def test_token_explanations(self):
        step = {
            "token": "milo",
            "orchestrator_result": {
                "individual_head_scores": {
                    "milo": {
                        "Character Keeper": {
                            "normalized_score": 0.85,
                            "explanation": "Maintains active character focus.",
                        }
                    }
                }
            },
        }
        expl = ExplanationService.get_token_explanations(step, "milo")
        assert "Character Keeper" in expl
        assert expl["Character Keeper"]["explanation"] == "Maintains active character focus."
