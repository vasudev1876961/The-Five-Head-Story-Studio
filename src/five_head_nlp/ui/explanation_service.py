"""Explanation and diagnostic inspection service for candidate token scoring."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from five_head_nlp.heads.hamilton_apportionment import HamiltonApportionment


class ExplanationService:
    """Provides human-interpretable diagnostics for candidate rankings and Hamilton d10 allocation."""

    HEAD_NAMES: Tuple[str, ...] = (
        "Grammar Scout",
        "Character Keeper",
        "Setting Painter",
        "Action Booster",
        "Repeat Blocker",
    )

    @classmethod
    def get_candidate_breakdown_table(
        cls,
        step: Dict[str, Any],
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """Extract tabular candidate comparison data for the selected step.

        Parameters
        ----------
        step : Dict[str, Any]
            Token prediction step record.
        top_n : int, default=10
            Number of top candidates to include.

        Returns
        -------
        List[Dict[str, Any]]
            Rows containing token, rank, prob, fused score, and individual head scores.
        """
        rows: List[Dict[str, Any]] = []
        orch_res = step.get("orchestrator_result")

        if orch_res and "fused_candidates" in orch_res:
            # Five-Head mode
            fused_cands = orch_res.get("fused_candidates", [])[:top_n]
            head_scores = orch_res.get("individual_head_scores", {})

            for fc in fused_cands:
                tok = fc.get("token", "")
                t_scores = head_scores.get(tok, {})

                row = {
                    "Rank": fc.get("rank", 0),
                    "Token": tok,
                    "Fused Score": round(float(fc.get("fused_score", 0.0)), 4),
                    "Grammar Scout": round(float(t_scores.get("Grammar Scout", {}).get("normalized_score", 0.0)), 3),
                    "Character Keeper": round(float(t_scores.get("Character Keeper", {}).get("normalized_score", 0.0)), 3),
                    "Setting Painter": round(float(t_scores.get("Setting Painter", {}).get("normalized_score", 0.0)), 3),
                    "Action Booster": round(float(t_scores.get("Action Booster", {}).get("normalized_score", 0.0)), 3),
                    "Repeat Blocker": round(float(t_scores.get("Repeat Blocker", {}).get("normalized_score", 0.0)), 3),
                }
                rows.append(row)
        else:
            # Baseline mode
            cands = step.get("candidates", [])[:top_n]
            for c in cands:
                rows.append({
                    "Rank": c.get("rank", 0),
                    "Token": c.get("token", ""),
                    "Probability": round(float(c.get("probability", 0.0)), 4),
                    "Log Probability": round(float(c.get("log_probability", 0.0)), 2),
                })

        return rows

    @classmethod
    def get_token_explanations(
        cls,
        step: Dict[str, Any],
        token: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Retrieve human-readable explanations and evidence for a candidate token.

        Parameters
        ----------
        step : Dict[str, Any]
            Token prediction step record.
        token : Optional[str]
            Token string to inspect. Defaults to the token selected at this step.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            {head_name: {"raw_score": ..., "normalized_score": ..., "explanation": ..., "evidence": ...}}
        """
        tok = token or step.get("token", "")
        orch_res = step.get("orchestrator_result")
        if not orch_res or "individual_head_scores" not in orch_res:
            return {}

        ind_scores = orch_res.get("individual_head_scores", {})
        return ind_scores.get(tok, {})

    @classmethod
    def compute_hamilton_breakdown(
        cls,
        weights: Dict[str, float],
        total_points: int = 10,
    ) -> Dict[str, Any]:
        """Compute the step-by-step mathematical breakdown of the Hamilton d10 apportionment.

        Parameters
        ----------
        weights : Dict[str, float]
            Dictionary of head weights.
        total_points : int, default=10
            Total integer budget slots.

        Returns
        -------
        Dict[str, Any]
            Detailed breakdown containing quotas, base allocations, remainders,
            surplus assignments, final slots, and pedagogical explanations.
        """
        tot_w = sum(weights.values())
        norm_w = {k: (v / tot_w if tot_w > 0 else 0.20) for k, v in weights.items()}

        quotas: Dict[str, float] = {}
        base_allocations: Dict[str, int] = {}
        remainders: Dict[str, float] = {}

        for k, w in norm_w.items():
            q = total_points * w
            quotas[k] = round(q, 4)
            b = math.floor(q)
            base_allocations[k] = b
            remainders[k] = round(q - b, 4)

        base_sum = sum(base_allocations.values())
        surplus = total_points - base_sum

        # Rank remainders descending
        priority = list(HamiltonApportionment.DEFAULT_PRIORITY)
        def tie_key(name: str) -> Tuple[float, int]:
            rem = remainders[name]
            prio = priority.index(name) if name in priority else 999
            return (-rem, prio)

        ranked = sorted(remainders.keys(), key=tie_key)
        surplus_awarded = {k: 0 for k in weights}
        for i in range(surplus):
            head_name = ranked[i % len(ranked)]
            surplus_awarded[head_name] += 1

        final_allocation = {
            k: base_allocations[k] + surplus_awarded[k] for k in weights
        }

        # Tabular breakdown
        breakdown_rows: List[Dict[str, Any]] = []
        for k in priority:
            if k in weights:
                breakdown_rows.append({
                    "Scoring Head": k,
                    "Normalized Weight": f"{norm_w[k]*100:.1f}%",
                    "Exact Quota (q)": quotas[k],
                    "Base Integer (⌊q⌋)": base_allocations[k],
                    "Remainder (r)": remainders[k],
                    "Surplus Point": "+1" if surplus_awarded[k] > 0 else "0",
                    "Final d10 Budget": final_allocation[k],
                })

        return {
            "total_budget": total_points,
            "base_sum": base_sum,
            "surplus_points": surplus,
            "final_allocation": final_allocation,
            "table": breakdown_rows,
            "pedagogical_note": (
                "Hamilton's largest-remainder method provides an exact, fair integer representation "
                "of continuous head weights for classroom 10-sided die (d10) gameplay. In software mode, "
                "continuous normalized weights are used for linear score fusion, while the d10 allocation "
                "provides an intuitive, glass-box visual interpretation of relative head influence."
            ),
        }
