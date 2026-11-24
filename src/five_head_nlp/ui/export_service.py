"""Export service for serializing stories, metadata, and scoring explanations."""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Dict, List, Optional


class ExportService:
    """Formats generated stories and scoring diagnostics into downloadable file formats."""

    @classmethod
    def export_plain_text(cls, story: Dict[str, Any]) -> str:
        """Format story as clean plain text with generation header."""
        cfg = story.get("configuration", {})
        lines = [
            f"THE FIVE-HEAD STORY STUDIO — GENERATED NARRATIVE",
            f"Story ID      : {story.get('story_id')}",
            f"Timestamp     : {story.get('timestamp')}",
            f"Mode          : {cfg.get('mode')}",
            f"Smoother      : {cfg.get('smoother_name')}",
            f"Decoder       : {cfg.get('decoder_name')} (Seed: {cfg.get('seed')})",
            f"Length        : {story.get('token_count')} tokens, {story.get('sentence_count')} sentences",
            f"Throughput    : {story.get('tokens_per_second')} tokens/sec ({story.get('generation_time_sec')}s total)",
            "=" * 70,
            "",
            story.get("text", ""),
            "",
            "=" * 70,
        ]
        return "\n".join(lines)

    @classmethod
    def export_markdown(
        cls,
        story: Dict[str, Any],
        steps: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Format story and metrics summary as GitHub-flavored Markdown."""
        cfg = story.get("configuration", {})
        metrics = story.get("metrics", {})
        rep = metrics.get("repetition", {})
        div = metrics.get("diversity", {})
        char = metrics.get("character", {})
        act = metrics.get("action", {})

        md = [
            f"# Generated Story: `{story.get('story_id')}`",
            "",
            "## Narrative Text",
            f"> {story.get('text', '')}",
            "",
            "## Generation Metadata",
            f"- **Timestamp**: {story.get('timestamp')}",
            f"- **Mode**: `{cfg.get('mode')}`",
            f"- **Smoother**: `{cfg.get('smoother_name')}`",
            f"- **Decoder**: `{cfg.get('decoder_name')}` (Seed: `{cfg.get('seed')}`)",
            f"- **Token Count**: {story.get('token_count')} tokens",
            f"- **Sentence Count**: {story.get('sentence_count')} sentences",
            f"- **Generation Time**: {story.get('generation_time_sec')}s ({story.get('tokens_per_second')} tokens/sec)",
            f"- **Termination Reason**: {story.get('termination_reason')}",
            "",
            "## Quality Metrics Summary",
            "| Quality Dimension | Metric Name | Value |",
            "| :--- | :--- | :--- |",
            f"| Repetition | Repeated Bigram Rate | {rep.get('repeated_bigram_rate', 0.0)*100:.1f}% |",
            f"| Repetition | Window (w=8) Repeat Rate | {rep.get('window_repetition_rate', 0.0)*100:.1f}% |",
            f"| Diversity | Type-Token Ratio (TTR) | {div.get('ttr', 0.0):.3f} |",
            f"| Character | Continuity Ratio | {char.get('character_continuity_ratio', 0.0):.3f} |",
            f"| Action | Dynamic Action Ratio | {act.get('dynamic_action_ratio', 0.0)*100:.1f}% |",
            "",
        ]

        if steps:
            md.extend([
                "## Token Generation History (First 15 Steps)",
                "| Step | Token | Context | Mode | Latency (ms) |",
                "| :--- | :--- | :--- | :--- | :--- |",
            ])
            for stp in steps[:15]:
                ctx_str = " ".join(stp.get("context", []))
                md.append(f"| {stp.get('step_index')} | `{stp.get('token')}` | `{ctx_str}` | {stp.get('mode')} | {stp.get('latency_ms')} |")
            md.append("")

        return "\n".join(md)

    @classmethod
    def export_scoring_csv(cls, steps: List[Dict[str, Any]]) -> str:
        """Export per-token candidate scoring breakdown to CSV string."""
        out = io.StringIO()
        fieldnames = [
            "step_index",
            "token",
            "context",
            "candidate_rank",
            "candidate_token",
            "candidate_prob",
            "fused_score",
            "grammar_score",
            "character_score",
            "setting_score",
            "action_score",
            "repeat_score",
        ]
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()

        for stp in steps:
            s_idx = stp.get("step_index", 0)
            tok = stp.get("token", "")
            ctx_str = " ".join(stp.get("context", []))

            orch_res = stp.get("orchestrator_result")
            if orch_res and "fused_candidates" in orch_res:
                head_scores = orch_res.get("individual_head_scores", {})
                for fc in orch_res.get("fused_candidates", [])[:5]:
                    c_tok = fc.get("token", "")
                    th_scores = head_scores.get(c_tok, {})
                    writer.writerow({
                        "step_index": s_idx,
                        "token": tok,
                        "context": ctx_str,
                        "candidate_rank": fc.get("rank", 0),
                        "candidate_token": c_tok,
                        "candidate_prob": round(float(fc.get("fused_score", 0.0)), 4),
                        "fused_score": round(float(fc.get("fused_score", 0.0)), 4),
                        "grammar_score": round(float(th_scores.get("Grammar Scout", {}).get("normalized_score", 0.0)), 3),
                        "character_score": round(float(th_scores.get("Character Keeper", {}).get("normalized_score", 0.0)), 3),
                        "setting_score": round(float(th_scores.get("Setting Painter", {}).get("normalized_score", 0.0)), 3),
                        "action_score": round(float(th_scores.get("Action Booster", {}).get("normalized_score", 0.0)), 3),
                        "repeat_score": round(float(th_scores.get("Repeat Blocker", {}).get("normalized_score", 0.0)), 3),
                    })
            else:
                for c in stp.get("candidates", [])[:5]:
                    writer.writerow({
                        "step_index": s_idx,
                        "token": tok,
                        "context": ctx_str,
                        "candidate_rank": c.get("rank", 0),
                        "candidate_token": c.get("token", ""),
                        "candidate_prob": c.get("probability", 0.0),
                        "fused_score": "",
                        "grammar_score": "",
                        "character_score": "",
                        "setting_score": "",
                        "action_score": "",
                        "repeat_score": "",
                    })

        return out.getvalue()

    @classmethod
    def export_configuration_json(cls, config: Dict[str, Any]) -> str:
        """Serialize configuration parameters to formatted JSON string."""
        return json.dumps(config, indent=2)

    @classmethod
    def export_story_json(cls, story: Dict[str, Any]) -> str:
        """Serialize entire story object to formatted JSON string."""
        return json.dumps(story, indent=2)
