"""Unit tests for BaseHead abstract class."""

import pytest

from five_head_nlp.heads.base_head import BaseHead
from five_head_nlp.heads.head_result import HeadScore


class DummyHead(BaseHead):
    """Minimal concrete head implementation for testing."""

    def score_candidates(self, context, candidates, state=None):
        return {
            c.token: HeadScore(
                head_name=self.name,
                token=c.token,
                raw_score=1.0,
                normalized_score=1.0,
                explanation="dummy explanation",
            )
            for c in candidates
        }


class TestBaseHead:
    """Test BaseHead properties and interface invariants."""

    def test_negative_weight_raises_error(self):
        with pytest.raises(ValueError, match="non-negative"):
            DummyHead(name="Invalid", weight=-0.5)

    def test_head_properties(self):
        head = DummyHead(name="TestHead", weight=0.3, enabled=True)
        assert head.name == "TestHead"
        assert head.weight == 0.3
        assert head.enabled is True

    def test_default_state_methods_do_not_raise(self):
        head = DummyHead(name="TestHead")
        head.update_state("token")
        head.reset_state()
