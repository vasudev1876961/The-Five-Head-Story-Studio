"""Unit tests for Absolute Discounting Backoff probability smoothing."""

import pytest

from five_head_nlp.models.trigram import TrigramModel
from five_head_nlp.smoothing.backoff import BackoffSmoother
from five_head_nlp.tokenization.corpus_loader import CorpusLoader


@pytest.fixture
def synthetic_corpus():
    # Sentence 1: a b c .
    # Sentence 2: a b d .
    # Sentence 3: a e .
    return [
        ["a", "b", "c", "."],
        ["a", "b", "d", "."],
        ["a", "e", "."],
    ]


@pytest.fixture
def trained_backoff(synthetic_corpus):
    model = TrigramModel()
    model.fit(synthetic_corpus)
    return BackoffSmoother(discount=0.5, model=model)


class TestBackoffSmootherSynthetic:
    """Mathematical verification of Backoff smoother."""

    def test_invalid_discount_raises_value_error(self, synthetic_corpus):
        model = TrigramModel()
        model.fit(synthetic_corpus)
        with pytest.raises(ValueError, match="must be in range"):
            BackoffSmoother(discount=0.0, model=model)
        with pytest.raises(ValueError, match="must be in range"):
            BackoffSmoother(discount=1.0, model=model)

    def test_observable_fallback_levels(self, trained_backoff):
        # ('a', 'b', 'c') occurred in training -> 'trigram'
        assert trained_backoff.get_fallback_level("c", context=("a", "b")) == "trigram"

        # ('a', 'b', 'e') did not occur, but ('b', 'e') did not occur either -> 'unigram'
        # But for ('a', 'e'): context ('a',) followed by 'e' occurred in bigram -> 'bigram'
        assert trained_backoff.get_fallback_level("e", context=("a",)) == "bigram"

        # Unknown context falls back to unigram
        assert trained_backoff.get_fallback_level("a", context=("x", "y")) == "unigram"

    def test_probability_simplex_normalization(self, trained_backoff):
        # Context ('a', 'b'): observed continuations 'c' and 'd'
        dist_ab = trained_backoff.get_distribution(context=("a", "b"))
        assert len(dist_ab) == 6
        assert pytest.approx(sum(dist_ab.values()), rel=1e-5) == 1.0

        # Unseen context ('x', 'y')
        dist_xy = trained_backoff.get_distribution(context=("x", "y"))
        assert len(dist_xy) == 6
        assert pytest.approx(sum(dist_xy.values()), rel=1e-5) == 1.0

    def test_degenerate_lower_order_zero_mass_fallback(self, trained_backoff, monkeypatch):
        """Audit requirement: verify degenerate contexts where lower-order mass over unseen outcomes is zero.

        Ensures that if unseen_prob_sum == 0, reserved mass beta(h) is distributed uniformly
        across unseen tokens so simplex normalization sum P(v | h) = 1.0 is strictly conserved.
        """
        # Simulate lower order assigning 0 mass to unseen tokens
        def patched_p_unigram(tok):
            if tok in ("b", "e"):
                return 0.5
            return 0.0

        monkeypatch.setattr(trained_backoff, "_p_unigram", patched_p_unigram)

        # Context 'a' followed by unseen continuation 'c'
        # Seen continuations: 'b', 'e'
        # c_ctx = 3. unseen_tokens = ['a', 'c', 'd', '.'] (len = 4)
        # unseen_prob_sum will be sum(patched_p_unigram(t) for t in unseen_tokens) = 0.0
        # beta = (0.5 * 2) / 3 = 1.0 / 3
        # Degenerate fallback distributes beta uniformly: beta / len(unseen_tokens) = (1/3) / 4 = 1/12
        p_c = trained_backoff.probability("c", context="a")
        assert pytest.approx(p_c, rel=1e-5) == (1.0 / 3.0) / 4.0

        # Simplex sum over all 6 tokens should equal 1.0
        dist = trained_backoff.get_distribution(context="a")
        assert pytest.approx(sum(dist.values()), rel=1e-5) == 1.0


class TestBackoffSmootherCorpusIntegration:
    """Integration test with the 34-sentence story corpus."""

    @pytest.fixture(scope="class")
    def corpus_backoff(self):
        loader = CorpusLoader()
        model = TrigramModel()
        model.fit_corpus(loader)
        return BackoffSmoother(discount=0.5, model=model)

    def test_corpus_fallback_levels(self, corpus_backoff):
        # ("the", "crystal", "castle") was observed -> trigram
        assert corpus_backoff.get_fallback_level("castle", context=("the", "crystal")) == "trigram"

        # ("near", "crystal", "castle") - trigram unseen, but ("crystal", "castle") seen -> bigram
        assert corpus_backoff.get_fallback_level("castle", context=("near", "crystal")) == "bigram"

        # ("unseen1", "unseen2", "milo") -> unigram
        assert corpus_backoff.get_fallback_level("milo", context=("unseen1", "unseen2")) == "unigram"

    def test_simplex_normalization_real_corpus(self, corpus_backoff):
        for ctx in [("crystal",), ("the", "dark"), ("near", "the"), ("unseen1", "unseen2")]:
            dist = corpus_backoff.get_distribution(context=ctx)
            assert len(dist) == 120
            assert pytest.approx(sum(dist.values()), rel=1e-5) == 1.0
