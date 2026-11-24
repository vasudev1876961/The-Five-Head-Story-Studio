"""Language models package for the Five-Head Story-Generation System."""

from five_head_nlp.models.base_ngram import BaseNGramModel
from five_head_nlp.models.bigram import BigramModel
from five_head_nlp.models.trigram import TrigramModel
from five_head_nlp.models.unigram import UnigramModel

__all__ = [
    "BaseNGramModel",
    "UnigramModel",
    "BigramModel",
    "TrigramModel",
]
