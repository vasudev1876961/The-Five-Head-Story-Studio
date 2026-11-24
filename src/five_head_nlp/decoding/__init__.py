"""Decoding strategies, temperature scaling, repetition suppression, and sampling."""

from enum import Enum

from five_head_nlp.prediction.decoding import BaseDecoder
from five_head_nlp.prediction.greedy import GreedyDecoder
from five_head_nlp.prediction.temperature import TemperatureDecoder
from five_head_nlp.prediction.top_k import TopKDecoder


class DecodingStrategy(str, Enum):
    """Enumeration of supported decoding strategies."""

    GREEDY = "greedy"
    TOP_K = "top_k"
    TEMPERATURE = "temperature"


__all__ = [
    "BaseDecoder",
    "DecodingStrategy",
    "GreedyDecoder",
    "TemperatureDecoder",
    "TopKDecoder",
]
