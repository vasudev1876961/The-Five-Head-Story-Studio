"""Unit tests for the five_head_nlp.decoding backward-compatibility facade."""

from five_head_nlp.decoding import (
    BaseDecoder,
    DecodingStrategy,
    GreedyDecoder,
    TemperatureDecoder,
    TopKDecoder,
)
from five_head_nlp.prediction.decoding import BaseDecoder as OrigBaseDecoder
from five_head_nlp.prediction.greedy import GreedyDecoder as OrigGreedyDecoder
from five_head_nlp.prediction.temperature import (
    TemperatureDecoder as OrigTemperatureDecoder,
)
from five_head_nlp.prediction.top_k import TopKDecoder as OrigTopKDecoder


def test_decoding_facade_re_exports() -> None:
    """Verify that five_head_nlp.decoding exports identical classes to prediction modules."""
    assert BaseDecoder is OrigBaseDecoder
    assert GreedyDecoder is OrigGreedyDecoder
    assert TemperatureDecoder is OrigTemperatureDecoder
    assert TopKDecoder is OrigTopKDecoder


def test_decoding_strategy_enum_members() -> None:
    """Verify that DecodingStrategy enum members are fully accessible via facade."""
    assert hasattr(DecodingStrategy, "GREEDY")
    assert hasattr(DecodingStrategy, "TEMPERATURE")
    assert hasattr(DecodingStrategy, "TOP_K")
    assert DecodingStrategy.GREEDY.value == "greedy"
    assert DecodingStrategy.TOP_K.value == "top_k"
    assert DecodingStrategy.TEMPERATURE.value == "temperature"
