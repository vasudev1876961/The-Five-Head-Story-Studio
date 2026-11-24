"""Unit tests for Hamilton integer apportionment."""

import pytest

from five_head_nlp.heads.hamilton_apportionment import HamiltonApportionment


class TestHamiltonApportionment:
    """Test Hamilton largest-remainder d10 point allocation."""

    def test_default_balanced_weights_exact_sum_ten(self):
        apportioner = HamiltonApportionment(total_points=10)
        weights = {
            "Grammar Scout": 0.40,
            "Character Keeper": 0.15,
            "Setting Painter": 0.15,
            "Action Booster": 0.10,
            "Repeat Blocker": 0.20,
        }
        alloc = apportioner.apportion(weights)

        assert sum(alloc.values()) == 10
        # Quotas: Grammar=4.0, Char=1.5, Setting=1.5, Action=1.0, Repeat=2.0
        # Base: Grammar=4, Char=1, Setting=1, Action=1, Repeat=2 (Sum=9)
        # Surplus = 1 point to largest remainder.
        # Remainders: Char=0.5, Setting=0.5 (Tied!).
        # Tie-break priority: Character Keeper precedes Setting Painter -> Character gets the +1.
        assert alloc["Grammar Scout"] == 4
        assert alloc["Character Keeper"] == 2
        assert alloc["Setting Painter"] == 1
        assert alloc["Action Booster"] == 1
        assert alloc["Repeat Blocker"] == 2

    def test_zero_weight_heads(self):
        apportioner = HamiltonApportionment(total_points=10)
        weights = {
            "Grammar Scout": 1.0,
            "Character Keeper": 0.0,
            "Setting Painter": 0.0,
        }
        alloc = apportioner.apportion(weights)
        assert sum(alloc.values()) == 10
        assert alloc["Grammar Scout"] == 10
        assert alloc["Character Keeper"] == 0
        assert alloc["Setting Painter"] == 0

    def test_invalid_weights_raise_error(self):
        apportioner = HamiltonApportionment(total_points=10)
        with pytest.raises(ValueError, match="non-negative"):
            apportioner.apportion({"H1": -0.5, "H2": 1.5})
        with pytest.raises(ValueError, match="strictly positive"):
            apportioner.apportion({"H1": 0.0, "H2": 0.0})
