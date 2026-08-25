# test_elo.py

import os
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import pytest

from core.elo import (
    DEFAULT_RATING, DEFAULT_K, expected_score, update_ratings, pfsp_weights,
)


def test_equal_ratings_expect_a_coin_flip():
    assert expected_score(1200.0, 1200.0) == pytest.approx(0.5)


def test_a_four_hundred_point_lead_is_ten_to_one():
    assert expected_score(1600.0, 1200.0) == pytest.approx(10 / 11, abs=1e-6)


def test_elo_is_zero_sum():
    a, b = update_ratings(1200.0, 1200.0, score_a=1.0)
    assert (a - 1200.0) == pytest.approx(-(b - 1200.0))


def test_beating_an_equal_opponent_moves_by_half_k():
    a, b = update_ratings(1200.0, 1200.0, score_a=1.0)
    assert a == pytest.approx(1200.0 + DEFAULT_K * 0.5)
    assert b == pytest.approx(1200.0 - DEFAULT_K * 0.5)


def test_a_draw_between_equals_changes_nothing():
    a, b = update_ratings(1200.0, 1200.0, score_a=0.5)
    assert a == pytest.approx(1200.0)
    assert b == pytest.approx(1200.0)


def test_beating_a_much_weaker_opponent_barely_moves_the_rating():
    a, _ = update_ratings(1600.0, 1200.0, score_a=1.0)
    assert 0.0 < (a - 1600.0) < 4.0


def test_pfsp_weights_sum_to_one_and_favour_hard_opponents():
    w = pfsp_weights([0.9, 0.5, 0.1])
    assert sum(w) == pytest.approx(1.0)
    assert w[2] > w[1] > w[0]   # lowest win rate gets the most probability


def test_pfsp_weights_are_uniform_when_every_opponent_is_equal():
    w = pfsp_weights([0.5, 0.5, 0.5, 0.5])
    for x in w:
        assert x == pytest.approx(0.25)


def test_pfsp_handles_a_fully_mastered_field_without_dividing_by_zero():
    w = pfsp_weights([1.0, 1.0, 1.0])
    assert sum(w) == pytest.approx(1.0)
    for x in w:
        assert x == pytest.approx(1 / 3)


def test_pfsp_rejects_an_empty_field():
    with pytest.raises(ValueError):
        pfsp_weights([])
