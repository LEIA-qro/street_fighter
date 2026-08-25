# elo.py
#
# Elo ratings and Prioritized Fictitious Self-Play weighting.
#
# Rolling win rate against a fixed CPU opponent cannot distinguish "the agent
# got better" from "the curriculum happened to sample easier savestates".
# Elo gives a single transitive scalar across a heterogeneous opponent pool,
# which is what FightLadder (arXiv:2406.02081) reports and what AlphaStar
# (Vinyals et al., Nature 575:350-354, 2019) uses for league bookkeeping.
#
# PFSP weighting f(x) = (1 - x)^p is from that same AlphaStar work: sample
# opponents you are currently losing to, in proportion to how badly.

from typing import List, Tuple

DEFAULT_RATING = 1200.0
DEFAULT_K = 32.0


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability that A beats B under the logistic Elo model."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_ratings(rating_a: float, rating_b: float, score_a: float,
                   k: float = DEFAULT_K) -> Tuple[float, float]:
    """Zero-sum Elo update. score_a is 1.0 win / 0.5 draw / 0.0 loss."""
    if not 0.0 <= score_a <= 1.0:
        raise ValueError(f"score_a must be in [0, 1], got {score_a}")
    exp_a = expected_score(rating_a, rating_b)
    delta = k * (score_a - exp_a)
    return rating_a + delta, rating_b - delta


def pfsp_weights(win_rates: List[float], p: float = 2.0) -> List[float]:
    """Prioritized Fictitious Self-Play sampling weights.

    Opponents the agent loses to most get the most probability. When every
    opponent is equally (un)beaten -- including the fully-mastered case where
    all the raw weights are zero -- the distribution falls back to uniform.
    """
    if not win_rates:
        raise ValueError("pfsp_weights requires at least one opponent")
    raw = [max(0.0, 1.0 - float(w)) ** p for w in win_rates]
    total = sum(raw)
    if total <= 0.0:
        return [1.0 / len(win_rates)] * len(win_rates)
    return [r / total for r in raw]
