# test_metrics_callback.py
#
# The metrics callback is the branch's measurement layer: it must aggregate
# info dicts exactly as the envs emit them. These tests drive the pure
# _ingest_infos/_compute_records pair directly -- no SB3 model, no logger.

import os
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import pytest

from agents.metrics_callback import MetricsCallback
from envs.action_macros import MACRO_NAMES, N_PRIMITIVES


def make_cb():
    return MetricsCallback()


def step_info(**overrides):
    info = {
        "my_hp": 176.0,
        "enemy_hp": 176.0,
        "hp_sentinel": False,
        "reward_parts": {"damage": 0.0, "taken": 0.0, "combo": 0.0,
                         "shaping": 0.1, "time": -0.002, "terminal": 0.0},
    }
    info.update(overrides)
    return info


def terminal_info(win=1, **overrides):
    info = step_info(
        win=win,
        loss=0 if win else 1,
        double_ko=False,
        timeout=False,
        episode_steps=570,
        ep_rel_dist_mean=75.0,
        ep_rel_dist_median=72.0,
        ep_rel_dist_frac_far=0.30,
    )
    info.update(overrides)
    return info


def test_reward_components_are_per_step_means():
    cb = make_cb()
    cb._ingest_infos([step_info(reward_parts={"damage": 10.0, "shaping": 1.0})])
    cb._ingest_infos([step_info(reward_parts={"damage": 0.0, "shaping": 3.0})])
    records = cb._compute_records()
    assert records["reward/damage_per_step"] == pytest.approx(5.0)
    assert records["reward/shaping_per_step"] == pytest.approx(2.0)


def test_sentinel_steps_have_empty_parts_and_count_toward_sentinel_frac():
    cb = make_cb()
    cb._ingest_infos([step_info(hp_sentinel=True, reward_parts={})])
    cb._ingest_infos([step_info()])
    records = cb._compute_records()
    assert records["env/hp_sentinel_frac"] == pytest.approx(0.5)
    # The sentinel step's empty dict must not dilute the per-step means.
    assert records["reward/shaping_per_step"] == pytest.approx(0.1)


def test_spacing_aggregates_average_over_episodes():
    cb = make_cb()
    cb._ingest_infos([terminal_info(ep_rel_dist_frac_far=0.5)])
    cb._ingest_infos([terminal_info(ep_rel_dist_frac_far=0.1)])
    records = cb._compute_records()
    assert records["spacing/frac_steps_far"] == pytest.approx(0.3)
    assert records["spacing/ep_rel_dist_mean"] == pytest.approx(75.0)


def test_macro_fraction_and_per_macro_usage():
    cb = make_cb()
    # 3 primitive actions, 1 hadouken (first macro id).
    for action in (0, 5, 62, N_PRIMITIVES):
        cb._ingest_infos([step_info(macro_action=action)])
    records = cb._compute_records()
    assert records["macros/frac_macro_actions"] == pytest.approx(0.25)
    assert records[f"macros/use_{MACRO_NAMES[0]}"] == pytest.approx(0.25)
    assert records[f"macros/use_{MACRO_NAMES[1]}"] == pytest.approx(0.0)


def test_no_macro_keys_without_macro_wrapper():
    cb = make_cb()
    cb._ingest_infos([step_info()])
    assert not any(k.startswith("macros/") for k in cb._compute_records())


def test_episode_outcome_rates():
    cb = make_cb()
    cb._ingest_infos([terminal_info(win=1)])
    cb._ingest_infos([terminal_info(win=0, timeout=True)])
    records = cb._compute_records()
    assert records["episodes/win_rate"] == pytest.approx(0.5)
    assert records["episodes/timeout_rate"] == pytest.approx(0.5)
    assert records["episodes/len_mean"] == pytest.approx(570.0)


def test_window_reset_clears_accumulators():
    cb = make_cb()
    cb._ingest_infos([terminal_info()])
    cb._reset_window()
    assert cb._compute_records() == {}


def test_socket_death_counted():
    cb = make_cb()
    cb._ingest_infos([{"socket_death": True, "hp_sentinel": False, "reward_parts": {}}])
    assert cb._compute_records()["env/socket_deaths"] == 1
