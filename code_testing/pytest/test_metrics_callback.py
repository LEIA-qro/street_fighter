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


def test_ep_air_frac_is_recorded_under_spacing():
    """spacing/ep_air_frac is THE 'no camina nada' metric: uniform random
    over MultiDiscrete([9,7]) sits ~0.33 (3 of 9 directions are jumps);
    walking drives it well under 0.15. The callback must average the
    per-episode fractions the env emits."""
    cb = make_cb()
    cb._ingest_infos([terminal_info(ep_air_frac=0.4)])
    cb._ingest_infos([terminal_info(ep_air_frac=0.2)])
    records = cb._compute_records()
    assert records["spacing/ep_air_frac"] == pytest.approx(0.3)


def test_no_ep_air_frac_key_when_the_env_never_emits_it():
    # e.g. a legacy 13-field Lua client run before ep_air_frac existed.
    cb = make_cb()
    cb._ingest_infos([terminal_info()])
    assert "spacing/ep_air_frac" not in cb._compute_records()


# --------------------------------------------------------------------------
# Outcome rates (Run A post-mortem). Run A's episodes/* tags could not be
# checked for consistency: loss_rate was never logged, so reading the run meant
# reconstructing it as 1 - win - double_ko - timeout and ASSUMING the four
# outcomes partitioned the episodes. They did not -- 23 of 23 losses and 6 of 6
# wins were being logged as DOUBLE_KO. Logging all four makes the partition
# checkable instead of assumed.
# --------------------------------------------------------------------------

def test_loss_and_draw_rates_are_logged():
    cb = make_cb()
    cb._ingest_infos([terminal_info(win=1)])
    cb._ingest_infos([terminal_info(win=0)])                    # loss=1
    cb._ingest_infos([terminal_info(win=0, loss=0, draw=True, double_ko=True)])
    cb._ingest_infos([terminal_info(win=0, loss=0, timeout=True)])
    records = cb._compute_records()
    assert records["episodes/win_rate"] == pytest.approx(0.25)
    assert records["episodes/loss_rate"] == pytest.approx(0.25)
    assert records["episodes/draw_rate"] == pytest.approx(0.25)
    assert records["episodes/timeout_rate"] == pytest.approx(0.25)


def test_outcome_rates_partition_the_episodes():
    """The four rates must sum to exactly 1.0. If they do not, an outcome is
    being misclassified -- which is the failure this whole fix exists for."""
    cb = make_cb()
    cb._ingest_infos([terminal_info(win=1) for _ in range(3)])
    cb._ingest_infos([terminal_info(win=0) for _ in range(5)])
    cb._ingest_infos([terminal_info(win=0, loss=0, draw=True, double_ko=True)])
    cb._ingest_infos([terminal_info(win=0, loss=0, timeout=True) for _ in range(2)])
    r = cb._compute_records()
    total = (r["episodes/win_rate"] + r["episodes/loss_rate"]
             + r["episodes/draw_rate"] + r["episodes/timeout_rate"])
    assert total == pytest.approx(1.0)
    assert r["episodes/win_rate"] == pytest.approx(3 / 11)
    assert r["episodes/loss_rate"] == pytest.approx(5 / 11)


def test_draw_rate_and_legacy_double_ko_rate_are_the_same_number():
    """double_ko_rate is kept so saved dashboards and every existing Run A
    comparison keep resolving to the same series."""
    cb = make_cb()
    cb._ingest_infos([terminal_info(win=0, loss=0, draw=True, double_ko=True)])
    cb._ingest_infos([terminal_info(win=1)])
    r = cb._compute_records()
    assert r["episodes/double_ko_rate"] == r["episodes/draw_rate"] == pytest.approx(0.5)


def test_a_draw_is_counted_from_the_draw_key_alone():
    """Envs emit both keys, but the callback must not depend on the legacy
    alias -- an env that only sets `draw` still has to be counted."""
    cb = make_cb()
    cb._ingest_infos([terminal_info(win=0, loss=0, draw=True)])  # no double_ko
    assert cb._compute_records()["episodes/draw_rate"] == pytest.approx(1.0)


def test_a_terminal_info_without_a_loss_key_does_not_crash_the_window():
    """Back-compat with any producer of terminal infos that predates the
    round-semantics fix -- a replayed log, a pinned worker, a third-party
    wrapper -- which emits neither `loss` nor `draw`. All three shipped envs
    emit both now. Those episodes must still aggregate."""
    cb = make_cb()
    cb._ingest_infos([{"win": 1, "hp_sentinel": False}])
    r = cb._compute_records()
    assert r["episodes/win_rate"] == pytest.approx(1.0)
    assert r["episodes/loss_rate"] == pytest.approx(0.0)
    assert r["episodes/draw_rate"] == pytest.approx(0.0)


def test_time_over_rate_splits_the_same_episodes_by_cause():
    """time_over_rate cuts by CAUSE (clock vs KO), not by outcome, so it must
    not disturb the win/loss/draw/timeout partition. It is the only metric
    that can show a transport which cannot see the round clock: it stays
    pinned at 0.0 there while real time overs pile up as timeouts."""
    cb = make_cb()
    cb._ingest_infos([terminal_info(win=1, loss=0, time_over=True),
                      terminal_info(win=0, loss=1, time_over=False)])
    r = cb._compute_records()
    assert r["episodes/time_over_rate"] == pytest.approx(0.5)
    assert (r["episodes/win_rate"] + r["episodes/loss_rate"]
            + r["episodes/draw_rate"] + r["episodes/timeout_rate"]) == pytest.approx(1.0)


def test_time_over_rate_is_absent_when_no_episode_reported_the_key():
    cb = make_cb()
    cb._ingest_infos([{"win": 1, "hp_sentinel": False}])
    assert "episodes/time_over_rate" not in cb._compute_records()
