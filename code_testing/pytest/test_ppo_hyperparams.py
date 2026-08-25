# test_ppo_hyperparams.py
#
# Offline tests for the PPO configuration builder. No model is constructed.

import os
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import pytest

from agents.ppo.hyperparams import (
    linear_schedule, build_ppo_kwargs, resolve_override,
)
from agents.dqn.agent import resolve_dqn_lr
from agents.sac.agent import resolve_sac_lr, resolve_sac_ent_coef


def test_linear_schedule_anneals_from_initial_to_final():
    sched = linear_schedule(3e-4, 0.0)
    assert sched(1.0) == pytest.approx(3e-4)   # start of training
    assert sched(0.5) == pytest.approx(1.5e-4)  # halfway
    assert sched(0.0) == pytest.approx(0.0)     # end of training


def test_resolve_override_accepts_explicit_zero():
    # None means "the CLI did not provide a value" -> use the phase value.
    assert resolve_override(None, 0.015) == pytest.approx(0.015)
    # 0.0 is a legitimate request to disable the term.
    assert resolve_override(0.0, 0.015) == pytest.approx(0.0)
    assert resolve_override(0.02, 0.015) == pytest.approx(0.02)


def test_build_ppo_kwargs_sets_every_coefficient_explicitly():
    kwargs = build_ppo_kwargs(lr=3e-4, ent_coef=0.01, clip_range=0.2)
    for key in ("learning_rate", "n_steps", "batch_size", "n_epochs",
                "gamma", "gae_lambda", "clip_range", "ent_coef",
                "vf_coef", "max_grad_norm", "normalize_advantage",
                "target_kl", "policy_kwargs", "device"):
        assert key in kwargs, f"missing explicit hyperparameter: {key}"

    # Presence alone would pass even if these shipped with the wrong values
    # (e.g. vf_coef=1.0 or normalize_advantage=False) -- pin the values the
    # plan calls load-bearing.
    assert kwargs["vf_coef"] == pytest.approx(0.5)
    assert kwargs["max_grad_norm"] == pytest.approx(0.5)
    assert kwargs["normalize_advantage"] is True
    assert kwargs["gae_lambda"] == pytest.approx(0.95)


def test_build_ppo_kwargs_annealing_produces_callables():
    kwargs = build_ppo_kwargs(lr=3e-4, ent_coef=0.01, clip_range=0.2,
                              anneal_lr=True)
    assert callable(kwargs["learning_rate"])
    assert kwargs["learning_rate"](1.0) == pytest.approx(3e-4)

    flat = build_ppo_kwargs(lr=3e-4, ent_coef=0.01, clip_range=0.2,
                            anneal_lr=False)
    assert flat["learning_rate"] == pytest.approx(3e-4)


def test_build_ppo_kwargs_defaults_disable_kl_early_stopping():
    """target_kl=0.03 with n_epochs=10 truncated most rollouts to 1-2 epochs.
    The default is now no early stop, with n_epochs lowered instead."""
    kwargs = build_ppo_kwargs(lr=3e-4, ent_coef=0.01, clip_range=0.2)
    assert kwargs["target_kl"] is None
    assert kwargs["n_epochs"] == 4


def test_build_ppo_kwargs_default_gamma_covers_a_full_round():
    """gamma=0.995 -> ~200 agent step horizon -> ~800 emulator frames,
    which spans a full SF2 round at FRAME_SKIP=4."""
    kwargs = build_ppo_kwargs(lr=3e-4, ent_coef=0.01, clip_range=0.2)
    assert kwargs["gamma"] == pytest.approx(0.995)


# Regression coverage for the DQN/SAC override sentinels. train.py's CLI now
# defaults --lr/--ent_coef/--clip_range to None (required for the PPO fix
# above), and that None is passed straight through to DQNAgent.train /
# SACAgent.train on every CLI invocation, even when no override is given.
# The old `lr if lr > 0.0 else ...` sentinel raises TypeError on `None > 0.0`,
# which would break `--algo dqn` and `--algo sac` from the CLI. These tests
# exercise the pure resolver functions directly -- no model or env is built.

def test_resolve_dqn_lr_accepts_none_and_explicit_zero():
    phase_params = {"lr": 5e-5}
    # Not provided (the CLI default) -> fall back to the phase value. This is
    # the exact call train.py makes on every unmodified `--algo dqn` run.
    assert resolve_dqn_lr(None, phase_params) == pytest.approx(5e-5)
    # 0.0 is a legitimate override and must survive, not be swallowed.
    assert resolve_dqn_lr(0.0, phase_params) == pytest.approx(0.0)
    assert resolve_dqn_lr(1e-3, phase_params) == pytest.approx(1e-3)


def test_resolve_sac_lr_accepts_none_and_explicit_zero():
    phase_params = {"lr": 5e-5}
    assert resolve_sac_lr(None, phase_params) == pytest.approx(5e-5)
    assert resolve_sac_lr(0.0, phase_params) == pytest.approx(0.0)
    assert resolve_sac_lr(1e-3, phase_params) == pytest.approx(1e-3)


def test_resolve_sac_ent_coef_accepts_none_and_explicit_zero():
    # Not provided -> SAC's own automatic entropy tuning.
    assert resolve_sac_ent_coef(None) == "auto"
    # 0.0 is a legitimate override that disables it, not a no-op.
    assert resolve_sac_ent_coef(0.0) == pytest.approx(0.0)
    assert resolve_sac_ent_coef(0.02) == pytest.approx(0.02)


def test_ppo_train_signature_defaults_lr_and_overrides_to_none():
    """PPOAgent.train defaults lr/ent_coef/clip_range to None (Task 4), the
    sentinel resolve_override() relies on. DQNAgent.train and SACAgent.train
    intentionally keep their pre-existing 0.0 signature defaults (train.py is
    their only caller and always passes an explicit value); their internal
    None-handling is covered separately by resolve_dqn_lr/resolve_sac_lr/
    resolve_sac_ent_coef above.
    """
    import inspect
    from agents.ppo.agent import PPOAgent

    ppo_defaults = inspect.signature(PPOAgent.train).parameters
    assert ppo_defaults["lr"].default is None
    assert ppo_defaults["ent_coef"].default is None
    assert ppo_defaults["clip_range"].default is None


def test_recurrent_kwargs_shorten_the_rollout_for_bptt():
    from agents.ppo.hyperparams import build_ppo_kwargs

    flat = build_ppo_kwargs(lr=3e-4, ent_coef=0.01, clip_range=0.2)
    lstm = build_ppo_kwargs(lr=3e-4, ent_coef=0.01, clip_range=0.2,
                            recurrent=True)
    assert lstm["n_steps"] == 512
    assert lstm["n_steps"] < flat["n_steps"]
    assert lstm["batch_size"] <= lstm["n_steps"]
