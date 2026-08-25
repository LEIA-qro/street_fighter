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
