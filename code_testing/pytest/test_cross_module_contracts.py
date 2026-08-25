# test_cross_module_contracts.py
#
# Regression tests for invariants that must hold BETWEEN independently
# developed modules -- exactly the seams a per-task/per-file review cannot
# see, because each task's diff looked correct in isolation. Each test here
# corresponds to a Critical/Important finding from the whole-branch review
# that no single per-task review could have caught.

import os
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest
import torch

from envs.reward import RewardConfig
from agents.ppo.hyperparams import build_ppo_kwargs
from core.sf2_extractor import SF2FeaturesExtractor
from envs.sf2_v4 import V4_FRAME_DIM
import core.config as config
from fakes.fake_bizhawk import FakeBizHawkEnvV4, make_payload


# --------------------------------------------------------------------------
# C3: the distance-shaping term in reward.py is potential-based shaping,
# F(s, s') = gamma * Phi(s') - Phi(s). That form is policy-invariant (Ng,
# Harada & Russell, ICML 1999) ONLY when gamma equals the discount of the
# agent that acts on the reward. Task 4 raised the PPO agent's discount to
# 0.995 without updating RewardConfig's default of 0.99; the residual
# 0.005*Phi(s') is then a real per-step reward, maximal at peak_dist and
# pointing backwards against the spacing behaviour Task 3 exists to reward.
# --------------------------------------------------------------------------

def test_reward_config_gamma_matches_the_ppo_agents_discount():
    ppo_gamma = build_ppo_kwargs(lr=3e-4, ent_coef=0.01, clip_range=0.2)["gamma"]
    assert RewardConfig().gamma == pytest.approx(ppo_gamma), (
        "RewardConfig.gamma and build_ppo_kwargs(...)['gamma'] have drifted "
        "apart -- potential-based shaping is only policy-invariant when the "
        "shaping discount matches the acting agent's discount. Both must "
        "be sourced from the same shared constant, not two separate "
        "literals."
    )


# --------------------------------------------------------------------------
# v4 frame index contract: SF2FeaturesExtractor slices the tail of each
# 23-float v4 frame at fixed offsets (p1_act_hi=15, p1_act_lo=17, p1_btn=19,
# p1_char=21, ...). test_sf2_extractor.py only re-derives the extractor's OWN
# assumptions about those offsets -- it says nothing about whether
# sf2_v4.py's `tail` array actually puts the right field at each one.
# Reordering sf2_v4.py's `tail` list would keep every existing test green
# while silently feeding action IDs into the character embedding. Drive the
# real env parser (not a hand-built tensor) with distinguishable field
# values and check they land exactly where the extractor reads them from.
# --------------------------------------------------------------------------

def test_v4_env_emitted_frame_matches_the_extractor_index_contract():
    env = FakeBizHawkEnvV4([make_payload(
        176, 176, extended=True,
        p1_act=7, p1_act_lo=9, p1_btn=11, p1_char=3,
    )])
    obs, _ = env.reset()
    frame0 = obs[:V4_FRAME_DIM]

    # sf2_v4.py's tail layout puts continuous(10) + rel_y_dist/p1_head/p2_head
    # (3) + p1_air/p2_air (2) first, then the 8 category IDs starting at 15:
    # p1_act_hi, p2_act_hi, p1_act_lo, p2_act_lo, p1_btn, p2_btn, p1_char,
    # p2_char -- i.e. indices [15, 17, 19, 21] below.
    assert frame0[15] == pytest.approx(7.0), "p1_act (hi byte) not at index 15"
    assert frame0[17] == pytest.approx(9.0), "p1_act_lo not at index 17"
    assert frame0[19] == pytest.approx(11.0), "p1_btn not at index 19"
    assert frame0[21] == pytest.approx(3.0), "p1_char not at index 21"

    # SF2FeaturesExtractor must read those SAME offsets as the action-hi
    # embedding input -- perturbing only index 15 must move its output, and
    # nothing about this assertion depends on the extractor's own internal
    # constants (unlike test_sf2_extractor.py's coverage).
    ex = SF2FeaturesExtractor(env.observation_space, n_frames=config.NUM_FRAMES)
    base = torch.from_numpy(obs).unsqueeze(0).float()
    moved = base.clone()
    moved[0, 15] = 200.0
    assert not torch.allclose(ex(base), ex(moved))
