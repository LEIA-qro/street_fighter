# test_cross_module_contracts.py
#
# Regression tests for invariants that must hold BETWEEN independently
# developed modules -- exactly the seams a per-task/per-file review cannot
# see, because each task's diff looked correct in isolation. Each test here
# corresponds to a Critical/Important finding from the whole-branch review
# that no single per-task review could have caught.

import ast
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

    # Stronger check: SF2FeaturesExtractor.forward passes
    # frames[:, :, :self._id_start] through UNCHANGED and only routes
    # [self._id_start:] through .round().long() into embedding tables. The
    # assertion above (moving index 15 by 193.0) would still pass even if
    # cont_dim/flag_dim drifted and index 15 landed in that passthrough
    # slice instead of the embedding path -- a float passthrough is also
    # "moved" by a large perturbation. Perturb by +0.4 instead: that rounds
    # to the SAME integer ID (7 -> 7.4 -> round() -> 7), so the embedding
    # lookup is unchanged. Passing this only holds if index 15 is genuinely
    # rounded into a discrete embedding index rather than fed through as a
    # continuous float, which is exactly the contract the comment above
    # claims to verify.
    nudged = base.clone()
    nudged[0, 15] = nudged[0, 15] + 0.4
    assert torch.allclose(ex(base), ex(nudged)), (
        "perturbing index 15 by +0.4 (same rounded integer ID) changed the "
        "extractor's output -- index 15 is being read as a continuous "
        "passthrough value, not rounded into the embedding table as the "
        "frame-index contract requires"
    )


# --------------------------------------------------------------------------
# Discount-factor guard: Task 4 raised the PPO agent's discount to
# AGENT_GAMMA (0.995) and pointed RewardConfig.gamma / build_ppo_kwargs at
# the shared constant, but four other live entrypoints
# (pbt_orchestrator.py, train_exploiter.py, train_league.py, dqn/agent.py)
# still hardcoded gamma=0.99 (or, in dqn/agent.py's case, gamma=0.995 that
# happened to agree by coincidence). Because the distance-shaping term in
# reward.py is potential-based shaping F(s, s') = gamma*Phi(s') - Phi(s),
# ANY entrypoint whose discount diverges from RewardConfig.gamma turns that
# shaping term into a real per-step reward -- see rl_constants.py's
# docstring. A point fix only closes today's four sites; this test scans
# src/ so the NEXT hardcoded gamma= or GAMMA = literal fails CI instead of
# shipping silently.
#
# AST (not regex/string search): a regex over "gamma" would also flag
# comments, docstrings, and the legitimate optuna trial.suggest_float(...)
# calls (see the exclusion note below). Walking the AST lets us match
# exactly `gamma=<Constant numeric>` keyword arguments and module-level
# `GAMMA = <Constant numeric>` assignments, and nothing else.
# --------------------------------------------------------------------------

# Files intentionally excluded from the scan, with the specific reason each
# one is safe to leave alone:
#
# - core/rl_constants.py: this IS the single source of truth. AGENT_GAMMA is
#   defined here on purpose; scanning it would be flagging the fix itself.
# - agents/sac/agent.py: SACAgent.train() and .tune() both raise
#   NotImplementedError as their first statement (see _SAC_DISCRETE_MESSAGE
#   -- SB3 has no SAC-Discrete implementation, so SAC is kept only as an
#   unreachable reference for a future contributor). The `gamma=0.99` inside
#   SAC's dead PPO-style construction code can never execute, so it cannot
#   create the shaping mismatch this test guards against.
# - agents/ppo/optuna_study.py, agents/dqn/optuna_study.py,
#   agents/sac/optuna_study.py: these tune gamma via
#   `trial.suggest_float("gamma", 0.95, 0.9999)` and pass the resulting
#   VARIABLE as `gamma=gamma`, which is an ast.Name, not an ast.Constant --
#   the AST check below does not flag it regardless of exclusion list, so
#   these files do not need to be listed here. They ARE a real, separate
#   problem (tuned gamma drifts from the shaping gamma across trials) that
#   is documented in rl_constants.py and explicitly left unfixed by this
#   task -- see that module for the magnitude and the follow-up.
_GAMMA_SCAN_EXCLUDED_FILES = {
    os.path.normpath(os.path.join(SRC_PATH, "core", "rl_constants.py")),
    os.path.normpath(os.path.join(SRC_PATH, "agents", "sac", "agent.py")),
}


def _numeric_literal_value(node):
    """Return the literal's numeric value, or None if node isn't one.

    Handles plain Constant(0.99) and the UnaryOp(-, Constant(...)) shape a
    negative literal parses to (not expected for a discount factor, but
    handled for completeness rather than silently missing it).
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _numeric_literal_value(node.operand)
        if inner is not None:
            return -inner if isinstance(node.op, ast.USub) else inner
    return None


def _find_gamma_literal_violations(py_path):
    """Return a list of (line, description) for hardcoded discount literals.

    Flags:
      * any call keyword argument `gamma=<numeric literal>`
      * any module-level (or class-body) assignment `GAMMA = <numeric literal>`
    """
    with open(py_path, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source, filename=py_path)
    violations = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "gamma":
                    value = _numeric_literal_value(kw.value)
                    if value is not None:
                        violations.append((kw.value.lineno, f"gamma={value!r} keyword literal"))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "GAMMA":
                    value = _numeric_literal_value(node.value)
                    if value is not None:
                        violations.append((node.lineno, f"GAMMA = {value!r} module-level literal"))

    return violations


def test_no_hardcoded_discount_factor_literals_outside_rl_constants():
    violations = []
    for root, dirs, files in os.walk(SRC_PATH):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fname in files:
            if not fname.endswith(".py"):
                continue
            full_path = os.path.normpath(os.path.join(root, fname))
            if full_path in _GAMMA_SCAN_EXCLUDED_FILES:
                continue
            for lineno, description in _find_gamma_literal_violations(full_path):
                rel_path = os.path.relpath(full_path, PROJECT_ROOT)
                violations.append(f"{rel_path}:{lineno}: {description}")

    assert not violations, (
        "Found hardcoded discount-factor literal(s) outside "
        "core/rl_constants.py. The distance-shaping term in envs/reward.py "
        "is only policy-invariant when every acting agent's gamma matches "
        "RewardConfig.gamma -- import AGENT_GAMMA from core.rl_constants "
        "instead of hardcoding a new literal:\n  " + "\n  ".join(violations)
    )
