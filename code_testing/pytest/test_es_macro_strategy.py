# Tests de la oleada run-4: politica de macros (72 acciones), interfaz
# polimorfica de accion, decaimiento de sigma anclado a identidad, y el
# registro de estrategias de la flota.

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from envs.action_macros import MACROS, N_ACTIONS, decode  # noqa: E402
from es import openes, protocol  # noqa: E402
from es.coordinator import Coordinator  # noqa: E402
from es.policy import (  # noqa: E402
    CharOneHotMacroPolicy, CharOneHotPolicy, MLPPolicy, OBS_DIM, POLICIES,
    wrap_env_for_policy,
)


class TestMacroPolicy:
    def test_registry_and_shape(self):
        assert POLICIES["v4onehot_macro"] is CharOneHotMacroPolicy
        assert CharOneHotMacroPolicy.OUT_DIM == N_ACTIONS == 72
        assert CharOneHotMacroPolicy.ACTION_KIND == "macro"
        # 212*64+64 + 64*64+64 + 72*64+72
        assert CharOneHotMacroPolicy.num_params() == 13632 + 4160 + 4680

    def test_act_returns_flat_int_in_range(self):
        pol = CharOneHotMacroPolicy(CharOneHotMacroPolicy.init_flat(3))
        a = pol.act(np.ones(OBS_DIM, dtype=np.float32))
        assert isinstance(a, int) and 0 <= a < N_ACTIONS

    def test_polymorphic_neutral_and_random(self):
        rng = np.random.default_rng(0)
        assert CharOneHotMacroPolicy.neutral_action() == 0
        assert 0 <= CharOneHotMacroPolicy.random_action(rng) < N_ACTIONS
        nm = MLPPolicy.neutral_action()
        assert list(nm) == [0, 0]
        rm = CharOneHotPolicy.random_action(rng)
        assert 0 <= rm[0] <= 8 and 0 <= rm[1] <= 6

    def test_decode_neutral_flat_zero(self):
        # la accion neutral plana 0 decodifica a (0,0) en ambos lados
        assert decode(0, True) == [(0, 0)]
        assert decode(0, False) == [(0, 0)]

    def test_wrap_env_for_policy_kinds(self):
        import gymnasium as gym

        class _Env(gym.Env):  # gym.Wrapper exige un gym.Env de verdad
            action_space = None
            observation_space = None

        plain = _Env()
        assert wrap_env_for_policy(plain, CharOneHotPolicy) is plain
        wrapped = wrap_env_for_policy(plain, CharOneHotMacroPolicy)
        assert wrapped is not plain
        assert wrapped.action_space.n == N_ACTIONS
        assert wrapped.frame_size == 23

    def test_macros_are_at_most_three_steps(self):
        # el failsafe de episodios y la aproximacion semi-MDP asumen macros
        # cortos; si alguien agrega uno largo, que lo haga a proposito
        assert all(len(steps) <= 3 for steps in MACROS.values())


class TestSigmaDecay:
    def _state(self, **kw):
        theta = MLPPolicy.init_flat(1)
        return openes.init_state(theta, sigma=0.02, lr=0.01, weight_decay=0.0,
                                 master_seed=1, **kw)

    def test_constant_without_schedule(self):
        s = self._state()
        assert openes.sigma_for_generation(s, 0) == pytest.approx(0.02)
        assert openes.sigma_for_generation(s, 10_000) == pytest.approx(0.02)

    def test_exponential_schedule_endpoints_and_floor(self):
        s = self._state(sigma_final=0.01, sigma_decay_gens=100)
        assert openes.sigma_for_generation(s, 0) == pytest.approx(0.02)
        assert openes.sigma_for_generation(s, 50) == pytest.approx(
            0.02 * (0.5 ** 0.5))
        assert openes.sigma_for_generation(s, 100) == pytest.approx(0.01)
        assert openes.sigma_for_generation(s, 500) == pytest.approx(0.01)

    def test_es_update_advances_sigma(self):
        s = self._state(sigma_final=0.01, sigma_decay_gens=10)
        fits = np.random.default_rng(0).normal(size=8)
        s2 = openes.es_update(s, fits)
        assert s2.sigma == pytest.approx(
            openes.sigma_for_generation(s, 1))
        assert s2.sigma < s.sigma

    def test_checkpoint_roundtrip_schedule(self, tmp_path):
        s = self._state(sigma_final=0.012, sigma_decay_gens=600)
        base = str(tmp_path / "gen_000001")
        openes.save_checkpoint(s, base)
        loaded = openes.load_checkpoint(base)
        assert loaded.sigma0 == pytest.approx(0.02)
        assert loaded.sigma_final == pytest.approx(0.012)
        assert loaded.sigma_decay_gens == 600


class TestStrategyRegistry:
    def test_registry_contract(self):
        strat = openes.STRATEGIES["openes"]
        assert strat.name == "openes"
        s = openes.init_state(MLPPolicy.init_flat(0), 0.02, 0.01, 0.0, 1)
        members = strat.members_for_generation(s, 8)
        assert len(members) == 8
        s2 = strat.update(s, np.zeros(8))
        assert s2.generation == 1

    def test_strategy_pinned_in_checkpoint(self, tmp_path):
        s = openes.init_state(MLPPolicy.init_flat(0), 0.02, 0.01, 0.0, 1,
                              strategy="openes")
        base = str(tmp_path / "gen_000000")
        openes.save_checkpoint(s, base)
        assert openes.load_checkpoint(base).strategy == "openes"
        # checkpoint viejo sin la clave -> openes
        import json
        meta = json.load(open(base + ".json"))
        del meta["strategy"]
        json.dump(meta, open(base + ".json", "w"))
        assert openes.load_checkpoint(base).strategy == "openes"


class TestMacroRunCaps:
    def test_macro_policy_requires_macro_cap(self, capsys):
        theta = CharOneHotMacroPolicy.init_flat(1)
        state = openes.init_state(theta, 0.02, 0.01, 0.0, 1, states=("A",),
                                  policy="v4onehot_macro")
        coord = Coordinator(state, pop_size=4, chunk_size=4, episodes=1,
                            lease_seconds=60, states=state.states)
        coord.start_generation()
        assert "macro" in coord.required_caps
        resp = coord.lease_response("viejo", caps="states,eval")
        assert resp["work"] is None and "macro" in resp["reason"]
        assert coord.lease_response("nuevo", caps="states,eval,macro")["work"]

    def test_non_macro_policy_does_not_require_it(self):
        theta = CharOneHotPolicy.init_flat(1)
        state = openes.init_state(theta, 0.02, 0.01, 0.0, 1, states=("A",),
                                  policy="v4onehot")
        coord = Coordinator(state, pop_size=4, chunk_size=4, episodes=1,
                            lease_seconds=60, states=state.states)
        assert "macro" not in coord.required_caps


class TestWorkerMacroExecution:
    def test_evaluate_member_with_macro_policy_uses_flat_actions(self, monkeypatch):
        from es import worker

        seen = []

        class FakeWrappedEnv:
            def reset(self, options=None):
                return np.zeros(OBS_DIM, dtype=np.float32), {}

            def step(self, action):
                seen.append(action)
                done = len(seen) >= 3
                info = {"win": 1, "my_hp": 176, "enemy_hp": 0} if done else {}
                return np.zeros(OBS_DIM, dtype=np.float32), 0.0, done, False, info

        made = {}

        def fake_make_env(policy_cls):
            made["kind"] = policy_cls.ACTION_KIND
            return FakeWrappedEnv()

        monkeypatch.setattr(worker, "_make_env", fake_make_env)
        monkeypatch.setattr(worker, "_ENV", None)
        monkeypatch.setattr(worker, "_ENV_KIND", None)
        theta = CharOneHotMacroPolicy.init_flat(0)
        fit, steps = worker.evaluate_member(
            (theta, 0.02, 7, 1, 1, None, "v4onehot_macro", None))
        assert made["kind"] == "macro"
        assert all(isinstance(a, int) for a in seen)  # acciones planas
