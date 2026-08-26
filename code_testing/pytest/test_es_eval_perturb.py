# Tests de las perturbaciones de evaluacion (run 3): RNG pareado por par
# antitetico, identidad en checkpoint, lease+echo en el wire, y aplicacion
# real en el worker (gemelos sufren perturbaciones identicas).

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from es import openes, protocol  # noqa: E402
from es.coordinator import Coordinator  # noqa: E402
from es.policy import MLPPolicy, OBS_DIM  # noqa: E402


class TestPairedRng:
    def test_same_pair_seed_same_stream(self):
        a = openes.eval_rng_for_episode(12345, 0)
        b = openes.eval_rng_for_episode(12345, 0)
        assert list(a.integers(0, 31, 10)) == list(b.integers(0, 31, 10))
        assert list(a.random(5)) == list(b.random(5))

    def test_episodes_get_independent_streams(self):
        a = openes.eval_rng_for_episode(12345, 0)
        b = openes.eval_rng_for_episode(12345, 1)
        assert list(a.integers(0, 1000, 8)) != list(b.integers(0, 1000, 8))

    def test_stream_disjoint_from_state_rotation_stream(self):
        # states_for_member y eval_rng usan spawn keys distintos del MISMO
        # pair seed: no deben producir la misma secuencia
        picks = openes.states_for_member(777, 8, 1000)
        rng = openes.eval_rng_for_episode(777, 0)
        assert picks != [int(x) for x in rng.integers(0, 1000, 8)]


class TestIdentity:
    def _state(self, **kw):
        theta = MLPPolicy.init_flat(3)
        return openes.init_state(theta, 0.02, 0.01, 0.0, 3, states=("A", "B"),
                                 **kw)

    def test_checkpoint_roundtrip(self, tmp_path):
        state = self._state(eval_desync_max=30, eval_action_noise=0.05)
        base = str(tmp_path / "gen_000002")
        openes.save_checkpoint(state, base)
        loaded = openes.load_checkpoint(base)
        assert loaded.eval_desync_max == 30
        assert loaded.eval_action_noise == 0.05

    def test_pre_perturbation_checkpoint_defaults_clean(self, tmp_path):
        import json
        state = self._state()
        base = str(tmp_path / "gen_000001")
        openes.save_checkpoint(state, base)
        with open(base + ".json") as f:
            meta = json.load(f)
        del meta["eval_desync_max"]
        del meta["eval_action_noise"]
        with open(base + ".json", "w") as f:
            json.dump(meta, f)
        loaded = openes.load_checkpoint(base)
        assert loaded.eval_desync_max == 0 and loaded.eval_action_noise == 0.0

    def test_resume_pins_checkpoint_over_cli(self, tmp_path, capsys):
        import argparse
        from es.coordinator import load_or_init_state
        state = self._state(eval_desync_max=30, eval_action_noise=0.05)
        openes.save_checkpoint(state, str(tmp_path / "gen_000004"))
        args = argparse.Namespace(checkpoint_dir=str(tmp_path), s3_bucket=None,
                                  sigma=0.02, lr=0.01, weight_decay=0.0,
                                  master_seed=3, policy="v4",
                                  eval_desync_max=0, eval_action_noise=0.0)
        resumed = load_or_init_state(args, states=("A", "B"))
        assert resumed.eval_desync_max == 30
        assert resumed.eval_action_noise == 0.05
        out = capsys.readouterr().out
        assert "checkpoint pins eval_desync_max=30" in out


class TestWire:
    def test_fingerprint_none_when_clean(self):
        assert protocol.eval_fingerprint(0, 0.0) is None
        assert protocol.eval_fingerprint(None, None) is None

    def test_fingerprint_stable_through_json(self):
        import json
        fp = protocol.eval_fingerprint(30, 0.05)
        roundtrip = json.loads(json.dumps({"desync_max": 30, "action_noise": 0.05}))
        assert protocol.eval_fingerprint(roundtrip["desync_max"],
                                         roundtrip["action_noise"]) == fp

    def _coord(self, **state_kw):
        theta = MLPPolicy.init_flat(1)
        state = openes.init_state(theta, 0.02, 0.01, 0.0, 1, states=("A",),
                                  **state_kw)
        coord = Coordinator(state, pop_size=4, chunk_size=4, episodes=2,
                            lease_seconds=60, states=state.states)
        coord.start_generation()
        return coord

    def test_lease_carries_eval_params(self):
        coord = self._coord(eval_desync_max=30, eval_action_noise=0.05)
        work = coord.lease_work("w1")
        assert work["eval"] == {"desync_max": 30, "action_noise": 0.05}

    def test_clean_run_lease_has_no_eval_key(self):
        coord = self._coord()
        assert "eval" not in coord.lease_work("w1")

    def test_result_without_echo_refused(self, capsys):
        coord = self._coord(eval_desync_max=30, eval_action_noise=0.05)
        work = coord.lease_work("w1")
        body = {"worker": "w1", "generation": work["generation"],
                "chunk_id": work["chunk_id"],
                "member_idx": [m[0] for m in work["members"]],
                "fitnesses": [0.0] * len(work["members"]),
                "states_fingerprint": protocol.states_fingerprint(["A"])}
        assert coord.submit_result(body) is False
        assert "eval perturbations" in capsys.readouterr().out
        body["eval_fingerprint"] = protocol.eval_fingerprint(30, 0.05)
        assert coord.submit_result(body) is True


class _CountingEnv:
    """Registra las acciones de cada episodio para comparar gemelos."""

    def __init__(self):
        self.episodes = []

    def reset(self, options=None):
        self.episodes.append([])
        return np.zeros(OBS_DIM, dtype=np.float32), {}

    def step(self, action):
        self.episodes[-1].append(tuple(int(x) for x in action))
        done = len(self.episodes[-1]) >= 40
        info = {"win": 1, "my_hp": 176, "enemy_hp": 0} if done else {}
        return np.zeros(OBS_DIM, dtype=np.float32), 0.0, done, False, info


class TestWorkerApplication:
    def _run_member(self, monkeypatch, sign, eval_params):
        from es import worker
        env = _CountingEnv()
        monkeypatch.setattr(worker, "_make_env", lambda: env)
        monkeypatch.setattr(worker, "_ENV", None)
        theta = MLPPolicy.init_flat(0)
        worker.evaluate_member((theta, 0.02, 999, sign, 2, None, "v4", eval_params))
        return env.episodes

    def test_antithetic_twins_get_identical_perturbations(self, monkeypatch):
        eval_params = {"desync_max": 12, "action_noise": 0.3}
        eps_plus = self._run_member(monkeypatch, +1, eval_params)
        eps_minus = self._run_member(monkeypatch, -1, eval_params)
        for ep_p, ep_m in zip(eps_plus, eps_minus):
            # mismo prefijo neutral sorteado (mismo largo de (0,0) iniciales)
            def neutral_prefix(ep):
                n = 0
                for a in ep:
                    if a != (0, 0):
                        break
                    n += 1
                return n
            # el desfase sorteado es identico; y como theta_+ y theta_- dan
            # politicas distintas, la igualdad del prefijo no es casualidad
            # de acciones iguales sino del MISMO sorteo del par
            assert neutral_prefix(ep_p) == neutral_prefix(ep_m)

    def test_clean_member_unchanged(self, monkeypatch):
        eps_none = self._run_member(monkeypatch, +1, None)
        eps_again = self._run_member(monkeypatch, +1, None)
        assert eps_none == eps_again  # determinista, sin perturbaciones
