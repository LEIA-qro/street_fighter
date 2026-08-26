# Tests del policy v4onehot (one-hot de character IDs) y su cableado:
# expansion de features, registro POLICIES, clave "policy" en el wire /theta,
# anclaje en checkpoint, y compatibilidad hacia atras del worker.

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from es import openes, protocol  # noqa: E402
from es import policy as policy_mod  # noqa: E402
from es.policy import (  # noqa: E402
    DEFAULT_POLICY, NUM_FRAMES, OBS_DIM, OBS_FRAME_DIM, POLICIES,
    CharOneHotPolicy, MLPPolicy, N_CHARS, ONEHOT_FRAME_DIM, ONEHOT_OBS_DIM,
    expand_char_onehot,
)


def _obs_with_chars(p1, p2, fill=0.0):
    obs = np.full(OBS_DIM, fill, dtype=np.float32)
    for f in range(NUM_FRAMES):
        obs[f * OBS_FRAME_DIM + 21] = p1
        obs[f * OBS_FRAME_DIM + 22] = p2
    return obs


class TestExpansion:
    def test_dims(self):
        assert ONEHOT_FRAME_DIM == 21 + 2 * 16 == 53
        assert ONEHOT_OBS_DIM == 212
        assert expand_char_onehot(np.zeros(OBS_DIM, dtype=np.float32)).shape == (212,)

    def test_onehot_positions(self):
        out = expand_char_onehot(_obs_with_chars(3, 11)).reshape(NUM_FRAMES,
                                                                 ONEHOT_FRAME_DIM)
        for f in range(NUM_FRAMES):
            p1_hot = out[f, 21:21 + N_CHARS]
            p2_hot = out[f, 21 + N_CHARS:]
            assert p1_hot[3] == 1.0 and p1_hot.sum() == 1.0
            assert p2_hot[11] == 1.0 and p2_hot.sum() == 1.0

    def test_continuous_passthrough_scaled(self):
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[0] = 88.0   # rel x, escala 176 -> 0.5
        obs[23] = 176.0  # frame 1, mismo canal -> 1.0
        out = expand_char_onehot(obs).reshape(NUM_FRAMES, ONEHOT_FRAME_DIM)
        assert out[0, 0] == pytest.approx(0.5)
        assert out[1, 0] == pytest.approx(1.0)

    def test_scale_matches_base_policy_channels(self):
        # los 21 canales continuos usan EXACTAMENTE la escala del policy v4
        rng = np.random.default_rng(7)
        obs = (rng.random(OBS_DIM, dtype=np.float32) * 100).astype(np.float32)
        obs = _obs_with_chars(0, 0) + obs * 0  # chars limpios
        obs[:21] = rng.random(21).astype(np.float32) * 50
        got = expand_char_onehot(obs).reshape(NUM_FRAMES, ONEHOT_FRAME_DIM)[0, :21]
        expected = obs[:21] * policy_mod.OBS_SCALE[:21]
        np.testing.assert_allclose(got, expected, rtol=1e-6)

    def test_corrupt_char_id_clipped_not_crash(self):
        out = expand_char_onehot(_obs_with_chars(200, -3))
        frame = out.reshape(NUM_FRAMES, ONEHOT_FRAME_DIM)[0]
        assert frame[21 + 15] == 1.0        # clip alto -> 15
        assert frame[21 + N_CHARS + 0] == 1.0  # clip bajo -> 0


class TestPolicyClasses:
    def test_num_params(self):
        assert MLPPolicy.num_params() == 14207
        # W1 64x212 + b1 + W2 64x64 + b2 + W3 63x64 + b3
        assert CharOneHotPolicy.num_params() == 13568 + 64 + 4096 + 64 + 4032 + 63

    def test_module_level_compat(self):
        assert policy_mod.NUM_PARAMS == 14207
        np.testing.assert_array_equal(policy_mod.init_flat(5), MLPPolicy.init_flat(5))

    def test_init_flat_deterministic_and_sized(self):
        a = CharOneHotPolicy.init_flat(42)
        b = CharOneHotPolicy.init_flat(42)
        np.testing.assert_array_equal(a, b)
        assert a.shape == (CharOneHotPolicy.num_params(),)
        assert a.dtype == np.float32

    def test_act_takes_raw_92_obs(self):
        pol = CharOneHotPolicy(CharOneHotPolicy.init_flat(1))
        action = pol.act(_obs_with_chars(2, 9, fill=1.0))
        assert action.shape == (2,)
        assert 0 <= action[0] <= 8 and 0 <= action[1] <= 6

    def test_set_flat_rejects_wrong_dim(self):
        with pytest.raises(ValueError):
            CharOneHotPolicy(np.zeros(14207, dtype=np.float32))
        with pytest.raises(ValueError):
            MLPPolicy(np.zeros(CharOneHotPolicy.num_params(), dtype=np.float32))

    def test_char_change_can_change_action(self):
        # la razon de ser: dos rivales distintos DEBEN poder producir acciones
        # distintas con el mismo estado fisico. Con pesos random basta un seed
        # donde difieran para probar que la arquitectura ramifica.
        for seed in range(20):
            pol = CharOneHotPolicy(CharOneHotPolicy.init_flat(seed))
            a = pol.act(_obs_with_chars(0, 4, fill=10.0))
            b = pol.act(_obs_with_chars(0, 12, fill=10.0))
            if not np.array_equal(a, b):
                return
        pytest.fail("20 seeds y el char ID nunca cambio la accion")

    def test_registry(self):
        assert POLICIES["v4"] is MLPPolicy
        assert POLICIES["v4onehot"] is CharOneHotPolicy
        assert DEFAULT_POLICY == "v4"


class TestWire:
    def test_encode_theta_carries_policy(self):
        theta = np.arange(4, dtype=np.float32)
        payload = protocol.encode_theta(theta, 3, policy="v4onehot")
        assert payload["policy"] == "v4onehot"
        version, decoded = protocol.decode_theta(payload)
        assert version == 3
        np.testing.assert_array_equal(decoded, theta)

    def test_encode_theta_omits_policy_when_none(self):
        payload = protocol.encode_theta(np.zeros(2, dtype=np.float32), 0)
        assert "policy" not in payload


class TestCheckpointIdentity:
    def _state(self, policy="v4onehot"):
        theta = POLICIES[policy].init_flat(9)
        return openes.init_state(theta, sigma=0.02, lr=0.01, weight_decay=0.0,
                                 master_seed=9, states=("A", "B"), policy=policy)

    def test_roundtrip_pins_policy(self, tmp_path):
        state = self._state()
        base = str(tmp_path / "gen_000005")
        openes.save_checkpoint(state, base)
        loaded = openes.load_checkpoint(base)
        assert loaded.policy == "v4onehot"
        np.testing.assert_array_equal(loaded.theta, state.theta)

    def test_resume_pins_checkpoint_policy_over_cli(self, tmp_path, capsys):
        # EL escenario de despliegue: un checkpoint v4 vivo + unit nuevo con
        # --policy v4onehot. Debe resumir v4 con warning, jamas reiniciar.
        import argparse
        from es.coordinator import load_or_init_state
        state = self._state(policy="v4")
        openes.save_checkpoint(state, str(tmp_path / "gen_000003"))
        args = argparse.Namespace(checkpoint_dir=str(tmp_path), s3_bucket=None,
                                  sigma=0.02, lr=0.01, weight_decay=0.0,
                                  master_seed=9, policy="v4onehot")
        resumed = load_or_init_state(args, states=("A", "B"))
        assert resumed.policy == "v4"
        assert resumed.generation == state.generation
        np.testing.assert_array_equal(resumed.theta, state.theta)
        assert "checkpoint pins policy=v4" in capsys.readouterr().out

    def test_pre_registry_checkpoint_defaults_v4(self, tmp_path):
        import json
        state = self._state(policy="v4")
        base = str(tmp_path / "gen_000001")
        openes.save_checkpoint(state, base)
        with open(base + ".json") as f:
            meta = json.load(f)
        del meta["policy"]  # checkpoint escrito antes del registro
        with open(base + ".json", "w") as f:
            json.dump(meta, f)
        assert openes.load_checkpoint(base).policy == "v4"


class TestWorkerCompat:
    def test_evaluate_member_6_tuple_defaults_v4(self, monkeypatch):
        from es import worker

        class FakeEnv:
            def reset(self, options=None):
                return np.zeros(OBS_DIM, dtype=np.float32), {}

            def step(self, action):
                info = {"win": 1, "my_hp": 176, "enemy_hp": 0}
                return np.zeros(OBS_DIM, dtype=np.float32), 0.0, True, False, info

        monkeypatch.setattr(worker, "_make_env", lambda: FakeEnv())
        monkeypatch.setattr(worker, "_ENV", None)
        theta = MLPPolicy.init_flat(0)
        fit6, steps6 = worker.evaluate_member((theta, 0.02, 123, 1, 1, None))
        fit7, steps7 = worker.evaluate_member((theta, 0.02, 123, 1, 1, None, "v4"))
        assert fit6 == fit7 and steps6 == steps7

    def test_evaluate_member_onehot_theta(self, monkeypatch):
        from es import worker

        class FakeEnv:
            def reset(self, options=None):
                return np.zeros(OBS_DIM, dtype=np.float32), {}

            def step(self, action):
                info = {"win": 0, "my_hp": 0, "enemy_hp": 176}
                return np.zeros(OBS_DIM, dtype=np.float32), 0.0, True, False, info

        monkeypatch.setattr(worker, "_make_env", lambda: FakeEnv())
        monkeypatch.setattr(worker, "_ENV", None)
        theta = CharOneHotPolicy.init_flat(0)
        fit, steps = worker.evaluate_member(
            (theta, 0.02, 5, -1, 1, None, "v4onehot"))
        assert steps == 1  # el theta de 21887 solo corre si construyo v4onehot
