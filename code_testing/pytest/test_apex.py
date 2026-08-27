# Tests del nucleo Ape-X: wire de transiciones (fp16 sin perdida en canales
# enteros), payload de pesos+config, escalera de epsilon, y ApexLearner
# (ingesta -> buffer -> train_tick -> versionado de pesos). Sin HTTP ni
# emulador: todo el nucleo es puro a proposito.

import random
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

torch = pytest.importorskip("torch")

from agents.apex import (ApexLearner, MAX_BATCH_B64_BYTES, apex_epsilon,  # noqa: E402
                         decode_transitions, decode_weights,
                         encode_transitions, encode_weights)
from agents.rainbow import QRDuelingNet  # noqa: E402
from es.policy import OBS_DIM  # noqa: E402


def _fake_transitions(n, rng):
    out = []
    for _ in range(n):
        # canales enteros como los reales (posiciones, HP, IDs < 2048)
        obs = rng.integers(0, 500, OBS_DIM).astype(np.float32)
        nxt = rng.integers(0, 500, OBS_DIM).astype(np.float32)
        out.append((obs, int(rng.integers(0, 63)),
                    np.float32(rng.normal()), nxt,
                    bool(rng.random() < 0.1), np.float32(0.99 ** 3)))
    return out


class TestEpsilonLadder:
    def test_endpoints_and_monotonic(self):
        assert apex_epsilon(0, 8) == pytest.approx(0.4)
        ladder = [apex_epsilon(r, 8) for r in range(8)]
        assert all(a > b for a, b in zip(ladder, ladder[1:]))
        assert ladder[-1] == pytest.approx(0.4 ** 8)

    def test_single_proc(self):
        assert apex_epsilon(0, 1) == pytest.approx(0.4)


class TestTransitionWire:
    def test_roundtrip_exact_for_integer_channels(self):
        rng = np.random.default_rng(0)
        trans = _fake_transitions(50, rng)
        back = decode_transitions(encode_transitions(trans))
        assert len(back) == 50
        for a, b in zip(trans, back):
            np.testing.assert_array_equal(a[0], b[0].astype(np.float32))
            np.testing.assert_array_equal(a[3], b[3].astype(np.float32))
            assert a[1] == b[1] and a[4] == b[4]
            assert a[2] == pytest.approx(b[2]) and a[5] == pytest.approx(b[5])

    def test_wrong_obs_dim_rejected(self):
        rng = np.random.default_rng(1)
        bad = [(np.zeros(10, dtype=np.float32), 0, np.float32(0),
                np.zeros(10, dtype=np.float32), False, np.float32(0.9))]
        with pytest.raises(ValueError, match="obs dim"):
            decode_transitions(encode_transitions(bad))

    def test_oversized_batch_rejected(self):
        import base64
        with pytest.raises(ValueError, match="too large"):
            decode_transitions(base64.b64encode(
                b"\x00" * (MAX_BATCH_B64_BYTES + 1)).decode())


class TestWeightsWire:
    def test_roundtrip_bitexact_and_config(self):
        net = QRDuelingNet(24, n_actions=5, n_quantiles=7, hidden=16)
        payload = encode_weights(net, 3, {"in_dim": 24, "quantiles": 7,
                                          "hidden": 16, "onehot": False,
                                          "gamma": 0.99, "n_step": 3})
        twin = QRDuelingNet(24, n_actions=5, n_quantiles=7, hidden=16)
        version, config, _state = decode_weights(payload, net=twin)
        assert version == 3 and config["quantiles"] == 7
        x = torch.randn(2, 24)
        assert torch.allclose(net(x), twin(x))

    def test_stale_arch_fails_loud(self):
        net = QRDuelingNet(24, n_actions=5, n_quantiles=7, hidden=16)
        payload = encode_weights(net, 1, {})
        wrong = QRDuelingNet(24, n_actions=5, n_quantiles=7, hidden=32)
        with pytest.raises(RuntimeError):
            decode_weights(payload, net=wrong)


class TestApexLearner:
    def _learner(self, **kw):
        defaults = dict(hidden=32, quantiles=9, onehot=True,
                        buffer_capacity=4096, device="cpu",
                        weights_every_grads=2)
        defaults.update(kw)
        return ApexLearner(**defaults)

    def test_ingest_fills_buffer_and_tracks_actors(self):
        learner = self._learner()
        rng = np.random.default_rng(2)
        body = {"actor": "legion-1",
                "b64": encode_transitions(_fake_transitions(30, rng)),
                "stats": {"procs": 8, "steps_per_s": 4000.0},
                "episodes": {"wins": 3, "count": 5}}
        ack = learner.ingest(body, now=100.0)
        assert ack["accepted"] and ack["buffer"] == 30
        s = learner.status(now=101.0)
        assert s["actors"]["legion-1"]["transitions"] == 30
        assert s["actors"]["legion-1"]["procs"] == 8
        assert s["win_rate_cum"] == pytest.approx(0.6)

    def test_ingest_tracks_recent_and_per_level(self):
        learner = self._learner()
        rng = np.random.default_rng(9)
        learner.ingest({"actor": "a",
                        "b64": encode_transitions(_fake_transitions(5, rng)),
                        "episodes": {"wins": 3, "count": 4,
                                     "levels": {"1": [3, 3], "3": [0, 1]}}},
                       now=1.0)
        s = learner.status(now=2.0)
        assert s["win_rate_recent200"] == pytest.approx(0.75)
        assert s["win_rate_recent_by_lvl"] == {"1": 1.0, "3": 0.0}
        # campo levels ausente (actor viejo): no truena, solo no segmenta
        learner.ingest({"actor": "b",
                        "b64": encode_transitions(_fake_transitions(3, rng)),
                        "episodes": {"wins": 1, "count": 1}}, now=3.0)
        assert learner.status(now=4.0)["episodes"] == 5

    def test_train_tick_none_until_batch_available(self):
        learner = self._learner()
        assert learner.train_tick(64, beta=0.4, rng=random.Random(0)) is None

    def test_train_tick_runs_and_versions_weights(self):
        learner = self._learner()
        rng = np.random.default_rng(3)
        learner.ingest({"actor": "a",
                        "b64": encode_transitions(_fake_transitions(200, rng))},
                       now=1.0)
        prng = random.Random(0)
        assert learner.weights_version == 0
        for _ in range(4):  # weights_every_grads=2 -> 2 bumps
            out = learner.train_tick(64, beta=0.4, rng=prng)
            assert out is not None
        assert learner.grad_steps == 4
        assert learner.weights_version == 2
        # el payload publicado corresponde a la version nueva
        assert learner.weights_payload()["version"] == 2

    def test_onehot_featurization_dim(self):
        learner = self._learner()
        rng = np.random.default_rng(4)
        feats = learner._featurize(_fake_transitions(3, rng))
        assert feats[0][0].shape == (212,)

    def test_raw_featurization_when_onehot_off(self):
        learner = self._learner(onehot=False)
        rng = np.random.default_rng(5)
        feats = learner._featurize(_fake_transitions(3, rng))
        assert feats[0][0].shape == (OBS_DIM,)
        assert feats[0][0].dtype == np.float32
