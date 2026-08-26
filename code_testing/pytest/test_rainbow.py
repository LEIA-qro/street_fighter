# Tests del nucleo Rainbow-lite (QR): sum-tree/PER, acumulador n-step,
# perdida de quantile regression, paso Double-DQN y wrappers de accion.
# Todo sin emulador: el nucleo es puro a proposito.

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

torch = pytest.importorskip("torch")

from agents.rainbow import (  # noqa: E402
    N_ACTIONS, NStepAccumulator, PERBuffer, QRDuelingNet, SumTree,
    linear_epsilon, make_taus, quantile_huber_loss, train_step,
)


class TestSumTree:
    def test_total_is_sum_of_leaves(self):
        t = SumTree(8)
        for i, p in enumerate([1, 2, 3, 4]):
            t.set(i, p)
        assert t.total() == pytest.approx(10.0)
        t.set(2, 0.5)
        assert t.total() == pytest.approx(7.5)

    def test_sample_lands_in_the_right_leaf(self):
        t = SumTree(4)
        t.set(0, 1.0)
        t.set(1, 3.0)
        t.set(2, 6.0)
        # acumulados: [0,1) -> 0, [1,4) -> 1, [4,10) -> 2
        assert t.sample(0.5) == 0
        assert t.sample(1.0) == 1
        assert t.sample(3.999) == 1
        assert t.sample(4.0) == 2
        assert t.sample(9.999) == 2

    def test_sampling_is_proportional(self):
        t = SumTree(4)
        t.set(0, 1.0)
        t.set(1, 9.0)
        rng = np.random.default_rng(0)
        hits = [t.sample(u) for u in rng.uniform(0, t.total(), 20_000)]
        frac1 = sum(1 for h in hits if h == 1) / len(hits)
        assert 0.88 < frac1 < 0.92  # esperado 0.9


class TestPER:
    def test_priorities_shift_sampling(self):
        buf = PERBuffer(64, alpha=1.0)
        for i in range(10):
            buf.push(("t", i))
        buf.update_priorities(list(range(10)), [0.001] * 9 + [100.0])
        idxs, _, _ = buf.sample(64, beta=1.0)
        assert sum(1 for i in idxs if i == 9) > 40  # domina la prioridad alta

    def test_is_weights_max_one_and_counterweight(self):
        buf = PERBuffer(64, alpha=1.0)
        for i in range(4):
            buf.push(("t", i))
        buf.update_priorities([0, 1, 2, 3], [1.0, 1.0, 1.0, 10.0])
        idxs, _, w = buf.sample(256, beta=1.0)
        assert w.max() == pytest.approx(1.0)
        by = {}
        for i, wi in zip(idxs, w):
            by[i] = wi
        # la transicion mas muestreada carga el peso IS mas chico
        assert by[3] < by[0]


class TestNStep:
    def test_three_step_return(self):
        acc = NStepAccumulator(3, gamma=0.9)
        assert acc.push("s0", 1, 1.0, "s1", False) == []
        assert acc.push("s1", 2, 2.0, "s2", False) == []
        out = acc.push("s2", 3, 4.0, "s3", False)
        assert len(out) == 1
        obs0, a0, R, obs_n, done, g = out[0]
        assert obs0 == "s0" and a0 == 1 and obs_n == "s3" and done is False
        assert R == pytest.approx(1.0 + 0.9 * 2.0 + 0.81 * 4.0)
        assert g == pytest.approx(0.9 ** 3)

    def test_done_drains_all_suffixes_with_partial_discounts(self):
        acc = NStepAccumulator(3, gamma=0.5)
        acc.push("s0", 0, 1.0, "s1", False)
        out = acc.push("s1", 0, 2.0, "s2", True)
        assert len(out) == 2
        (o0, _a, R0, _on, d0, g0), (o1, _a1, R1, _on1, d1, g1) = out
        assert o0 == "s0" and R0 == pytest.approx(1.0 + 0.5 * 2.0) and d0 is True
        assert g0 == pytest.approx(0.25)
        assert o1 == "s1" and R1 == pytest.approx(2.0) and d1 is True
        assert g1 == pytest.approx(0.5)

    def test_flush_bootstraps_through_truncation(self):
        acc = NStepAccumulator(3, gamma=0.5)
        acc.push("s0", 0, 1.0, "s1", False)
        out = acc.flush()
        assert len(out) == 1
        assert out[0][4] is False  # done=False: truncation bootstrapea


class TestQuantileLoss:
    def test_minimizer_is_the_quantile(self):
        # con target ~ muestras de una distribucion, el argmin de la perdida
        # QR para tau=0.5 es la MEDIANA. Optimizamos un cuantil libre y
        # verificamos que aterrice ahi.
        torch.manual_seed(0)
        samples = torch.tensor([[1.0, 2.0, 3.0, 4.0, 100.0]])  # mediana 3
        pred = torch.nn.Parameter(torch.zeros(1, 1))
        taus = torch.tensor([0.5])
        opt = torch.optim.SGD([pred], lr=0.5)
        for _ in range(3000):
            opt.zero_grad()
            quantile_huber_loss(pred, samples, taus).sum().backward()
            opt.step()
        assert 2.0 < float(pred) < 4.5  # mediana-ish, NO la media (22)

    def test_asymmetry_orders_quantiles(self):
        torch.manual_seed(0)
        samples = torch.randn(1, 256)
        pred = torch.nn.Parameter(torch.zeros(1, 3))
        taus = torch.tensor([0.1, 0.5, 0.9])
        opt = torch.optim.Adam([pred], lr=0.05)
        for _ in range(2000):
            opt.zero_grad()
            quantile_huber_loss(pred, samples, taus).sum().backward()
            opt.step()
        vals = pred.detach().numpy().ravel()
        assert vals[0] < vals[1] < vals[2]


class TestNetAndTrainStep:
    def test_dueling_shapes_and_identifiability(self):
        net = QRDuelingNet(12, n_actions=5, n_quantiles=7, hidden=16)
        out = net(torch.zeros(3, 12))
        assert out.shape == (3, 5, 7)
        # dueling: la media de advantage por accion se resta -> sumar una
        # constante al advantage no cambia la salida (identifiabilidad)
        with torch.no_grad():
            net.advantage.bias.add_(3.0)
            out2 = net(torch.zeros(3, 12))
        assert torch.allclose(out, out2, atol=1e-5)

    def test_train_step_reduces_loss_on_fixed_batch(self):
        torch.manual_seed(1)
        net = QRDuelingNet(8, n_actions=4, n_quantiles=9, hidden=32)
        tgt = QRDuelingNet(8, n_actions=4, n_quantiles=9, hidden=32)
        tgt.load_state_dict(net.state_dict())
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        taus = make_taus(9, torch.device("cpu"))
        rng = np.random.default_rng(0)
        batch = [(rng.random(8).astype(np.float32), int(rng.integers(4)),
                  np.float32(rng.random()), rng.random(8).astype(np.float32),
                  bool(rng.random() < 0.2), np.float32(0.97 ** 3))
                 for _ in range(32)]
        w = np.ones(32, dtype=np.float32)
        first, _ = train_step(net, tgt, batch, w, taus, 0.99, opt,
                              torch.device("cpu"))
        for _ in range(60):
            last, td = train_step(net, tgt, batch, w, taus, 0.99, opt,
                                  torch.device("cpu"))
        assert last < first
        assert len(td) == 32

    def test_epsilon_schedule(self):
        assert linear_epsilon(0) == pytest.approx(1.0)
        assert linear_epsilon(100_000, decay_steps=200_000) == pytest.approx(0.525)
        assert linear_epsilon(10 ** 9) == pytest.approx(0.05)


class TestActionConvention:
    def test_flat_wrapper_matches_es_divmod(self):
        # action() no toca estado de instancia: __new__ basta y evita
        # construir un env real solo para probar el mapeo
        from envs.discrete_sf2 import FlatDiscreteActions

        w = FlatDiscreteActions.__new__(FlatDiscreteActions)
        for a in range(N_ACTIONS):
            move, attack = divmod(a, 7)
            got = FlatDiscreteActions.action(w, a)
            assert got[0] == move and got[1] == attack
            assert 0 <= move <= 8 and 0 <= attack <= 6
