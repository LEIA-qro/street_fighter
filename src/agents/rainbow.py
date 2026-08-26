# rainbow.py -- nucleo puro del track Rainbow-DQN (variante QR), sin I/O.
#
# "Rainbow-lite (QR)": Double DQN + Dueling + PER proporcional + n-step +
# distribucional por QUANTILE REGRESSION (Dabney et al. 2017) en lugar del
# C51 canonico. La sustitucion es deliberada: C51 requiere fijar el soporte
# [v_min, v_max] de los retornos por adelantado (los famosos [-10, 10] de
# Atari) y un soporte mal calibrado degrada en silencio; QR aprende las
# posiciones de los cuantiles y no pide conocer la escala de reward de este
# juego. Noisy nets quedan fuera (epsilon-greedy con schedule lineal): una
# pieza menos que tunear en la primera pasada.
#
# Todo lo de este modulo es puro y testeable sin emulador: redes (torch),
# sum-tree/PER, acumulador n-step y el paso de entrenamiento. El cableado a
# RetroSF2Env vive en envs/discrete_sf2.py y tools/train_rainbow.py.
#
# Convencion de acciones COMPARTIDA con el ES (es/policy.py): accion plana
# a en [0, 63) -> (move, attack) = divmod(a, 7). Misma convencion en el
# wrapper y en el brazo rainbow del banco -- los tres tracks se examinan
# con exactamente el mismo mapa de acciones.

import math
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

N_ACTIONS = 63  # 9 direcciones x 7 botones, aplanado (divmod(a, 7))


# ---------------------------------------------------------------------------
# Red: MLP dueling con salida de cuantiles [acciones x cuantiles]
# ---------------------------------------------------------------------------

class QRDuelingNet(nn.Module):
    """MLP dueling: value (1 x Q cuantiles) + advantage (A x Q cuantiles).

    La entrada es la observacion YA featurizada (92 cruda o 212 con char
    one-hot -- decision del wrapper, no de la red). theta(s, a, q) =
    V(s, q) + A(s, a, q) - mean_a A(s, a, q), el promedio sobre acciones
    como en el paper de dueling.
    """

    def __init__(self, in_dim, n_actions=N_ACTIONS, n_quantiles=51, hidden=256):
        super().__init__()
        self.n_actions = int(n_actions)
        self.n_quantiles = int(n_quantiles)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value = nn.Linear(hidden, self.n_quantiles)
        self.advantage = nn.Linear(hidden, self.n_actions * self.n_quantiles)

    def forward(self, obs):
        """obs (B, in_dim) -> cuantiles (B, n_actions, n_quantiles)."""
        h = self.trunk(obs)
        v = self.value(h).view(-1, 1, self.n_quantiles)
        a = self.advantage(h).view(-1, self.n_actions, self.n_quantiles)
        return v + a - a.mean(dim=1, keepdim=True)

    def q_values(self, obs):
        """(B, n_actions): media de cuantiles = Q esperado por accion."""
        return self.forward(obs).mean(dim=2)


# ---------------------------------------------------------------------------
# PER: sum-tree proporcional (Schaul et al. 2015)
# ---------------------------------------------------------------------------

class SumTree:
    """Arbol binario en arreglo plano: hojas = prioridades, padres = sumas.

    total() en la raiz; sample(u) baja de la raiz eligiendo el hijo cuyo
    rango acumulado contiene u. Operaciones O(log n), sin numpy fanciness
    para que el comportamiento sea obvio en los tests.
    """

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity, dtype=np.float64)

    def total(self):
        return float(self.tree[1])

    def set(self, idx, priority):
        """idx en [0, capacity); priority >= 0."""
        i = idx + self.capacity
        delta = float(priority) - self.tree[i]
        while i >= 1:
            self.tree[i] += delta
            i //= 2

    def get(self, idx):
        return float(self.tree[idx + self.capacity])

    def sample(self, u):
        """u en [0, total()) -> indice de hoja cuyo rango acumulado lo contiene."""
        i = 1
        while i < self.capacity:
            left = 2 * i
            if u < self.tree[left]:
                i = left
            else:
                u -= self.tree[left]
                i = left + 1
        return i - self.capacity


class PERBuffer:
    """Replay proporcional con importance sampling.

    Guarda transiciones n-step ya cocinadas: (obs, action, return_n, obs_n,
    done_n, gamma_n). gamma_n viaja por transicion porque un episodio que
    termina antes de los n pasos descuenta menos -- recalcularlo en el
    sample seria repetir la logica del acumulador.
    """

    def __init__(self, capacity, alpha=0.5, eps=1e-3):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.tree = SumTree(self.capacity)
        self.data = [None] * self.capacity
        self.next_idx = 0
        self.size = 0
        self.max_priority = 1.0

    def push(self, transition):
        idx = self.next_idx
        self.data[idx] = transition
        # prioridad maxima vista: toda transicion nueva se muestrea pronto
        self.tree.set(idx, self.max_priority ** self.alpha)
        self.next_idx = (idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta, rng=random):
        """-> (indices, transiciones, pesos IS normalizados a max=1).

        `rng` acepta cualquier objeto con .random() (random.Random seedeado
        para reproducibilidad; el default global existe solo por comodidad
        en tests).
        """
        total = self.tree.total()
        # estratificado: un u uniforme por segmento, variancia menor que
        # batch_size uniformes sobre [0, total)
        seg = total / batch_size
        idxs = [self.tree.sample(min((i + rng.random()) * seg, total * (1 - 1e-12)))
                for i in range(batch_size)]
        probs = np.array([max(self.tree.get(i), 1e-12) for i in idxs]) / total
        weights = (self.size * probs) ** (-float(beta))
        weights = weights / weights.max()
        return idxs, [self.data[i] for i in idxs], weights.astype(np.float32)

    def update_priorities(self, idxs, td_errors):
        for idx, err in zip(idxs, np.abs(np.asarray(td_errors, dtype=np.float64))):
            p = float(err) + self.eps
            self.max_priority = max(self.max_priority, p)
            self.tree.set(int(idx), p ** self.alpha)


class NStepAccumulator:
    """Convierte transiciones 1-step en n-step por env. Uno por env paralelo.

    push() devuelve la transicion n-step lista cuando la ventana se llena, y
    flush() drena lo que quede al terminar el episodio (con el descuento
    parcial correcto). Retornos: (obs0, a0, R, obs_n, done, gamma_eff) donde
    R = sum_{k<m} gamma^k r_k y gamma_eff = gamma^m (m <= n pasos reales).
    """

    def __init__(self, n, gamma):
        self.n = int(n)
        self.gamma = float(gamma)
        self.window = deque()

    def _cook(self, done_flag):
        obs0, a0, _r, _o, _d = self.window[0]
        R, g = 0.0, 1.0
        for (_s, _a, r, _sn, _dn) in self.window:
            R += g * float(r)
            g *= self.gamma
        obs_n, done_n = self.window[-1][3], done_flag
        return (obs0, a0, np.float32(R), obs_n, bool(done_n), np.float32(g))

    def push(self, obs, action, reward, next_obs, done):
        self.window.append((obs, int(action), float(reward), next_obs, bool(done)))
        out = []
        if done:
            # drenar TODO: cada sufijo de la ventana es una transicion valida
            while self.window:
                out.append(self._cook(True))
                self.window.popleft()
        elif len(self.window) == self.n:
            out.append(self._cook(False))
            self.window.popleft()
        return out

    def flush(self):
        """Para truncation (fin de episodio SIN done de bootstrap-corte)."""
        out = []
        while self.window:
            out.append(self._cook(False))
            self.window.popleft()
        return out


# ---------------------------------------------------------------------------
# Perdida de quantile regression + paso de entrenamiento Double-DQN
# ---------------------------------------------------------------------------

def quantile_huber_loss(pred, target, taus, kappa=1.0):
    """Perdida QR (Dabney 2017 eq. 10) por elemento del batch.

    pred (B, Q): cuantiles de Q(s, a) elegida. target (B, Q'): cuantiles
    objetivo (sin gradiente). taus (Q,): puntos medios de los cuantiles de
    pred. Devuelve (B,) para que el caller aplique pesos IS.
    """
    # pares (B, Q, Q'): residuo de cada cuantil predicho contra cada target
    u = target.unsqueeze(1) - pred.unsqueeze(2)
    abs_u = u.abs()
    huber = torch.where(abs_u <= kappa, 0.5 * u ** 2, kappa * (abs_u - 0.5 * kappa))
    # |tau - 1{u<0}|: asimetria que hace que cada salida aprenda SU cuantil.
    # La division entre kappa es parte de la eq. 10 del paper (rho = L_k/k);
    # sin ella la perdida queda escalada por kappa (inocuo en kappa=1, y todo
    # call site usa 1, pero canonico es canonico).
    weight = (taus.view(1, -1, 1) - (u.detach() < 0).float()).abs()
    return (weight * huber / kappa).mean(dim=2).sum(dim=1)


def train_step(online, target, batch, weights, taus, gamma_unused, optimizer,
               device, kappa=1.0, max_grad_norm=10.0):
    """Un paso Double-DQN sobre transiciones n-step. -> (loss, td_errors).

    Doble: la accion del bootstrap la elige la red ONLINE (argmax de su Q
    medio), la evalua la red TARGET. El descuento por transicion (gamma_eff)
    viene del acumulador n-step, por eso gamma_unused no se usa aqui.
    """
    obs = torch.as_tensor(np.stack([t[0] for t in batch]), dtype=torch.float32,
                          device=device)
    actions = torch.as_tensor([t[1] for t in batch], dtype=torch.int64, device=device)
    returns = torch.as_tensor([t[2] for t in batch], dtype=torch.float32, device=device)
    next_obs = torch.as_tensor(np.stack([t[3] for t in batch]), dtype=torch.float32,
                               device=device)
    dones = torch.as_tensor([float(t[4]) for t in batch], dtype=torch.float32,
                            device=device)
    gammas = torch.as_tensor([t[5] for t in batch], dtype=torch.float32, device=device)
    w = torch.as_tensor(weights, dtype=torch.float32, device=device)

    with torch.no_grad():
        next_online_q = online.q_values(next_obs)          # (B, A) para elegir
        next_actions = next_online_q.argmax(dim=1)         # Double: online elige
        next_quant = target(next_obs)                      # target evalua
        next_quant = next_quant[torch.arange(len(batch)), next_actions]  # (B, Q)
        target_quant = returns.unsqueeze(1) + \
            (1.0 - dones).unsqueeze(1) * gammas.unsqueeze(1) * next_quant

    pred_quant = online(obs)[torch.arange(len(batch)), actions]  # (B, Q)
    per_sample = quantile_huber_loss(pred_quant, target_quant, taus, kappa)
    loss = (w * per_sample).mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), max_grad_norm)
    optimizer.step()

    # prioridad = |TD| sobre las MEDIAS de los cuantiles (escalar por muestra)
    td = (target_quant.mean(dim=1) - pred_quant.detach().mean(dim=1))
    return float(loss.item()), td.cpu().numpy()


def make_taus(n_quantiles, device):
    """Puntos medios: tau_i = (2i + 1) / 2Q, i en [0, Q)."""
    return (torch.arange(n_quantiles, dtype=torch.float32, device=device) * 2 + 1) \
        / (2.0 * n_quantiles)


def linear_epsilon(step, start=1.0, end=0.05, decay_steps=200_000):
    """Schedule lineal clasico de DQN."""
    if step >= decay_steps:
        return float(end)
    frac = step / float(decay_steps)
    return float(start + frac * (end - start))
