# apex.py -- nucleo del Rainbow DISTRIBUIDO (arquitectura Ape-X, Horgan 2018)
# sobre la receta QR de agents/rainbow.py.
#
#   learner (1 maquina, GPU): replay PER central + red + loop de gradiente,
#     sirve /weights (con la config del run adentro) e ingiere /transitions.
#   actores (N maquinas): emuladores + copia local de la red + epsilon propio
#     por proceso (la escalera de Ape-X), cocinan n-step LOCALMENTE (el mismo
#     NStepAccumulator) y POSTean lotes comprimidos.
#
# Decisiones de wire (los patrones del ES, reciclados):
#   - La CONFIG del run (gamma, n_step, onehot, quantiles, hidden, in_dim)
#     viaja DENTRO de /weights: el learner es la unica fuente de verdad y un
#     actor viejo con arquitectura incompatible truena ruidoso al cargar
#     (mismo espiritu que el guard de theta del worker ES).
#   - Las obs viajan CRUDAS (92 floats) en float16: todos los canales v4 son
#     enteros < 2048, donde fp16 es EXACTO -- sin perdida y ~4.6x menos red
#     que mandar el one-hot expandido en f32. El learner expande el one-hot
#     al muestrear (256 expansiones por paso de gradiente: microsegundos).
#   - Lotes via np.savez_compressed + base64 (el idioma de encode_theta).
#
# Todo lo de este modulo es puro/testeable: encoders, escalera de epsilon,
# y ApexLearner (ingesta -> buffer -> train_tick) sin HTTP ni emulador.

import base64
import io
import threading

import numpy as np
import torch

from agents.rainbow import PERBuffer, QRDuelingNet, make_taus, train_step
from es.policy import OBS_DIM, ONEHOT_OBS_DIM, expand_char_onehot

MAX_BATCH_B64_BYTES = 8_000_000  # tope de un POST /transitions decodificado


def apex_epsilon(rank, total, base=0.4, alpha=7.0):
    """La escalera de exploracion de Ape-X: eps_i = base^(1 + alpha*i/(N-1)).

    rank 0 explora salvaje (eps=base), el ultimo casi puro greedy. La
    DIVERSIDAD de experiencia en el buffer es un bonus algoritmico del
    distribuido, no solo velocidad.
    """
    if total <= 1:
        return float(base)
    return float(base ** (1.0 + alpha * rank / (total - 1)))


# ---------------------------------------------------------------------------
# Lotes de transiciones n-step: (obs92, action, R, next_obs92, done, gamma_eff)
# ---------------------------------------------------------------------------

FP16_CLIP = 2047.0  # fp16 es exacto para enteros hasta 2048; mas alla pierde
                    # exactitud y en 65520 se vuelve inf (NaN en la loss). Los
                    # canales v4 declarados topan en ~500, pero varios vienen
                    # de RAM >u2 sin clip aguas arriba: cinturon y tirantes.


def encode_transitions(transitions):
    """Lista de tuplas n-step (obs cruda 92) -> b64 de un npz comprimido."""
    obs = np.clip(np.stack([t[0] for t in transitions]),
                  -FP16_CLIP, FP16_CLIP).astype(np.float16)
    next_obs = np.clip(np.stack([t[3] for t in transitions]),
                       -FP16_CLIP, FP16_CLIP).astype(np.float16)
    buf = io.BytesIO()
    np.savez_compressed(
        buf, obs=obs, next_obs=next_obs,
        actions=np.array([t[1] for t in transitions], dtype=np.uint8),
        returns=np.array([t[2] for t in transitions], dtype=np.float32),
        dones=np.array([t[4] for t in transitions], dtype=np.uint8),
        gammas=np.array([t[5] for t in transitions], dtype=np.float32))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_transitions(b64):
    """b64 -> lista de tuplas con obs float16 (el buffer las guarda asi:
    mitad de RAM; la conversion a f32 + one-hot ocurre al muestrear)."""
    raw = base64.b64decode(b64)
    if len(raw) > MAX_BATCH_B64_BYTES:
        raise ValueError(f"transition batch too large: {len(raw)} bytes")
    arrays = np.load(io.BytesIO(raw))
    obs, next_obs = arrays["obs"], arrays["next_obs"]
    if obs.shape[1] != OBS_DIM:
        raise ValueError(f"obs dim {obs.shape[1]} != {OBS_DIM} (actor stale?)")
    return [(obs[i], int(arrays["actions"][i]), np.float32(arrays["returns"][i]),
             next_obs[i], bool(arrays["dones"][i]), np.float32(arrays["gammas"][i]))
            for i in range(obs.shape[0])]


# ---------------------------------------------------------------------------
# Pesos + config en un solo payload: la unica fuente de verdad del run
# ---------------------------------------------------------------------------

def encode_weights(net, version, config):
    buf = io.BytesIO()
    np.savez_compressed(buf, **{k: v.detach().cpu().numpy()
                                for k, v in net.state_dict().items()})
    return {"version": int(version), "config": dict(config),
            "npz_b64": base64.b64encode(buf.getvalue()).decode("ascii")}


def decode_weights(payload, net=None):
    """payload -> (version, config, state_dict de tensores CPU).

    Si `net` viene, se cargan los pesos ahi (load_state_dict estricto: una
    arquitectura que no cuadra truena ruidoso -- actor stale, no evaluacion
    silenciosa con red equivocada)."""
    raw = base64.b64decode(payload["npz_b64"])
    arrays = np.load(io.BytesIO(raw))
    state = {k: torch.from_numpy(np.array(arrays[k])) for k in arrays.files}
    if net is not None:
        net.load_state_dict(state)
    return int(payload["version"]), dict(payload.get("config", {})), state


# ---------------------------------------------------------------------------
# El learner: buffer central + red + un lock (el idioma del Coordinator)
# ---------------------------------------------------------------------------

class ApexLearner:
    """Nucleo del learner, sin HTTP: ingesta, pesos versionados, train_tick.

    El lock cubre el buffer y los contadores (los handlers HTTP y el loop de
    gradiente son hilos del mismo proceso). El train_tick muestrea BAJO el
    lock pero entrena FUERA (el forward/backward es lo caro y el buffer solo
    necesita consistencia en sample/update_priorities).
    """

    def __init__(self, hidden=256, quantiles=51, onehot=True, gamma=0.99,
                 n_step=3, buffer_capacity=500_000, lr=1e-4, per_alpha=0.5,
                 device="cpu", weights_every_grads=100, n_actions=63,
                 macros=False):
        self.onehot = bool(onehot)
        self.in_dim = ONEHOT_OBS_DIM if self.onehot else OBS_DIM
        self.n_actions = int(n_actions)
        self.device = torch.device(device)
        self.online = QRDuelingNet(self.in_dim, n_actions=self.n_actions,
                                   n_quantiles=quantiles,
                                   hidden=hidden).to(self.device)
        self.target = QRDuelingNet(self.in_dim, n_actions=self.n_actions,
                                   n_quantiles=quantiles,
                                   hidden=hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr,
                                          eps=1.5e-4)
        self.taus = make_taus(quantiles, self.device)
        self.buffer = PERBuffer(buffer_capacity, alpha=per_alpha)
        self.lock = threading.Lock()
        self.config = {"hidden": int(hidden), "quantiles": int(quantiles),
                       "onehot": self.onehot, "in_dim": self.in_dim,
                       "gamma": float(gamma), "n_step": int(n_step),
                       # los actores construyen su env y su exploracion de
                       # ESTO: el vocabulario es config del run, no flag local
                       "n_actions": self.n_actions, "macros": bool(macros)}
        self.weights_version = 0
        self.weights_every_grads = int(weights_every_grads)
        self._weights_payload = encode_weights(self.online, 0, self.config)
        self.grad_steps = 0
        self.transitions_in = 0
        self.actors = {}   # name -> {"last_seen", "transitions", "stats"...}
        self.ep_wins = 0
        self.ep_count = 0
        # ventana reciente: el termometro de tendencia SIN el lastre del
        # arranque aleatorio ni resets (deque de los ultimos 200 episodios)
        from collections import deque
        self.recent_eps = deque(maxlen=200)

    # -- ingesta (hilo HTTP) ------------------------------------------------
    def ingest(self, body, now):
        """POST /transitions -> ack. body: {actor, b64, stats?, episodes?}."""
        transitions = decode_transitions(body["b64"])
        with self.lock:
            for t in transitions:
                self.buffer.push(t)
            self.transitions_in += len(transitions)
            name = str(body.get("actor", "?"))
            rec = self.actors.setdefault(name, {"transitions": 0})
            rec["last_seen"] = now
            rec["transitions"] += len(transitions)
            stats = body.get("stats")
            if isinstance(stats, dict):
                rec.update({k: stats[k] for k in ("procs", "steps_per_s", "host")
                            if k in stats})
            eps = body.get("episodes")
            if isinstance(eps, dict):
                wins = int(eps.get("wins", 0) or 0)
                count = int(eps.get("count", 0) or 0)
                self.ep_wins += wins
                self.ep_count += count
                # el lote agrega episodios en bloque: se reparten como
                # wins unos y (count-wins) ceros -- el orden intra-lote
                # no importa para una media movil
                self.recent_eps.extend([1] * wins + [0] * (count - wins))
        return {"accepted": True, "buffer": self.buffer.size,
                "weights_version": self.weights_version}

    def weights_payload(self):
        with self.lock:
            return self._weights_payload

    # -- entrenamiento (hilo principal) -------------------------------------
    def _featurize(self, batch):
        if not self.onehot:
            return [(t[0].astype(np.float32), t[1], t[2],
                     t[3].astype(np.float32), t[4], t[5]) for t in batch]
        return [(expand_char_onehot(t[0]), t[1], t[2],
                 expand_char_onehot(t[3]), t[4], t[5]) for t in batch]

    def train_tick(self, batch_size, beta, rng):
        """Un paso de gradiente. -> (loss, td) o None si el buffer no alcanza."""
        with self.lock:
            if self.buffer.size < batch_size:
                return None
            idxs, batch, weights = self.buffer.sample(batch_size, beta, rng=rng)
        feats = self._featurize(batch)
        loss, td = train_step(self.online, self.target, feats, weights,
                              self.taus, None, self.optimizer, self.device)
        with self.lock:
            self.buffer.update_priorities(idxs, td)
            self.grad_steps += 1
            if self.grad_steps % self.weights_every_grads == 0:
                self.weights_version += 1
                self._weights_payload = encode_weights(
                    self.online, self.weights_version, self.config)
        return loss, td

    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def status(self, now):
        with self.lock:
            return {"grad_steps": self.grad_steps,
                    "weights_version": self.weights_version,
                    "buffer": self.buffer.size,
                    "transitions_in": self.transitions_in,
                    "win_rate_cum": round(self.ep_wins / max(self.ep_count, 1), 3),
                    "win_rate_recent200": round(
                        sum(self.recent_eps) / max(len(self.recent_eps), 1), 3),
                    "episodes": self.ep_count,
                    "actors": {n: dict(r, age=round(now - r.get("last_seen", 0), 1))
                               for n, r in self.actors.items()}}
