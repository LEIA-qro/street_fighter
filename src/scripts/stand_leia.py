# stand_leia.py -- HUMANO vs el campeon DQN, en BizHawk (modo stand LEIA).
#
#   .venv\Scripts\python.exe src\scripts\stand_leia.py --ckpt benchmarks\apex_milestones\apex_v1592_benchmarked.pt
#
# P1 = la IA (Ryu, el campeon Ape-X con macros). P2 puede ser humano, CPU del
# juego, otro Ape-X o un modelo clasico SB3 (PPO/SAC/DQN). En modo humano este
# script manda ".........." para P2: el Lua no inyecta y el pad pasa directo. Corre sobre
# lua/v2.0/stand_env_client.lua, que manda las 25 variables de RAM crudas del
# data.json; la observacion se arma aqui con el MISMO assemble_v4_frame del
# entrenamiento y los macros se reproducen con la MISMA semantica que
# MacroActionWrapper (cola de pasos por macro, espejo por lado, SIN sticky --
# el campeon entreno con el sticky apagado).
#
# Cadencia del socket (revisada adversarialmente 2026-08-27): el Lua manda su
# payload al tope del loop y BizHawk lo enmarca como "<len> 0 v1,...". Al
# conectar hay UN payload inicial pre-reset en vuelo, que el cold-start drena.
# Los rematches ya estan sincronizados y consumen un solo payload post-reset;
# un segundo receive sin comando bloquearia Lua hasta su dead-man switch. Con
# esa fase, cada receive del combate trae el estado generado por la accion
# recien mandada (cero lag extra).
#
# Rondas: KO por el signo del word de HP (hp_to_signed, la misma
# discriminacion del entrenamiento; el HP del frame va con piso en 0 y no
# distingue muerte de garbage de transicion) y time-over por round_timer == 0
# decidido por HP restante. Deliberadamente NO se usa el RoundTracker del
# entrenamiento: el stand necesita un marcador de feria, no contabilidad de
# curriculum -- si algun borde muerde, ahi esta envs/reward.RoundTracker.
#
# La paridad de la tuberia (parseo -> v4 -> onehot -> macro -> 10 bits) esta
# cubierta offline en code_testing/pytest/test_stand_leia.py; el end-to-end
# con emulador se prueba corriendo esto en la maquina del stand.

import argparse
import atexit
import hashlib
import json
import os
import random
import re
import signal
import sys
import time
import traceback
from collections import deque
from datetime import datetime, timezone

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np  # noqa: E402

from envs.action_macros import N_ACTIONS, decode  # noqa: E402
from envs.retro_env import (  # noqa: E402
    NUM_FRAMES, V4_FRAME_DIM, RamTrack, assemble_v4_frame,
    discrete_to_project_bits,
)
from envs.reward import hp_to_signed  # noqa: E402
from es.policy import expand_char_onehot  # noqa: E402

# Orden EXACTO del payload de stand_env_client.lua (generado de data.json).
PAYLOAD_KEYS = (
    "p1_hp", "p2_hp", "p1_x", "p2_x", "p1_y", "p2_y",
    "p1_state_word", "p2_state_word",
    "p1_proj_x", "p2_proj_x", "p1_char", "p2_char", "rel_dist",
    "p1_btn", "p2_btn", "p1_air_raw", "p2_air_raw", "rel_y_dist",
    "p1_chest", "p1_head", "p2_chest", "p2_head",
    "matches_won", "enemy_matches_won", "round_timer",
)

OPPONENTS = ("RYU", "EHONDA", "BLANKA", "GUILE", "KEN", "CHUNLI",
             "ZANGIEF", "DHALSIM", "MBISON", "SAGAT", "BALROG", "VEGA")

HUMAN_PASSTHROUGH = ".........."   # 10 puntos: el Lua no toca ese pad

# Campeon congelado despues de completar el benchmark de la escalera 1-8.
# Se usa un nombre versionado para que una run activa no cambie silenciosamente
# el modelo que recibe otra maquina mediante git pull.
DEFAULT_CHECKPOINT = os.path.join(
    "benchmarks", "apex_milestones", "apex_v1592_benchmarked.pt")


def checkpoint_provenance(checkpoint_path: str) -> dict:
    """Identidad reproducible del modelo y resultados publicados al lado."""
    result = {"checkpoint_path": os.path.normpath(checkpoint_path)}
    try:
        digest = hashlib.sha256()
        with open(checkpoint_path, "rb") as checkpoint_file:
            for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                digest.update(chunk)
        result["checkpoint_sha256"] = digest.hexdigest()
    except OSError as exc:
        result["checkpoint_sha256"] = None
        result["checkpoint_identity_error"] = str(exc)

    sidecar_path = checkpoint_path + ".json"
    try:
        with open(sidecar_path, "r", encoding="utf-8") as sidecar_file:
            sidecar = json.load(sidecar_file)
        if isinstance(sidecar, dict):
            result["checkpoint_benchmark"] = sidecar
    except (OSError, ValueError):
        pass
    return result


class MatchSessionLog:
    """JSONL durable: cada evento queda flush+fsync antes de seguir jugando."""

    def __init__(self, log_dir: str, session_fields=None):
        self._file = None
        self._closed = False
        self._warning_printed = False
        self._started_monotonic = time.monotonic()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.session_id = f"{stamp}_{os.getpid()}"
        self.path = os.path.join(
            log_dir, f"apex_viewer_{self.session_id}.jsonl")
        try:
            os.makedirs(log_dir, exist_ok=True)
            self._file = open(
                self.path, "a", encoding="utf-8", buffering=1, newline="\n")
        except OSError as exc:
            self._warn_once(f"no se pudo crear el log persistente: {exc}")
            self.path = None
            return
        self.write("session_start", **(session_fields or {}))

    @staticmethod
    def _utc_now():
        return datetime.now(timezone.utc).isoformat(
            timespec="milliseconds").replace("+00:00", "Z")

    def _warn_once(self, message):
        if not self._warning_printed:
            print(f"[viewer] ADVERTENCIA DE LOG: {message}", flush=True)
            self._warning_printed = True

    def write(self, event: str, **fields) -> bool:
        if self._file is None or self._closed:
            return False
        record = {
            "schema_version": 1,
            "timestamp_utc": self._utc_now(),
            "session_id": self.session_id,
            "event": event,
            **fields,
        }
        try:
            self._file.write(json.dumps(
                record, ensure_ascii=False, separators=(",", ":")) + "\n")
            self._file.flush()
            os.fsync(self._file.fileno())
            return True
        except (OSError, TypeError, ValueError) as exc:
            self._warn_once(f"no se pudo guardar el evento {event}: {exc}")
            return False

    def close(self, reason: str, **fields):
        if self._closed:
            return
        self.write(
            "session_end",
            reason=reason,
            duration_seconds=round(
                time.monotonic() - self._started_monotonic, 3),
            **fields,
        )
        self._closed = True
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass


def opponent_from_state(state_path: str) -> str:
    name = os.path.basename(state_path)
    match = re.fullmatch(
        r"RYU_(.+)_R1_(?:PvP|lvl[1-7]|HARD)\.State", name,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()
    return name


def ram_for_player(ram: dict, player: int) -> dict:
    """Devuelve el RAM con el jugador elegido ocupando el contrato P1.

    QR-DQN fue entrenado desde la perspectiva del luchador controlado. Para
    usarlo como P2 hay que intercambiar todos los campos por lado y adaptar
    las dos lecturas de airborne, cuyos valores crudos usan convenciones
    distintas en las direcciones P1 y P2.
    """
    if player == 1:
        return dict(ram)
    if player != 2:
        raise ValueError("player debe ser 1 o 2")

    swapped = dict(ram)
    for stem in (
        "hp", "x", "y", "state_word", "proj_x", "char", "btn",
        "chest", "head",
    ):
        swapped[f"p1_{stem}"], swapped[f"p2_{stem}"] = (
            ram[f"p2_{stem}"], ram[f"p1_{stem}"])

    # P1: 0=floor, distinto de 0=air. P2: 14=floor, 13=air.
    swapped["p1_air_raw"] = 257 if int(ram["p2_air_raw"]) == 13 else 0
    swapped["p2_air_raw"] = 13 if int(ram["p1_air_raw"]) != 0 else 14
    swapped["matches_won"] = ram["enemy_matches_won"]
    swapped["enemy_matches_won"] = ram["matches_won"]
    return swapped


def parse_payload(raw: str) -> dict:
    """'<len> 0 v1,v2,...' -> dict con las claves de data.json (crudas).

    BizHawk enmarca cada envio del Lua con su longitud ("<len> "), y el Lua
    ya manda el rank "0 ": el CSV no contiene espacios, asi que el ultimo
    token es siempre el CSV (mismo idioma que base_env._parse_payload).
    """
    body = raw.strip().split(" ")[-1]
    values = body.split(",")
    if len(values) != len(PAYLOAD_KEYS):
        raise ValueError(f"payload de {len(values)} campos; esperaba "
                         f"{len(PAYLOAD_KEYS)} (¿Lua viejo? usa stand_env_client.lua)")
    return {k: int(v) for k, v in zip(PAYLOAD_KEYS, values)}


def sb3_payload_from_ram(ram: dict) -> str:
    """Adapta el RAM del stand al payload legacy de 13 campos de SB3.

    El payload expandido del stand tiene 25 campos, pero no comparte el layout
    opcional de 24/26/27 campos de ``StreetFighterBaseEnv``. Los primeros 13 sí
    son idénticos y contienen todo lo que consumen las observaciones v2/v3.
    """
    return "0 " + ",".join(str(ram[key]) for key in PAYLOAD_KEYS[:13])


def resolve_round(p1_ko: bool, p2_ko: bool, timer: int,
                  p1_hp: float, p2_hp: float):
    """-> (winner, msg) al resolverse el round, o None si sigue vivo.

    winner: 'ia' | 'retador' | 'empate'. KO manda; sin KO, round_timer == 0
    es time-over y decide el HP restante del frame.
    """
    if p1_ko or p2_ko:
        if p2_ko and not p1_ko:
            return "ia", "GANA LA IA"
        if p1_ko and not p2_ko:
            return "retador", "GANA EL RETADOR!"
        return "empate", "DOBLE K.O."
    if timer == 0:
        if p1_hp > p2_hp:
            return "ia", "GANA LA IA (por tiempo)"
        if p2_hp > p1_hp:
            return "retador", "GANA EL RETADOR! (por tiempo)"
        return "empate", "EMPATE (por tiempo)"
    return None


class MacroPlayer:
    """La semantica de MacroActionWrapper sin env: cola de (dir, boton).

    El espejo se decide UNA vez por macro, del rel_x del frame mas reciente
    (indice 2), y se refresca al terminar cada macro y en cada reset --
    identico a _update_facing del wrapper.
    """

    def __init__(self, frame_size: int = V4_FRAME_DIM, rel_x_index: int = 2):
        self.frame_size = frame_size
        self.rel_x_index = rel_x_index
        self.facing_right = True
        self.queue = []

    def update_facing(self, stacked_obs: np.ndarray) -> None:
        n_frames = max(1, len(stacked_obs) // self.frame_size)
        latest = (n_frames - 1) * self.frame_size
        self.facing_right = float(stacked_obs[latest + self.rel_x_index]) >= 0.0

    def reset(self, stacked_obs: np.ndarray) -> None:
        self.queue = []
        self.update_facing(stacked_obs)

    def next_step(self, stacked_obs: np.ndarray, choose_action) -> tuple:
        """-> (direction, button) del paso actual; consulta al modelo solo
        cuando la cola esta vacia (una decision por macro, como el wrapper)."""
        if not self.queue:
            self.update_facing(stacked_obs)
            action = int(choose_action(stacked_obs))
            self.queue = list(decode(action, self.facing_right))
        return self.queue.pop(0)


def bits_command(direction: int, button: int) -> str:
    return "".join(str(b) for b in discrete_to_project_bits((direction, button)))


def load_champion(ckpt_path: str, device: str):
    import torch
    torch.set_num_threads(1)
    from agents.rainbow import QRDuelingNet

    # Los checkpoints del proyecto solo contienen tensores + metadata simple;
    # weights_only evita ejecutar pickles arbitrarios al seleccionarlos desde
    # el dashboard.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    meta = ckpt["meta"]
    if not meta.get("macros", False) or int(meta.get("n_actions", 0)) != N_ACTIONS:
        raise SystemExit(f"[viewer] el checkpoint no es del campeon con macros "
                         f"(meta: {meta}) -- este driver reproduce macros")
    net = QRDuelingNet(meta["in_dim"], n_actions=N_ACTIONS,
                       n_quantiles=meta["quantiles"], hidden=meta["hidden"])
    net.load_state_dict(ckpt["state_dict"])
    net.to(device).eval()
    onehot = bool(meta.get("onehot", True))

    def choose_action(stacked_obs: np.ndarray) -> int:
        feats = expand_char_onehot(stacked_obs) if onehot else stacked_obs
        with torch.no_grad():
            q = net.q_values(torch.as_tensor(
                feats, dtype=torch.float32, device=device).unsqueeze(0))
        return int(q.argmax(dim=1).item())

    return choose_action, meta


def sb3_action_to_command(action, algo: str, env_version: str) -> str:
    """Convierte la salida batch de SB3 al protocolo de 10 bits de BizHawk."""
    algo = str(algo).lower()
    env_version = str(env_version).lower()
    if algo == "sac":
        if env_version == "v3":
            from envs.sf2_v3 import discrete_to_binary
            row = np.asarray(action)[0]
            discrete = np.array([
                np.argmax(row[:9]), np.argmax(row[9:]),
            ])
            return discrete_to_binary(discrete)
        return "".join(str(int(bit)) for bit in (
            np.asarray(action)[0] > 0.0).astype(np.int8))
    if algo == "dqn":
        value = int(np.asarray(action).reshape(-1)[0])
        if env_version == "v3":
            from envs.sf2_v3 import discrete_to_binary
            return discrete_to_binary(np.array([value // 7, value % 7]))
        return format(value, "010b")
    if env_version == "v3":
        from envs.sf2_v3 import discrete_to_binary
        return discrete_to_binary(np.asarray(action)[0])
    return "".join(str(int(bit)) for bit in np.asarray(action)[0])


class SB3PerspectiveParser:
    """Aísla el estado de velocidad de la perspectiva P2 del parser v2."""

    def __init__(self, env):
        self.env = env
        self.prev_p1_x = 0
        self.prev_p2_x = 0

    def parse(self, raw_payload: str, is_reset: bool = False) -> np.ndarray:
        self.env.player = 2
        self.env.prev_p1_x = self.prev_p1_x
        self.env.prev_p2_x = self.prev_p2_x
        observation = self.env._parse_payload(
            raw_payload, is_reset=is_reset)
        self.prev_p1_x = self.env.prev_p1_x
        self.prev_p2_x = self.env.prev_p2_x
        return observation


class SB3Opponent:
    """Frame stack + normalización + inferencia para un rival SB3 en P2."""

    def __init__(self, parser, normalizer, model, algo: str,
                 env_version: str, n_frames: int):
        self.parser = parser
        self.normalizer = normalizer
        self.model = model
        self.algo = algo
        self.env_version = env_version
        self.frames = deque(maxlen=n_frames)

    def reset(self, raw_payload: str) -> None:
        observation = self.parser.parse(raw_payload, is_reset=True)
        self.frames.clear()
        for _ in range(self.frames.maxlen):
            self.frames.append(observation.copy())

    def observe(self, raw_payload: str) -> None:
        self.frames.append(self.parser.parse(raw_payload, is_reset=False))

    def command(self) -> str:
        stacked = np.concatenate(self.frames)[np.newaxis, :]
        normalized = self.normalizer.normalize_obs(
            stacked.copy(), update=False)
        action, _state = self.model.predict(normalized, deterministic=False)
        return sb3_action_to_command(action, self.algo, self.env_version)


def load_sb3_opponent(env, algo: str, env_version: str, model_path: str,
                      vecnorm_path: str, device: str) -> SB3Opponent:
    """Carga un PPO/SAC/DQN sin abrir un segundo emulador/socket."""
    import gymnasium as gym
    import torch
    from gymnasium import spaces
    from stable_baselines3 import DQN, PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv

    from core import config
    from core.selective_norm import SelectiveVecNormalize
    from envs.base_env import TOTAL_OBS_DIM

    torch.set_num_threads(1)
    model_classes = {"ppo": PPO, "sac": SAC, "dqn": DQN}
    algo = str(algo).lower()
    env_version = str(env_version).lower()
    if algo not in model_classes:
        raise ValueError(f"algoritmo SB3 no válido: {algo}")
    if env_version not in ("v2", "v3"):
        raise ValueError(f"environment SB3 no válido: {env_version}")

    class _MockEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(TOTAL_OBS_DIM * config.NUM_FRAMES,),
                dtype=np.float32,
            )
            self.action_space = (
                spaces.MultiDiscrete([9, 7]) if env_version == "v3"
                else spaces.MultiBinary(config.ACTION_DIM)
            )

        def reset(self, **_kwargs):
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}

        def step(self, _action):
            return (
                np.zeros(self.observation_space.shape, dtype=np.float32),
                0.0, False, False, {},
            )

    dummy_env = DummyVecEnv([_MockEnv])
    normalizer = SelectiveVecNormalize.load(vecnorm_path, dummy_env)
    normalizer.training = False
    normalizer.norm_reward = False
    custom_objects = {"learning_rate": 0.0, "clip_range": 0.0}
    if algo in ("dqn", "sac"):
        custom_objects["buffer_size"] = 1
    model = model_classes[algo].load(
        model_path, device=device, custom_objects=custom_objects)
    return SB3Opponent(
        SB3PerspectiveParser(env), normalizer, model, algo, env_version,
        config.NUM_FRAMES,
    )


def pick_state(states_dir: str, opponent: str, opponent_type: str = "human",
               cpu_level: int = 1) -> str:
    opponent_type = str(opponent_type).lower()
    if opponent_type not in ("human", "cpu", "model", "sb3"):
        raise SystemExit(f"[viewer] tipo de rival inválido: {opponent_type}")

    if opponent_type == "cpu":
        cpu_level = int(cpu_level)
        if cpu_level not in range(1, 9):
            raise SystemExit("[viewer] el nivel de CPU debe estar entre 1 y 8")
        state_suffix = (
            "_R1_HARD.State" if cpu_level == 8
            else f"_R1_lvl{cpu_level}.State"
        )
    else:  # humano o cualquier modelo P2 usan un savestate PvP
        state_suffix = "_R1_PvP.State"

    if opponent == "RANDOM":
        available = [
            name for name in OPPONENTS
            if os.path.isfile(os.path.join(
                states_dir, f"RYU_{name}{state_suffix}"))
        ]
        if not available:
            raise SystemExit(
                f"[viewer] no hay savestates para rival {opponent_type} "
                f"nivel {cpu_level if opponent_type == 'cpu' else 'PvP'} "
                f"en {states_dir}")
        opponent = random.choice(available)
    path = os.path.join(states_dir, f"RYU_{opponent}{state_suffix}")
    if not os.path.exists(path):
        raise SystemExit(f"[viewer] no existe {path}")
    return path


def request_round_reset(env, state_path: str, p1_wins: int, p2_wins: int,
                        discard_pending_payload: bool = False,
                        payload_parser=parse_payload):
    """Carga un savestate y devuelve el primer RAM válido posterior al reset.

    Al conectar por primera vez, Lua ya dejó un payload inicial en el socket;
    ese único frame sí debe drenarse. Durante un rematch, en cambio, el loop
    acabó de consumir el payload anterior: hacer dos receive() seguidos deja a
    Lua esperando un comando y activa su dead-man switch, que cierra BizHawk.
    """
    env.send_command(f"RESET {state_path}|{p1_wins}|{p2_wins}\n")
    if discard_pending_payload:
        env.receive_payload()
    raw = env.receive_payload()
    if not raw:
        raise RuntimeError(
            "[viewer] el socket se cerró durante RESET; no hay estado "
            "post-reset")
    try:
        return payload_parser(raw)
    except ValueError as exc:
        # Lua ya espera el siguiente comando. Reintentar receive() aquí
        # recrearía el deadlock de 120 s; abortamos con diagnóstico explícito.
        raise RuntimeError(
            "[viewer] payload post-reset inválido; sesión abortada para no "
            "bloquear BizHawk") from exc


def main():
    ap = argparse.ArgumentParser(description="Viewer humano vs el campeon DQN")
    ap.add_argument("--ckpt", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--opponent", default="RANDOM",
                    choices=("RANDOM",) + OPPONENTS,
                    help="personaje DEL RETADOR (el rival de la IA; la IA "
                         "siempre es Ryu); RANDOM rota por round")
    ap.add_argument("--opponent-type", default="human",
                    choices=("human", "cpu", "model", "sb3"),
                    help="humano, CPU integrada, segundo Ape-X o PPO/SAC/DQN")
    ap.add_argument("--cpu-level", type=int, default=1, choices=range(1, 9),
                    help="nivel exacto de la CPU integrada (1-8)")
    ap.add_argument("--p2-ckpt", default=None,
                    help="checkpoint Ape-X de Player 2 en modo model")
    ap.add_argument("--p2-device", default=None,
                    help="device de P2; por defecto usa el mismo que P1")
    ap.add_argument("--p2-algo", choices=("ppo", "sac", "dqn"),
                    help="algoritmo del modelo clásico de P2")
    ap.add_argument("--p2-env", choices=("v2", "v3"), default="v2",
                    help="versión del environment usada por el modelo P2")
    ap.add_argument("--p2-model-zip",
                    help="modelo SB3 .zip de P2, relativo al proyecto")
    ap.add_argument("--p2-model-pkl",
                    help="normalización .pkl de P2, relativa al proyecto")
    ap.add_argument("--rematch-delay", type=float, default=4.0,
                    help="segundos de pantalla de KO antes del rematch")
    ap.add_argument("--infinite-match", action="store_true",
                    help="hace RESET automático tras cada round")
    ap.add_argument("--device", default="cpu",
                    help="cpu basta y sobra (una inferencia cada 1-3 pasos)")
    args = ap.parse_args()

    # Imports tardios del rig Windows (core.config es fatal fuera de Windows).
    from core import config
    from core.env_tools import failsafe_env
    from envs.sf2_v2 import StreetFighterEnvV2

    # Etiquetas ASCII a proposito: viajan por generated_config.lua a
    # gui.text, cuyo manejo de UTF-8 no esta garantizado en BizHawk.
    config.P1_MODEL_NAME = "LEIA - IA"
    if args.opponent_type == "cpu":
        config.P2_MODEL_NAME = f"CPU NIVEL {args.cpu_level}"
    elif args.opponent_type == "model":
        config.P2_MODEL_NAME = "LEIA - IA P2"
    elif args.opponent_type == "sb3":
        model_name = os.path.basename(args.p2_model_zip or "MODELO")
        config.P2_MODEL_NAME = model_name.replace(".zip", "")
    else:
        config.P2_MODEL_NAME = "RETADOR (tu)"
    config.generate_lua_config()

    agent_state_path = os.path.join(config.PROJECT_ROOT, ".agent_state")
    env = None
    cleanup_done = False
    ckpt_path = os.path.join(config.PROJECT_ROOT, args.ckpt)
    checkpoint_info = checkpoint_provenance(ckpt_path)
    # El JSONL se comparte entre testers: guardar la ruta relativa seleccionada,
    # no C:\Users\... ni otro dato personal de la máquina anfitriona.
    checkpoint_info["checkpoint_path"] = os.path.normpath(args.ckpt)
    if args.opponent_type == "model" and not args.p2_ckpt:
        raise SystemExit("[viewer] modo model requiere --p2-ckpt")
    if args.opponent_type == "sb3" and not all((
            args.p2_algo, args.p2_model_zip, args.p2_model_pkl)):
        raise SystemExit(
            "[viewer] modo sb3 requiere --p2-algo, --p2-model-zip y "
            "--p2-model-pkl")
    p2_ckpt_path = (
        os.path.join(config.PROJECT_ROOT, args.p2_ckpt)
        if args.p2_ckpt else None
    )
    p2_checkpoint_info = {}
    if p2_ckpt_path:
        p2_checkpoint_info = {
            f"p2_{key}": value
            for key, value in checkpoint_provenance(p2_ckpt_path).items()
        }
        p2_checkpoint_info["p2_checkpoint_path"] = os.path.normpath(
            args.p2_ckpt)
    if args.opponent_type == "sb3":
        for prefix, selected in (
                ("p2_model", args.p2_model_zip),
                ("p2_normalization", args.p2_model_pkl)):
            identity = checkpoint_provenance(os.path.join(
                config.PROJECT_ROOT, selected))
            identity.pop("checkpoint_benchmark", None)
            for key, value in identity.items():
                p2_checkpoint_info[f"{prefix}_{key}"] = value
    session_log = MatchSessionLog(
        os.path.join(config.LOG_DIR, "model_testing", "apex_viewer"),
        {
            "mode": f"apex_vs_{args.opponent_type}",
            "device": args.device,
            "opponent_type": args.opponent_type,
            "requested_opponent": args.opponent,
            "cpu_level": args.cpu_level if args.opponent_type == "cpu" else None,
            "p2_algorithm": (
                args.p2_algo if args.opponent_type == "sb3" else None),
            "p2_environment": (
                args.p2_env if args.opponent_type == "sb3" else None),
            "rematch_delay_seconds": args.rematch_delay,
            "infinite_match": args.infinite_match,
            **checkpoint_info,
            **p2_checkpoint_info,
        },
    )
    if session_log.path:
        print(f"[viewer] log persistente: {session_log.path}", flush=True)

    def cleanup_session():
        nonlocal cleanup_done
        if cleanup_done:
            return
        cleanup_done = True
        # Un CTRL_BREAK tardio no debe interrumpir el propio cierre.
        if hasattr(signal, "SIGBREAK"):
            try:
                signal.signal(signal.SIGBREAK, signal.SIG_IGN)
            except (OSError, ValueError):
                pass
        try:
            with open(agent_state_path, "w") as f:
                f.write("PAUSE")
        except OSError:
            pass
        failsafe_env(env=env)

    # Cubre tambien errores durante carga del modelo o arranque de BizHawk,
    # antes de que el loop principal alcance su finally.
    atexit.register(cleanup_session)

    try:
        # El Lua arranca PAUSADO y solo corre cuando .agent_state dice PLAY.
        with open(agent_state_path, "w") as f:
            f.write("PLAY")

        # El dashboard manda CTRL_BREAK al grupo del subprocess para detener
        # una evaluacion con limpieza. Ctrl+C conserva la misma ruta.
        if hasattr(signal, "SIGBREAK"):
            def interrupt_stand(_signum, _frame):
                raise KeyboardInterrupt

            signal.signal(signal.SIGBREAK, interrupt_stand)

        stop_file_path = os.path.join(config.PROJECT_ROOT, ".stop_training")

        def stop_requested() -> bool:
            return os.path.exists(stop_file_path)

        choose_action_p1, meta = load_champion(ckpt_path, args.device)
        print(f"[viewer] campeon: {os.path.basename(ckpt_path)} "
              f"({meta['in_dim']} in, {N_ACTIONS} acciones, macros)", flush=True)
        choose_action_p2 = None
        sb3_opponent = None
        if args.opponent_type == "model":
            p2_device = args.p2_device or args.device
            choose_action_p2, p2_meta = load_champion(p2_ckpt_path, p2_device)
            print(f"[viewer] modelo P2: {os.path.basename(p2_ckpt_path)} "
                  f"({p2_meta['in_dim']} in, {N_ACTIONS} acciones, macros)",
                  flush=True)

        lua_path = os.path.join(config.PROJECT_ROOT, "lua", "v2.0",
                                "stand_env_client.lua")
        env = StreetFighterEnvV2(lua_path=lua_path, trainable=False, rank=0,
                                 player=1)
        if args.opponent_type == "sb3":
            p2_device = args.p2_device or args.device
            sb3_opponent = load_sb3_opponent(
                env,
                args.p2_algo,
                args.p2_env,
                os.path.join(config.PROJECT_ROOT, args.p2_model_zip),
                os.path.join(config.PROJECT_ROOT, args.p2_model_pkl),
                p2_device,
            )
            print(
                f"[viewer] modelo P2: {os.path.basename(args.p2_model_zip)} "
                f"({args.p2_algo.upper()} {args.p2_env}, {p2_device})",
                flush=True,
            )
    except BaseException as exc:
        session_log.write(
            "session_error",
            phase="startup",
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback=traceback.format_exc(),
        )
        session_log.close(
            "stopped" if isinstance(exc, KeyboardInterrupt) else "startup_error",
            completed_rounds=0,
            ia_wins=0,
            retador_wins=0,
        )
        cleanup_session()
        raise

    frames = deque(maxlen=NUM_FRAMES)
    track = RamTrack()
    player = MacroPlayer()
    frames_p2 = deque(maxlen=NUM_FRAMES)
    track_p2 = RamTrack()
    player_p2 = MacroPlayer()
    p1_wins = p2_wins = 0
    match_count = 1
    completed_rounds = 0
    round_started = False
    round_started_at = None
    current_opponent = None
    ko_time = None
    winner_msg = None

    def reset_to(state_path: str, discard_pending_payload: bool = False):
        nonlocal track, round_started, round_started_at, ko_time, winner_msg
        nonlocal track_p2, current_opponent
        ram = request_round_reset(
            env, state_path, p1_wins, p2_wins,
            discard_pending_payload=discard_pending_payload,
        )
        track = RamTrack()
        frame, track, _s1, _s2 = assemble_v4_frame(ram, track, is_reset=True)
        frames.clear()
        for _ in range(NUM_FRAMES):
            frames.append(frame)
        player.reset(np.concatenate(frames))
        if choose_action_p2 is not None:
            p2_frame, track_p2, _p2s1, _p2s2 = assemble_v4_frame(
                ram_for_player(ram, 2), RamTrack(), is_reset=True)
            frames_p2.clear()
            for _ in range(NUM_FRAMES):
                frames_p2.append(p2_frame)
            player_p2.reset(np.concatenate(frames_p2))
        if sb3_opponent is not None:
            sb3_opponent.reset(sb3_payload_from_ram(ram))
        round_started = False
        round_started_at = time.time()
        ko_time = None
        winner_msg = None
        current_opponent = opponent_from_state(state_path)
        session_log.write(
            "round_start",
            round=match_count,
            opponent_type=args.opponent_type,
            opponent=current_opponent,
            cpu_level=args.cpu_level if args.opponent_type == "cpu" else None,
            savestate=os.path.basename(state_path),
            ia_wins=p1_wins,
            retador_wins=p2_wins,
        )

    print("\n" + "=" * 50)
    print("  MODEL TESTING -- reta a la IA (control = Player 2)")
    print("  Ctrl+C termina la sesion y cierra el emulador")
    print("=" * 50 + "\n", flush=True)

    session_end_reason = "error"
    try:
        # Solo el cold-start hereda el payload que Lua emitió al conectarse.
        reset_to(
            pick_state(
                config.STATES_DIR, args.opponent,
                opponent_type=args.opponent_type,
                cpu_level=args.cpu_level,
            ),
            discard_pending_payload=True,
        )
        while True:
            if stop_requested():
                raise KeyboardInterrupt
            stacked = np.concatenate(frames)
            direction, button = player.next_step(stacked, choose_action_p1)
            p2_command = HUMAN_PASSTHROUGH
            if choose_action_p2 is not None:
                stacked_p2 = np.concatenate(frames_p2)
                p2_direction, p2_button = player_p2.next_step(
                    stacked_p2, choose_action_p2)
                p2_command = bits_command(p2_direction, p2_button)
            elif sb3_opponent is not None:
                p2_command = sb3_opponent.command()
            env.send_command(bits_command(direction, button) + p2_command + "\n")

            raw = env.receive_payload()
            if not raw:
                continue
            try:
                ram = parse_payload(raw)
            except ValueError:
                continue  # frame ilegible transitorio: el deque conserva el ultimo bueno
            frame, track, _s1, _s2 = assemble_v4_frame(ram, track)
            frames.append(frame)
            if choose_action_p2 is not None:
                p2_frame, track_p2, _p2s1, _p2s2 = assemble_v4_frame(
                    ram_for_player(ram, 2), track_p2)
                frames_p2.append(p2_frame)
            if sb3_opponent is not None:
                sb3_opponent.observe(sb3_payload_from_ram(ram))

            p1_hp_f, p2_hp_f = float(frame[0]), float(frame[1])
            if not round_started and p1_hp_f > 0 and p2_hp_f > 0:
                round_started = True
            if round_started and ko_time is None:
                p1_ko = hp_to_signed(ram["p1_hp"]) < 0
                p2_ko = hp_to_signed(ram["p2_hp"]) < 0
                round_timer = int(ram["round_timer"])
                result = resolve_round(
                    p1_ko, p2_ko, round_timer, p1_hp_f, p2_hp_f)
                if result is not None:
                    winner, winner_msg = result
                    if winner == "ia":
                        p1_wins += 1
                    elif winner == "retador":
                        p2_wins += 1
                    completed_rounds += 1
                    ko_time = time.time()
                    session_log.write(
                        "round_end",
                        round=match_count,
                        opponent_type=args.opponent_type,
                        opponent=current_opponent,
                        cpu_level=(
                            args.cpu_level if args.opponent_type == "cpu" else None
                        ),
                        winner=winner,
                        result=winner_msg,
                        ending="ko" if p1_ko or p2_ko else "time_over",
                        p1_hp=p1_hp_f,
                        p2_hp=p2_hp_f,
                        round_timer=round_timer,
                        duration_seconds=(
                            round(time.time() - round_started_at, 3)
                            if round_started_at is not None else None
                        ),
                        ia_wins=p1_wins,
                        retador_wins=p2_wins,
                    )
            if ko_time is not None and time.time() - ko_time >= args.rematch_delay:
                print(f"[round {match_count}] {winner_msg}  |  "
                      f"IA {p1_wins} - {p2_wins} Retador", flush=True)
                if args.infinite_match:
                    match_count += 1
                    reset_to(pick_state(
                        config.STATES_DIR, args.opponent,
                        opponent_type=args.opponent_type,
                        cpu_level=args.cpu_level,
                    ))
                else:
                    # Sin auto-rematch dejamos BizHawk abierto en el resultado.
                    # Lua sigue avanzando frames en PAUSE sin esperar socket, y
                    # el botón Toggle puede reanudar esta misma sesión.
                    with open(agent_state_path, "w") as state_file:
                        state_file.write("PAUSE")
                    session_log.write(
                        "match_paused",
                        round=match_count,
                        reason="infinite_match_disabled",
                        ia_wins=p1_wins,
                        retador_wins=p2_wins,
                    )
                    print(
                        "[viewer] Auto-rematch desactivado: BizHawk queda "
                        "abierto. Usa Toggle para reanudar o Terminate Match "
                        "para cerrar.",
                        flush=True,
                    )
                    while not stop_requested():
                        try:
                            with open(agent_state_path, "r") as state_file:
                                resumed = "PLAY" in state_file.read()
                        except OSError:
                            resumed = False
                        if resumed:
                            round_started = False
                            round_started_at = time.time()
                            ko_time = None
                            winner_msg = None
                            print("[viewer] sesión reanudada.", flush=True)
                            break
                        time.sleep(0.1)
                    if stop_requested():
                        raise KeyboardInterrupt
    except KeyboardInterrupt:
        session_end_reason = "stopped"
        print(f"\n[viewer] sesion terminada. Marcador final: "
              f"IA {p1_wins} - {p2_wins} Retador", flush=True)
    except BaseException as exc:
        session_end_reason = "error"
        session_log.write(
            "session_error",
            phase="gameplay",
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        session_log.close(
            session_end_reason,
            completed_rounds=completed_rounds,
            ia_wins=p1_wins,
            retador_wins=p2_wins,
        )
        cleanup_session()


if __name__ == "__main__":
    main()
