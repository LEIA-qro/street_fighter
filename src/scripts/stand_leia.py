# stand_leia.py -- HUMANO vs el campeon DQN, en BizHawk (modo stand LEIA).
#
#   .venv\Scripts\python.exe src\scripts\stand_leia.py --ckpt benchmarks\apex_milestones\apex_v1592_benchmarked.pt
#
# P1 = la IA (Ryu, el campeon Ape-X con macros). P2 = el visitante, con el
# control fisico configurado como Player 2 en BizHawk (este script manda
# ".........." para P2: el Lua no inyecta y el pad pasa directo). Corre sobre
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
import os
import random
import signal
import sys
import time
from collections import deque

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


def pick_state(states_dir: str, opponent: str) -> str:
    if opponent == "RANDOM":
        available = [
            name for name in OPPONENTS
            if os.path.isfile(os.path.join(
                states_dir, f"RYU_{name}_R1_PvP.State"))
        ]
        if not available:
            raise SystemExit(
                f"[viewer] no hay savestates RYU_*_R1_PvP.State en "
                f"{states_dir}")
        opponent = random.choice(available)
    path = os.path.join(states_dir, f"RYU_{opponent}_R1_PvP.State")
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
    ap.add_argument("--rematch-delay", type=float, default=4.0,
                    help="segundos de pantalla de KO antes del rematch")
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
    config.P2_MODEL_NAME = "RETADOR (tu)"
    config.generate_lua_config()

    agent_state_path = os.path.join(config.PROJECT_ROOT, ".agent_state")
    env = None
    cleanup_done = False

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

        ckpt_path = os.path.join(config.PROJECT_ROOT, args.ckpt)
        choose_action, meta = load_champion(ckpt_path, args.device)
        print(f"[viewer] campeon: {os.path.basename(ckpt_path)} "
              f"({meta['in_dim']} in, {N_ACTIONS} acciones, macros)", flush=True)

        lua_path = os.path.join(config.PROJECT_ROOT, "lua", "v2.0",
                                "stand_env_client.lua")
        env = StreetFighterEnvV2(lua_path=lua_path, trainable=False, rank=0,
                                 player=1)
    except BaseException:
        cleanup_session()
        raise

    frames = deque(maxlen=NUM_FRAMES)
    track = RamTrack()
    player = MacroPlayer()
    p1_wins = p2_wins = 0
    match_count = 1
    round_started = False
    ko_time = None
    winner_msg = None

    def reset_to(state_path: str, discard_pending_payload: bool = False):
        nonlocal track, round_started, ko_time, winner_msg
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
        round_started = False
        ko_time = None
        winner_msg = None

    print("\n" + "=" * 50)
    print("  MODEL TESTING -- reta a la IA (control = Player 2)")
    print("  Ctrl+C termina la sesion y cierra el emulador")
    print("=" * 50 + "\n", flush=True)

    try:
        # Solo el cold-start hereda el payload que Lua emitió al conectarse.
        reset_to(
            pick_state(config.STATES_DIR, args.opponent),
            discard_pending_payload=True,
        )
        while True:
            if stop_requested():
                raise KeyboardInterrupt
            stacked = np.concatenate(frames)
            direction, button = player.next_step(stacked, choose_action)
            env.send_command(bits_command(direction, button)
                             + HUMAN_PASSTHROUGH + "\n")

            raw = env.receive_payload()
            if not raw:
                continue
            try:
                ram = parse_payload(raw)
            except ValueError:
                continue  # frame ilegible transitorio: el deque conserva el ultimo bueno
            frame, track, _s1, _s2 = assemble_v4_frame(ram, track)
            frames.append(frame)

            p1_hp_f, p2_hp_f = float(frame[0]), float(frame[1])
            if not round_started and p1_hp_f > 0 and p2_hp_f > 0:
                round_started = True
            if round_started and ko_time is None:
                result = resolve_round(
                    hp_to_signed(ram["p1_hp"]) < 0,
                    hp_to_signed(ram["p2_hp"]) < 0,
                    int(ram["round_timer"]), p1_hp_f, p2_hp_f)
                if result is not None:
                    winner, winner_msg = result
                    if winner == "ia":
                        p1_wins += 1
                    elif winner == "retador":
                        p2_wins += 1
                    ko_time = time.time()
            if ko_time is not None and time.time() - ko_time >= args.rematch_delay:
                print(f"[round {match_count}] {winner_msg}  |  "
                      f"IA {p1_wins} - {p2_wins} Retador", flush=True)
                match_count += 1
                reset_to(pick_state(config.STATES_DIR, args.opponent))
    except KeyboardInterrupt:
        print(f"\n[viewer] sesion terminada. Marcador final: "
              f"IA {p1_wins} - {p2_wins} Retador", flush=True)
    finally:
        cleanup_session()


if __name__ == "__main__":
    main()
