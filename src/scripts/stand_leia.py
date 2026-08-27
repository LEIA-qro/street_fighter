# stand_leia.py -- HUMANO vs el campeon DQN, en BizHawk (modo stand LEIA).
#
#   .venv\Scripts\python.exe src\scripts\stand_leia.py --ckpt benchmarks\apex_milestones\apex_v781_escalera831.pt
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
# resetear siempre hay UN payload pre-reset en vuelo: reset_to lo descarta y
# prima del primer payload post-reset -- con eso cada receive posterior trae
# el estado GENERADO POR la accion recien mandada (cero lag, mejor paridad
# con el rig retro que el match test).
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
import os
import random
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

MAX_BAD_PAYLOADS = 120  # ~2 s de basura consistente = cliente equivocado


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

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = ckpt["meta"]
    if not meta.get("macros", False) or int(meta.get("n_actions", 0)) != N_ACTIONS:
        raise SystemExit(f"[stand] el checkpoint no es del campeon con macros "
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
        opponent = random.choice(OPPONENTS)
    path = os.path.join(states_dir, f"RYU_{opponent}_R1_PvP.State")
    if not os.path.exists(path):
        raise SystemExit(f"[stand] no existe {path}")
    return path


def main():
    ap = argparse.ArgumentParser(description="Stand LEIA: humano vs el campeon DQN")
    ap.add_argument("--ckpt",
                    default=os.path.join("benchmarks", "apex_milestones",
                                         "apex_v781_escalera831.pt"))
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

    # El Lua arranca PAUSADO y solo corre cuando .agent_state dice PLAY; el
    # unico que escribe ese archivo es el web dashboard, asi que en una
    # maquina fresca (el stand) nadie lo pondria jamas: lo escribimos aqui.
    agent_state_path = os.path.join(config.PROJECT_ROOT, ".agent_state")
    with open(agent_state_path, "w") as f:
        f.write("PLAY")

    ckpt_path = os.path.join(config.PROJECT_ROOT, args.ckpt)
    choose_action, meta = load_champion(ckpt_path, args.device)
    print(f"[stand] campeon: {os.path.basename(ckpt_path)} "
          f"({meta['in_dim']} in, {N_ACTIONS} acciones, macros)", flush=True)

    lua_path = os.path.join(config.PROJECT_ROOT, "lua", "v2.0",
                            "stand_env_client.lua")
    env = StreetFighterEnvV2(lua_path=lua_path, trainable=False, rank=0,
                             player=1)

    frames = deque(maxlen=NUM_FRAMES)
    track = RamTrack()
    player = MacroPlayer()
    p1_wins = p2_wins = 0
    match_count = 1
    round_started = False
    ko_time = None
    winner_msg = None

    def receive_valid_ram() -> dict:
        """Recibe hasta parsear un payload valido; basura transitoria se
        tira (el socket interactivo la emite a proposito al reconectar),
        basura CONSISTENTE truena con el diagnostico del Lua equivocado."""
        bad = 0
        while True:
            raw = env.receive_payload()
            if not raw:
                continue
            try:
                return parse_payload(raw)
            except ValueError:
                bad += 1
                if bad >= MAX_BAD_PAYLOADS:
                    raise

    def reset_to(state_path: str):
        nonlocal track, round_started, ko_time, winner_msg
        env.send_command(f"RESET {state_path}|{p1_wins}|{p2_wins}\n")
        # Siempre hay UN payload pre-reset en vuelo (generado antes del
        # savestate.load): descartarlo y primar del primero post-reset.
        env.receive_payload()
        ram = receive_valid_ram()
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
    print("  STAND LEIA -- reta a la IA (control = Player 2)")
    print("  Ctrl+C termina la sesion y cierra el emulador")
    print("=" * 50 + "\n", flush=True)

    try:
        reset_to(pick_state(config.STATES_DIR, args.opponent))
        while True:
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
        print(f"\n[stand] sesion terminada. Marcador final: "
              f"IA {p1_wins} - {p2_wins} Retador", flush=True)
    finally:
        try:
            with open(agent_state_path, "w") as f:
                f.write("PAUSE")
        except OSError:
            pass
        failsafe_env(env=env)


if __name__ == "__main__":
    main()
