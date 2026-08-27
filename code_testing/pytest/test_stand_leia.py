# test_stand_leia.py
#
# Paridad offline del driver del stand (src/scripts/stand_leia.py) sin
# emulador, sin torch y sin el rig Windows (core.config y sf2_v2 se importan
# tarde dentro de main(), asi que importar el modulo es seguro aqui):
#   1. el parseo del payload de 25 campos round-tripea contra el formato Lua,
#   2. GOLD: MacroPlayer reproduce EXACTAMENTE la secuencia de (dir, boton)
#      y de espejos que MacroActionWrapper ejecuta para el mismo stream de
#      decisiones y las mismas observaciones,
#   3. los 10 bits del comando calcan el orden del Lua (oracle escrito a mano),
#   4. la extraccion de HP del frame mas reciente usa los indices correctos.

import os
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
for sub in ("src", os.path.join("src", "scripts")):
    p = os.path.join(PROJECT_ROOT, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pytest

import gymnasium as gym
from gymnasium import spaces

from envs.action_macros import MACROS, N_ACTIONS, N_PRIMITIVES
from envs.macro_wrapper import MacroActionWrapper
from envs.retro_env import V4_FRAME_DIM
from stand_leia import (
    HUMAN_PASSTHROUGH, PAYLOAD_KEYS, MacroPlayer, bits_command, parse_payload,
    resolve_round,
)


# --------------------------------------------------------------------------
# 1. Payload
# --------------------------------------------------------------------------

def test_parse_payload_round_trip_with_bizhawk_framing():
    # En el alambre real BizHawk antepone "<len> " a lo que el Lua manda
    # ("0 <csv>"): el parser debe sobrevivir ambos prefijos (y de paso al
    # CSV pelon, por si el framing cambia).
    values = {k: i * 7 - 3 for i, k in enumerate(PAYLOAD_KEYS)}
    csv = ",".join(str(values[k]) for k in PAYLOAD_KEYS)
    for raw in (f"{len('0 ' + csv)} 0 {csv}\n", "0 " + csv + "\n", csv):
        assert parse_payload(raw) == values


def test_parse_payload_rejects_old_lua():
    with pytest.raises(ValueError, match="stand_env_client"):
        parse_payload("29 0 1,2,3,4,5,6,7,8,9,10,11,12,13")  # el de 13 campos


# --------------------------------------------------------------------------
# 1b. Resolucion del round (KO por signo + time-over por HP)
# --------------------------------------------------------------------------

def test_resolve_round_ko_by_sign():
    assert resolve_round(False, True, 90, 100.0, 0.0)[0] == "ia"
    assert resolve_round(True, False, 90, 0.0, 80.0)[0] == "retador"
    assert resolve_round(True, True, 90, 0.0, 0.0)[0] == "empate"
    # AMBOS frames en 0 (garbage de transicion) sin KO firmado: sigue vivo
    assert resolve_round(False, False, 90, 0.0, 0.0) is None


def test_resolve_round_time_over():
    winner, msg = resolve_round(False, False, 0, 120.0, 50.0)
    assert winner == "ia" and "tiempo" in msg
    winner, _msg = resolve_round(False, False, 0, 10.0, 50.0)
    assert winner == "retador"
    assert resolve_round(False, False, 0, 60.0, 60.0)[0] == "empate"
    assert resolve_round(False, False, 153, 60.0, 60.0) is None


# --------------------------------------------------------------------------
# 2. GOLD: MacroPlayer == MacroActionWrapper
# --------------------------------------------------------------------------

class RecordingEnv(gym.Env):
    """Registra cada accion interna y sirve una observacion por paso desde
    una lista pre-armada (23 floats x 4 frames apilados)."""

    def __init__(self, obs_stream):
        super().__init__()
        self.action_space = spaces.MultiDiscrete([9, 7])
        self.observation_space = spaces.Box(-1e6, 1e6,
                                            shape=(V4_FRAME_DIM * 4,),
                                            dtype=np.float32)
        self.obs_stream = list(obs_stream)
        self.recorded = []

    def reset(self, **kwargs):
        return self.obs_stream[0], {}

    def step(self, action):
        self.recorded.append((int(action[0]), int(action[1])))
        obs = self.obs_stream[min(len(self.recorded), len(self.obs_stream) - 1)]
        return obs, 0.0, False, False, {}


def stacked_obs(rel_x: float) -> np.ndarray:
    obs = np.zeros(V4_FRAME_DIM * 4, dtype=np.float32)
    obs[3 * V4_FRAME_DIM + 2] = rel_x  # rel_x del frame MAS RECIENTE
    return obs


def test_macro_player_matches_wrapper_bit_for_bit():
    # Decisiones: un primitivo, un hadouken, cruza de lado a mitad del
    # tatsumaki (el espejo NO debe cambiar a mitad de macro), otro primitivo
    # ya del lado izquierdo, y un shoryuken espejado.
    macro_ids = {name: N_PRIMITIVES + i for i, name in enumerate(MACROS)}
    decisions = [17, macro_ids["hadouken_hp"], macro_ids["tatsumaki_lk"],
                 5, macro_ids["shoryuken_lp"]]

    # Obs por paso interno: 1 + 3 + 3 + 1 + 3 = 11 pasos; el signo de rel_x
    # se voltea DENTRO del tatsumaki (pasos 5-7) y se queda negativo.
    rel_xs = [80, 80, 80, 80, 40, -60, -60, -60, -60, -60, -60, -60]
    obs_stream = [stacked_obs(x) for x in rel_xs]

    env = RecordingEnv(obs_stream)
    wrapper = MacroActionWrapper(env, obs_rel_x_index=2,
                                 frame_size=V4_FRAME_DIM)
    obs, _ = wrapper.reset()
    it = iter(decisions)
    facings_wrapper = []
    for action in it:
        facings_wrapper.append(wrapper._facing_right)
        obs, _r, _t, _tr, _info = wrapper.step(action)
    expected_steps = env.recorded

    # El player del stand: mismas decisiones, mismas obs por paso interno.
    player = MacroPlayer()
    player.reset(obs_stream[0])
    decision_it = iter(decisions)
    got_steps, facings_player = [], []
    obs_idx = 0

    def choose(_stacked):
        facings_player.append(player.facing_right)
        return next(decision_it)

    for _ in range(len(expected_steps)):
        step = player.next_step(obs_stream[obs_idx], choose)
        got_steps.append(step)
        obs_idx += 1

    assert got_steps == expected_steps
    assert facings_player == facings_wrapper
    assert player.queue == []  # ningun macro quedo a medias


def test_macro_player_one_decision_per_macro():
    calls = []

    def choose(_):
        calls.append(1)
        return N_PRIMITIVES  # el primer macro (3 pasos)

    player = MacroPlayer()
    player.reset(stacked_obs(50))
    for _ in range(3):
        player.next_step(stacked_obs(50), choose)
    assert len(calls) == 1  # una consulta cubre los 3 pasos del macro


# --------------------------------------------------------------------------
# 3. Bits del comando (oracle manual del orden del Lua)
# --------------------------------------------------------------------------

def test_bits_command_oracle():
    # Orden Lua: Up Down Left Right A B C X Y Z
    assert bits_command(0, 0) == "0000000000"
    assert bits_command(1, 0) == "1000000000"          # Up
    assert bits_command(4, 6) == "0001000001"          # Right + Z (HP)
    assert bits_command(2, 1) == "0100100000"          # Down + A (LK)
    assert bits_command(8, 4) == "0101000100"          # Down+Right + X (LP)
    assert len(HUMAN_PASSTHROUGH) == 10
    for direction in range(9):
        for button in range(7):
            cmd = bits_command(direction, button)
            assert len(cmd) == 10 and set(cmd) <= {"0", "1"}


# --------------------------------------------------------------------------
# 4. HP del frame mas reciente
# --------------------------------------------------------------------------

def test_latest_frame_hp_indices():
    frames = [np.full(V4_FRAME_DIM, i, dtype=np.float32) for i in range(4)]
    frames[-1][0], frames[-1][1] = 111.0, 42.0
    stacked = np.concatenate(frames)
    assert float(stacked[-V4_FRAME_DIM]) == 111.0
    assert float(stacked[-V4_FRAME_DIM + 1]) == 42.0


def test_n_actions_is_72():
    assert N_ACTIONS == 72  # el vocabulario del campeon con macros
