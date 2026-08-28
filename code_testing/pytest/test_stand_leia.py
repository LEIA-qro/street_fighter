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

import json
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
    DEFAULT_CHECKPOINT, HUMAN_PASSTHROUGH, PAYLOAD_KEYS, MacroPlayer,
    MatchSessionLog, bits_command, checkpoint_provenance, opponent_from_state,
    parse_payload, pick_state, ram_for_player, request_round_reset, resolve_round,
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
# 1c. Reset: cold-start tiene un payload pendiente; rematch no
# --------------------------------------------------------------------------

class ResetProtocolEnv:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.events = []

    def send_command(self, command):
        self.events.append(("send", command))

    def receive_payload(self):
        self.events.append(("receive", None))
        return self.payloads.pop(0)


def test_cold_start_discards_exactly_one_pre_reset_payload():
    env = ResetProtocolEnv(["pre-reset", "post-reset"])

    payload = request_round_reset(
        env, "RYU_KEN_R1_PvP.State", 0, 0,
        discard_pending_payload=True,
        payload_parser=lambda raw: raw,
    )

    assert payload == "post-reset"
    assert env.events == [
        ("send", "RESET RYU_KEN_R1_PvP.State|0|0\n"),
        ("receive", None),
        ("receive", None),
    ]


def test_rematch_consumes_one_post_reset_payload_without_deadlock():
    env = ResetProtocolEnv(["post-reset"])

    payload = request_round_reset(
        env, "RYU_GUILE_R1_PvP.State", 3, 2,
        payload_parser=lambda raw: raw,
    )

    assert payload == "post-reset"
    assert env.events == [
        ("send", "RESET RYU_GUILE_R1_PvP.State|3|2\n"),
        ("receive", None),
    ]


def test_invalid_post_reset_payload_fails_without_second_receive():
    env = ResetProtocolEnv(["invalid"])

    def reject_payload(_raw):
        raise ValueError("bad payload")

    with pytest.raises(RuntimeError, match="payload post-reset inválido"):
        request_round_reset(
            env, "RYU_GUILE_R1_PvP.State", 0, 0,
            payload_parser=reject_payload,
        )

    assert env.events == [
        ("send", "RESET RYU_GUILE_R1_PvP.State|0|0\n"),
        ("receive", None),
    ]


def test_random_opponent_only_chooses_an_available_savestate(tmp_path, monkeypatch):
    (tmp_path / "RYU_KEN_R1_PvP.State").touch()
    (tmp_path / "RYU_GUILE_R1_PvP.State").touch()
    seen = []

    def choose_available(options):
        seen.append(tuple(options))
        return "GUILE"

    monkeypatch.setattr("stand_leia.random.choice", choose_available)

    state_path = pick_state(str(tmp_path), "RANDOM")

    assert Path(state_path).name == "RYU_GUILE_R1_PvP.State"
    assert seen == [("GUILE", "KEN")]


def test_random_opponent_without_savestates_fails_before_launch(tmp_path):
    with pytest.raises(SystemExit, match="no hay savestates"):
        pick_state(str(tmp_path), "RANDOM")


def test_cpu_opponent_uses_exact_selected_level_and_hard_for_level_8(tmp_path):
    level3 = tmp_path / "RYU_KEN_R1_lvl3.State"
    hard = tmp_path / "RYU_KEN_R1_HARD.State"
    level3.touch()
    hard.touch()

    assert Path(pick_state(
        str(tmp_path), "KEN", opponent_type="cpu", cpu_level=3
    )).name == level3.name
    assert Path(pick_state(
        str(tmp_path), "KEN", opponent_type="cpu", cpu_level=8
    )).name == hard.name


def test_cpu_random_filters_to_the_exact_selected_level(tmp_path, monkeypatch):
    (tmp_path / "RYU_KEN_R1_lvl2.State").touch()
    (tmp_path / "RYU_GUILE_R1_lvl3.State").touch()
    (tmp_path / "RYU_KEN_R1_lvl3.State").touch()
    seen = []

    def choose_available(options):
        seen.append(tuple(options))
        return "GUILE"

    monkeypatch.setattr("stand_leia.random.choice", choose_available)
    selected = pick_state(
        str(tmp_path), "RANDOM", opponent_type="cpu", cpu_level=3)

    assert Path(selected).name == "RYU_GUILE_R1_lvl3.State"
    assert seen == [("GUILE", "KEN")]


def test_ram_for_player_two_swaps_fighters_and_air_conventions():
    ram = {key: 0 for key in PAYLOAD_KEYS}
    ram.update({
        "p1_hp": 170, "p2_hp": 80,
        "p1_x": 100, "p2_x": 350,
        "p1_y": 20, "p2_y": 40,
        "p1_state_word": 0x1234, "p2_state_word": 0x5678,
        "p1_proj_x": 10, "p2_proj_x": 20,
        "p1_char": 0, "p2_char": 4,
        "p1_btn": 11, "p2_btn": 22,
        "p1_air_raw": 0, "p2_air_raw": 13,
        "p1_chest": 31, "p2_chest": 41,
        "p1_head": 32, "p2_head": 42,
        "matches_won": 2, "enemy_matches_won": 5,
    })

    p2 = ram_for_player(ram, 2)

    assert (p2["p1_hp"], p2["p2_hp"]) == (80, 170)
    assert (p2["p1_x"], p2["p2_x"]) == (350, 100)
    assert (p2["p1_char"], p2["p2_char"]) == (4, 0)
    assert (p2["p1_air_raw"], p2["p2_air_raw"]) == (257, 14)
    assert (p2["matches_won"], p2["enemy_matches_won"]) == (5, 2)


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


def test_default_checkpoint_tracks_frozen_benchmarked_champion():
    assert Path(DEFAULT_CHECKPOINT).as_posix().endswith(
        "benchmarks/apex_milestones/apex_v1592_benchmarked.pt")


def test_match_session_log_flushes_rounds_and_summary(tmp_path):
    log = MatchSessionLog(str(tmp_path), {"checkpoint": "champion.pt"})
    assert log.path is not None

    log.write(
        "round_end", round=1, opponent="KEN", winner="ia",
        ia_wins=1, retador_wins=0,
    )
    # Debe poder leerse antes de close(): write() hace flush + fsync.
    live_records = [json.loads(line) for line in Path(log.path).read_text(
        encoding="utf-8").splitlines()]
    assert [record["event"] for record in live_records] == [
        "session_start", "round_end"]
    assert live_records[1]["opponent"] == "KEN"

    log.close("stopped", completed_rounds=1, ia_wins=1, retador_wins=0)
    records = [json.loads(line) for line in Path(log.path).read_text(
        encoding="utf-8").splitlines()]
    assert records[-1]["event"] == "session_end"
    assert records[-1]["completed_rounds"] == 1
    assert len({record["session_id"] for record in records}) == 1


def test_checkpoint_provenance_records_hash_and_sidecar(tmp_path):
    checkpoint = tmp_path / "champion.pt"
    checkpoint.write_bytes(b"model-bytes")
    Path(str(checkpoint) + ".json").write_text(
        '{"weights_version":1592,"wr_media":0.945}', encoding="utf-8")

    identity = checkpoint_provenance(str(checkpoint))

    assert identity["checkpoint_sha256"] == (
        "357e5d6fafa34d27360fec24b4326d3534905e33c6acdee60198fb078b7b79e5")
    assert identity["checkpoint_benchmark"]["weights_version"] == 1592


def test_opponent_from_state_reports_random_choice_character():
    assert opponent_from_state("C:/LEIA/states/RYU_CHUNLI_R1_PvP.State") == "CHUNLI"
    assert opponent_from_state("C:/LEIA/states/RYU_KEN_R1_lvl7.State") == "KEN"
    assert opponent_from_state("C:/LEIA/states/RYU_GUILE_R1_HARD.State") == "GUILE"
