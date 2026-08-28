# retro_env.py
#
# Headless stable-retro backend for Street Fighter II' Special Champion Edition
# that speaks the EXACT v4 contract of the BizHawk rig (sf2_v4.py + base_env.py):
# same MultiDiscrete([9, 7]) action space, same 23-float frame layout stacked 4
# deep, same HP-sentinel discipline, same reward bookkeeping and info keys.
# Wrappers, extractors, curriculum callbacks and trained models port unchanged;
# only the emulation transport differs (libretro core in-process instead of a
# BizHawk+Lua TCP socket, ~3,700 fps/proc vs ~350 agent steps/s/env).
#
# The Lua client (lua/v2.0/training_env_client.lua) derives several payload
# fields before they hit Python: it splits the state words into hi/lo bytes,
# normalizes the airborne reads to 0/1, masks p2_y to its low byte, and turns
# the raw projectile X into "raw if it moved since last read, else -1".
# stable-retro can only expose raw RAM, so retro_integration/.../data.json
# ships the RAW fields and assemble_v4_frame() reproduces every one of those
# derivations bit-for-bit in Python.
#
# Everything that does not need a running emulator (action translation, sticky
# movement, frame assembly, episode spacing aggregates) is a module-level pure
# function so the unit tests run on machines with no stable-retro, no ROM and
# no EmuHawk.exe. stable_retro itself is imported lazily inside RetroSF2Env.

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import gymnasium
from gymnasium import spaces

from envs.reward import (RewardConfig, RewardState, RoundTracker,
                         compute_reward, hp_to_signed)

# --------------------------------------------------------------------------
# Constants duplicated from core/config.py and envs/base_env.py, NOT imported:
# core.config raises FileNotFoundError at import time when EmuHawk.exe is not
# at its Windows path, and base_env imports core.config -- either import would
# kill this backend on the Linux/WSL2/macOS fleet it exists for. These values
# are contract constants (they define the wire/obs format, not tunables) and
# test_retro_env.py pins the ones that have an importable counterpart.
# --------------------------------------------------------------------------
FRAME_SKIP = 4              # emulator frames per agent step (Lua client's loop cadence)
NUM_FRAMES = 4              # config.NUM_FRAMES: stacked agent frames per observation
MAX_STEPS_PER_ROUND = 1500  # config.MAX_STEPS_PER_ROUND: artificial round timeout
# StreetFighterBaseEnv.HP_SENTINEL_THRESHOLD. Kept for the obs clamp only:
# death is decided by the SIGN of the HP word (envs.reward.hp_to_signed), never
# by this threshold. A LIVE fighter's HP is always 0..176, so nothing real ever
# trips it; a dead one's word is negative, which the sign test catches first
# (and which, read unsigned, can be as low as 65509 -- see hp_to_signed for why
# "no reading ever lands in 177..65525" is NOT the invariant). The threshold's
# only surviving job is keeping synthetic or corrupt out-of-range values out of
# the observation.
HP_SENTINEL_THRESHOLD = 200
FAR_DIST_THRESHOLD = 80.0   # base_env.FAR_DIST_THRESHOLD (old shaping dead zone boundary)
ACT_CATEGORIES = 256        # base_env.ACT_CATEGORIES
CHAR_CATEGORIES = 16        # base_env.CHAR_CATEGORIES
V4_FRAME_DIM = 23           # sf2_v4.V4_FRAME_DIM

GAME = "StreetFighterIISpecialChampionEdition-Genesis-v0"
DEFAULT_STATE = "Champion.Level1.RyuVsGuile"

# retro_integration/ at the repo root holds the custom integration whose
# data.json exposes the full 24-field RAM table (the shipped stable-retro
# integration only has health/score/timer). Registered with
# Integrations.add_custom_path at env init; Integrations.CUSTOM searches it
# FIRST and falls back to the shipped dir, which is where rom.md lives.
INTEGRATION_ROOT = str(Path(__file__).resolve().parents[2] / "retro_integration")

# Same tables as src/envs/sf2_v3.py, duplicated because sf2_v3 transitively
# imports core.config (see the constants block above for why that is fatal
# off-Windows). test_retro_env.py asserts these stay identical to sf2_v3's.
# Bits: [Up, Down, Left, Right, A(LK), B(MK), C(HK), X(LP), Y(MP), Z(HP)]
DIRECTION_MAP = {
    0: [0, 0, 0, 0], 1: [1, 0, 0, 0], 2: [0, 1, 0, 0],
    3: [0, 0, 1, 0], 4: [0, 0, 0, 1], 5: [1, 0, 1, 0],
    6: [1, 0, 0, 1], 7: [0, 1, 1, 0], 8: [0, 1, 0, 1],
}
BUTTON_MAP = {
    0: [0, 0, 0, 0, 0, 0], 1: [1, 0, 0, 0, 0, 0], 2: [0, 1, 0, 0, 0, 0],
    3: [0, 0, 1, 0, 0, 0], 4: [0, 0, 0, 1, 0, 0], 5: [0, 0, 0, 0, 1, 0],
    6: [0, 0, 0, 0, 0, 1],
}

# The project's 10-bit command order (what the Lua bridge consumes), by retro
# button NAME. Translation to retro's per-core order goes through env.buttons
# at runtime instead of a hardcoded index table, so a stable-retro release
# that reorders the Genesis pad cannot silently scramble the mapping.
PROJECT_BUTTON_ORDER = ("UP", "DOWN", "LEFT", "RIGHT", "A", "B", "C", "X", "Y", "Z")

# Single-frame observation bounds, identical to sf2_v4.StreetFighterEnvV4's
# (which cannot be imported here -- see the constants block above).
V4_SINGLE_LOW = ([0., 0., -500., -200., 0., -1., -1., -100., -100., 0., 0., 0., 0.]
                 + [0., 0.] + [0.] * 8)
V4_SINGLE_HIGH = ([176., 176., 500., 200., 250., 500., 500., 100., 100., 187.,
                   255., 192., 192.]
                  + [1., 1.]
                  + [float(ACT_CATEGORIES - 1)] * 6 + [float(CHAR_CATEGORIES - 1)] * 2)


def discrete_to_project_bits(action) -> list:
    """MultiDiscrete([9, 7]) sample -> 10-bit list in PROJECT_BUTTON_ORDER."""
    return DIRECTION_MAP[int(action[0])] + BUTTON_MAP[int(action[1])]


def project_bits_to_retro(bits, retro_buttons) -> np.ndarray:
    """10 project-order bits -> boolean action array in retro's button order.

    retro_buttons is env.buttons (e.g. Genesis: ['B','A','MODE','START','UP',
    'DOWN','LEFT','RIGHT','C','Y','X','Z']). Names absent from the project
    order (MODE, START) are never pressed.
    """
    arr = np.zeros(len(retro_buttons), dtype=np.uint8)
    for name, bit in zip(PROJECT_BUTTON_ORDER, bits):
        if bit:
            arr[retro_buttons.index(name)] = 1
    return arr


def apply_sticky(bits, direction, counter):
    """Pure port of base_env.step's sticky-direction block.

    Holds a fresh directional input for two extra agent steps so the policy
    can walk instead of jittering; crouch or the opposite direction cancels.
    Returns (new_bits, new_direction, new_counter) without mutating inputs.
    Bit indices per PROJECT_BUTTON_ORDER: 1=Down, 2=Left, 3=Right.
    """
    bits = list(bits)
    agent_left = bits[2] == 1
    agent_right = bits[3] == 1
    agent_crouch = bits[1] == 1

    opposite_input = ((direction == 'L' and agent_right)
                      or (direction == 'R' and agent_left))
    if agent_crouch or opposite_input:
        counter = 0
        direction = None

    if counter > 0:
        if direction == 'L':
            bits[2], bits[3] = 1, 0  # prevent conflicting Left+Right inputs
        elif direction == 'R':
            bits[3], bits[2] = 1, 0
        counter -= 1
    elif not agent_crouch:
        if agent_left:
            direction, counter = 'L', 2
        elif agent_right:
            direction, counter = 'R', 2

    return bits, direction, counter


@dataclass
class RamTrack:
    """Cross-step RAM memory the Lua client keeps in locals.

    prev_*_x feed the velocity deltas; prev_*_proj_raw feed the projectile
    freshness test. The Lua script initializes its projectile trackers to 0 at
    boot and this backend's reset() is a boot (savestate load), so a fresh
    track starts them at 0 as well.
    """
    prev_p1_x: int = 0
    prev_p2_x: int = 0
    prev_p1_proj_raw: int = 0
    prev_p2_proj_raw: int = 0


def assemble_v4_frame(ram: dict, track: RamTrack, is_reset: bool = False):
    """Pure raw-RAM -> v4 frame assembly. Returns (frame, next_track, p1_sent, p2_sent).

    `ram` is the data.json variable dict (retro's step info / data.lookup_all()).
    Reproduces, in order: the Lua client's field derivations (state-word split,
    air-flag normalization, p2_y low-byte mask, projectile freshness), then
    base_env._parse_payload's sentinel zeroing / translation-invariance clips,
    then sf2_v4's compact 23-float layout:
      [0-9]   HP(2), RelX, RelY, WallDist, ProjX(2), VelX(2), RelDist
      [10]    rel_y_dist   [11-12] p1/p2_head   [13-14] air flags
      [15-16] act_hi       [17-18] act_lo       [19-20] btn   [21-22] chars
    """
    p1_hp_raw, p2_hp_raw = int(ram["p1_hp"]), int(ram["p2_hp"])
    # A NEGATIVE HP word means that fighter has just been KO'd -- it is the
    # cleanest death signal the ROM emits, not an unreadable frame (see
    # hp_to_signed's docstring for the measurements). The obs still shows 0,
    # because V4_SINGLE_LOW pins the HP floor at 0; the SIGN is what callers
    # need, and RetroSF2Env._ingest re-derives it from the same raw words.
    p1_sentinel = hp_to_signed(p1_hp_raw) < 0 or p1_hp_raw > HP_SENTINEL_THRESHOLD
    p2_sentinel = hp_to_signed(p2_hp_raw) < 0 or p2_hp_raw > HP_SENTINEL_THRESHOLD
    p1_hp = 0 if p1_sentinel else p1_hp_raw
    p2_hp = 0 if p2_sentinel else p2_hp_raw

    p1_x, p2_x = int(ram["p1_x"]), int(ram["p2_x"])
    p1_y = int(ram["p1_y"])
    p2_y = int(ram["p2_y"]) & 0xFF  # Lua: band(read_u16_be(0x828A), 0xFF)

    # 0x804E/0x82CE encode position-state (hi byte) and move id (lo byte).
    p1_word, p2_word = int(ram["p1_state_word"]), int(ram["p2_state_word"])
    p1_act_hi, p1_act_lo = (p1_word >> 8) & 0xFF, p1_word & 0xFF
    p2_act_hi, p2_act_lo = (p2_word >> 8) & 0xFF, p2_word & 0xFF

    # Airborne flags, normalized to 0/1 exactly as the Lua client does.
    # 0x80C0: 0 = floor, 257 = air.  0x86F4: 14 = floor, 13 = air.
    p1_air = 1.0 if int(ram["p1_air_raw"]) != 0 else 0.0
    p2_air = 1.0 if int(ram["p2_air_raw"]) == 13 else 0.0

    # Projectile X freshness: if the raw read moved since the previous agent
    # step the projectile is live, if frozen it is dead (-1). Same test the
    # Lua client runs once per command cycle.
    p1_proj_raw, p2_proj_raw = int(ram["p1_proj_x"]), int(ram["p2_proj_x"])
    p1_proj = p1_proj_raw if p1_proj_raw != track.prev_p1_proj_raw else -1
    p2_proj = p2_proj_raw if p2_proj_raw != track.prev_p2_proj_raw else -1

    rel_dist = int(ram["rel_dist"])

    # Translation invariance + wall awareness, same clips as base_env.
    rel_x = int(np.clip(p2_x - p1_x, -500, 500))
    rel_y = int(np.clip(p2_y - p1_y, -200, 200))
    p1_corner_dist = min(p1_x, 500 - p1_x)
    p1_vel_x = 0 if is_reset else int(np.clip(p1_x - track.prev_p1_x, -100, 100))
    p2_vel_x = 0 if is_reset else int(np.clip(p2_x - track.prev_p2_x, -100, 100))

    # In the BizHawk path char ids pass through _one_hot()'s clamp before v4's
    # argmax recovers them, so out-of-range transition garbage lands on 0/15.
    # Clamp here for bit-identical observations. act_hi/act_lo/btn are already
    # byte-ranged by construction.
    p1_char = min(max(int(ram["p1_char"]), 0), CHAR_CATEGORIES - 1)
    p2_char = min(max(int(ram["p2_char"]), 0), CHAR_CATEGORIES - 1)

    frame = np.array([
        p1_hp, p2_hp, rel_x, rel_y, p1_corner_dist, p1_proj, p2_proj,
        p1_vel_x, p2_vel_x, rel_dist,
        float(ram["rel_y_dist"]), float(ram["p1_head"]), float(ram["p2_head"]),
        p1_air, p2_air,
        p1_act_hi, p2_act_hi, p1_act_lo, p2_act_lo,
        float(ram["p1_btn"]), float(ram["p2_btn"]),
        p1_char, p2_char,
    ], dtype=np.float32)

    next_track = RamTrack(prev_p1_x=p1_x, prev_p2_x=p2_x,
                          prev_p1_proj_raw=p1_proj_raw, prev_p2_proj_raw=p2_proj_raw)
    return frame, next_track, p1_sentinel, p2_sentinel


def attach_episode_spacing(info: dict, rel_dists, air_steps: int = 0) -> None:
    """Same keys and threshold as base_env._attach_episode_spacing.

    Baseline to beat (random policy, 2026-08-24 telemetry run): median 83,
    52.2% of steps at rel_dist >= 80. air_steps counts the same non-sentinel
    step set rel_dists samples, so ep_air_frac shares its denominator.
    """
    if rel_dists:
        arr = np.asarray(rel_dists, dtype=np.float32)
        info["ep_rel_dist_mean"] = float(arr.mean())
        info["ep_rel_dist_median"] = float(np.median(arr))
        info["ep_rel_dist_frac_far"] = float((arr >= FAR_DIST_THRESHOLD).mean())
        info["ep_air_frac"] = float(air_steps / len(rel_dists))


class RetroSF2Env(gymnasium.Env):
    """Headless stable-retro SF2 env speaking the v4 observation/reward contract.

    Single-agent, P1 perspective, vs the game's built-in CPU (whatever the
    loaded savestate set up). render_mode is ALWAYS None: retro.make with any
    other mode opens a pyglet window and throttles emulation to the monitor's
    vsync, turning a ~3,700 fps headless core into a 60 fps one.

    OBSERVATION PHASE: this backend samples RAM after the action's 4 frames,
    so step(a) returns the state a produced. The BizHawk rig's wire protocol
    instead delivers observations lagged one agent step behind the action
    (see base_env.step's protocol phase note). Policies transferred between
    backends therefore see a systematic one-step phase shift; evaluate
    cross-backend numbers with that in mind.
    """

    HP_SENTINEL_THRESHOLD = HP_SENTINEL_THRESHOLD

    def __init__(self, state: str = DEFAULT_STATE, trainable: bool = True,
                 verbose: bool = False, ground_gate: bool = False,
                 render_mode=None, frame_hook=None):
        # Lazy import: everything above this class must stay importable on
        # machines without stable-retro (the unit tests run there).
        import stable_retro as retro
        from stable_retro.data import Integrations

        if INTEGRATION_ROOT not in Integrations.CUSTOM_ONLY.paths:
            Integrations.add_custom_path(INTEGRATION_ROOT)

        self._retro = retro
        self._integrations = Integrations
        # obs_type=RAM skips the per-step screen blit; we never use retro's
        # observation, only the data.json variables in its info dict.
        # render_mode: None para TODO lo de entrenamiento/banco (una ventana
        # pyglet ata la emulacion al vsync del monitor: 3,700 fps -> 60);
        # "human" SOLO para el visor (tools/watch_es.py), donde 60 fps es
        # exactamente lo que se quiere.
        self._env = retro.make(
            game=GAME, state=state, inttype=Integrations.CUSTOM,
            obs_type=retro.Observations.RAM, render_mode=render_mode,
        )
        self._loaded_state = state
        self._buttons = list(self._env.buttons)
        self._frame_hook = frame_hook

        # All 63 (direction, button) products, pre-translated. Sticky can only
        # rewrite the direction bits into another DIRECTION_MAP image, so every
        # reachable pattern is cached; the .get fallback is pure paranoia.
        self._action_cache = {}
        for d in DIRECTION_MAP:
            for b in BUTTON_MAP:
                bits = tuple(DIRECTION_MAP[d] + BUTTON_MAP[b])
                self._action_cache[bits] = project_bits_to_retro(bits, self._buttons)

        self.action_space = spaces.MultiDiscrete([9, 7])
        self.observation_space = spaces.Box(
            low=np.array(V4_SINGLE_LOW * NUM_FRAMES, dtype=np.float32),
            high=np.array(V4_SINGLE_HIGH * NUM_FRAMES, dtype=np.float32),
            dtype=np.float32,
        )

        self.player = 1
        self.trainable = trainable
        self.verbose = verbose
        self.active_training_states = [state]
        self.current_state_file = state

        from collections import deque
        self.frames = deque(maxlen=NUM_FRAMES)
        self._track = RamTrack()

        self.reward_cfg = RewardConfig(ground_gate_shaping=ground_gate)
        self.reward_state = RewardState(
            prev_my_hp=176.0, prev_enemy_hp=176.0, prev_rel_dist=80.0,
            combo_counter=0, frames_since_last_hit=0,
        )

        self._steps = 0
        self.footsie_steps = 0
        self.prev_my_hp = 176.0
        self.prev_enemy_hp = 176.0
        self.prev_rel_dist = 80.0
        self.combo_counter = 0
        self.frames_since_last_hit = 0
        self.sticky_direction = None
        self.sticky_counter = 0
        self.hp_sentinel = False
        self.p1_sentinel = False
        self.p2_sentinel = False
        self.p1_ko = False
        self.p2_ko = False
        self.matches_won = 0
        self.enemy_matches_won = 0
        self.round_timer = None
        # Counter baseline, clock arming and the once-per-round edge latch all
        # live here so base_env / league_env / this backend cannot drift.
        self._round = RoundTracker()
        self._ep_rel_dists: list = []
        self._ep_air_steps = 0

        # MacroActionWrapper turns this off (exact multi-step input sequences).
        self.sticky_enabled = True

    def set_training_states(self, new_states):
        """Receives broadcast from the Main Process and updates local memory."""
        self.active_training_states = new_states

    def set_ground_gate(self, enabled: bool) -> None:
        """Flips the anti-jump shaping gate; same contract as base_env's."""
        self.reward_cfg.ground_gate_shaping = bool(enabled)

    def _get_obs(self):
        return np.concatenate(self.frames)

    def _read_ram_frame(self, is_reset=False):
        ram = self._env.data.lookup_all()
        return self._ingest(ram, is_reset=is_reset)

    def _ingest(self, ram, is_reset=False):
        frame, self._track, self.p1_sentinel, self.p2_sentinel = assemble_v4_frame(
            ram, self._track, is_reset=is_reset)
        self.hp_sentinel = self.p1_sentinel or self.p2_sentinel

        # Death flags off the SIGNED HP words -- the authoritative round result
        # (see hp_to_signed). Strictly `< 0`: HP == 0 is a live reading, and
        # treating it as death is what let round-transition frames masquerade
        # as double KOs.
        self.p1_ko = hp_to_signed(ram["p1_hp"]) < 0
        self.p2_ko = hp_to_signed(ram["p2_hp"]) < 0

        # Independent winner counters (0xFF81DA / 0xFF845A). They tick exactly
        # +1 emulator frame after the loser's HP goes negative -- confirmed on
        # every one of the 8 + 21 KOs across two live runs -- so they are a
        # cross-check on the HP-derived result, never the trigger: at
        # FRAME_SKIP=4 the sampled frame can be the death frame itself, one
        # frame before the counter moves. .get() keeps an older data.json
        # without these variables working (the cross-check just goes silent).
        self.matches_won = int(ram.get("matches_won", 0))
        self.enemy_matches_won = int(ram.get("enemy_matches_won", 0))

        # ROUND CLOCK (0xFF972A, one BCD byte, 0x99 -> 0x00). The PRIMARY
        # time-over signal: it reads 0 for 91-131 agent steps at every time
        # over, ~10 agent steps before the winner's counter moves, and it is
        # the ONLY marker of a DRAW GAME (equal HP on the buzzer), where no
        # counter ticks at all. .get() keeps an older data.json working -- the
        # clock rule just goes silent and detection falls back to the counters.
        self.round_timer = (int(ram["round_timer"])
                            if "round_timer" in ram else None)
        return frame

    def step(self, action):
        bits = discrete_to_project_bits(action)
        if not self.sticky_enabled:
            self.sticky_counter = 0
            self.sticky_direction = None
        else:
            bits, self.sticky_direction, self.sticky_counter = apply_sticky(
                bits, self.sticky_direction, self.sticky_counter)

        key = tuple(bits)
        retro_action = self._action_cache.get(key)
        if retro_action is None:
            retro_action = project_bits_to_retro(key, self._buttons)
            self._action_cache[key] = retro_action

        # FRAME_SKIP emulator frames under one held action = one agent step,
        # the same cadence the Lua client's command loop enforces. The custom
        # scenario.json has no done condition and no reward variables, so
        # retro's own terminated/reward are inert; RAM is sampled once, after
        # the last frame, exactly like the Lua client's once-per-cycle read.
        ram = None
        for _ in range(FRAME_SKIP):
            _, _, _, _, ram = self._env.step(retro_action)
            # Gancho por FRAME de emulador, no por paso de agente: quien graba
            # video necesita los 60 fps reales: muestrear una vez por paso de
            # agente da 15 fps y el juego se ve a tirones. None por defecto,
            # asi que el camino de entrenamiento no paga ni una comparacion
            # de mas... salvo esta, que es un `is not None` por frame.
            if self._frame_hook is not None:
                self._frame_hook(self._env.em.get_screen())

        observation = self._ingest(ram)
        self.frames.append(observation)

        current_my_hp, current_enemy_hp = float(observation[0]), float(observation[1])
        rel_dist = float(observation[9])
        self._steps += 1

        if rel_dist <= self.reward_cfg.peak_dist:
            self.footsie_steps += 1
        else:
            self.footsie_steps = 0

        airborne = bool(observation[13])  # frame index 13 = p1_air

        # BOTH words reading exactly 0 is the ROM blanking the bars between
        # rounds -- one side at 0 is an ordinary live reading, both at once
        # never is -- and diffing that against the last real HP invents ~+73
        # of damage out of a blank screen. Together with the sentinel flag it
        # defines "this frame's HP is not a health value", which the round
        # rules, the reward and the spacing aggregates all share so they can
        # never disagree about which frames count.
        blanked = bool(current_my_hp == 0 and current_enemy_hp == 0)
        hp_readable = not (self.hp_sentinel or blanked)

        # A round ends by KO (HP word negative, window 33-457 emulator frames,
        # so a 4-frame sampler cannot miss it) or on the CLOCK -- decisively,
        # or level, which is a DRAW GAME the counters never report. The rules
        # live in resolve_round_result and the per-episode state (counter
        # baseline, clock arming, once-per-round latch) in RoundTracker; both
        # are shared with base_env and league_env.
        #
        # The latch is what keeps a trainable=False env honest: `terminated`
        # is forced False there, so without it nothing ever consumes the
        # result and the terminal payoff is paid on every step of a window
        # that is hundreds of frames wide (measured: 1,773 payments in 2,500
        # steps, episode return -22,290).
        my_ko, enemy_ko = self._round.resolve(
            self.p1_ko, self.p2_ko,
            my_hp=current_my_hp, enemy_hp=current_enemy_hp,
            hp_readable=hp_readable,
            matches_won=self.matches_won,
            enemy_matches_won=self.enemy_matches_won,
            timer=self.round_timer)
        ko = bool(my_ko or enemy_ko)
        mw_delta, emw_delta = self._round.mw_delta, self._round.emw_delta

        # A KO frame carries a sentinel HP word (that IS the negative value),
        # but it is the single most informative frame of the round: both
        # fighters are on screen at real positions and the winner's HP is
        # intact and frozen. "Unreadable" therefore means "not a health value
        # AND no round result".
        unreadable = bool(not hp_readable and not ko)

        if not unreadable:
            self._ep_rel_dists.append(rel_dist)
            if airborne:
                self._ep_air_steps += 1

        if unreadable:
            # Do NOT diff a real previous HP against a fabricated zero. Skip
            # reward and leave reward_state untouched so the next real frame
            # diffs against the last real HP. A KO frame no longer lands here:
            # it used to, which is why the terminal payoff was never paid on
            # the frame that earned it.
            reward, reward_parts = 0.0, {}
        else:
            reward, self.reward_state, reward_parts = compute_reward(
                self.reward_state, current_my_hp, current_enemy_hp,
                rel_dist, ko, self.reward_cfg,
                airborne=airborne,
                prev_airborne=self.reward_state.prev_airborne,
                my_ko=my_ko, enemy_ko=enemy_ko,
            )
            # Mirror into the legacy attributes other modules still read.
            self.prev_my_hp = self.reward_state.prev_my_hp
            self.prev_enemy_hp = self.reward_state.prev_enemy_hp
            self.prev_rel_dist = self.reward_state.prev_rel_dist
            self.combo_counter = self.reward_state.combo_counter
            self.frames_since_last_hit = self.reward_state.frames_since_last_hit

        terminated = ko if self.trainable else False
        truncated = (bool(self._steps >= MAX_STEPS_PER_ROUND) and not terminated) \
            if self.trainable else False

        info = {
            "my_hp": current_my_hp,
            "enemy_hp": current_enemy_hp,
            "hp_sentinel": self.hp_sentinel,
            "reward_parts": reward_parts,
        }
        if terminated or truncated:
            draw = bool(terminated and my_ko and enemy_ko)
            info["draw"] = draw
            # double_ko is the legacy name for the same event; metrics_callback
            # and every saved TensorBoard run key off it, so it stays.
            info["double_ko"] = draw
            info["timeout"] = bool(truncated)
            info["episode_steps"] = self._steps
            info["win"] = 1 if (terminated and enemy_ko and not my_ko) else 0
            info["loss"] = 1 if (terminated and my_ko and not enemy_ko) else 0
            info["state_file"] = self.current_state_file
            # Winner counters as an independent audit trail. The env does NOT
            # branch on them (they lag the HP sign by one emulator frame and
            # the sampler can land on the death frame itself), but logging the
            # deltas makes a future disagreement between the two signals
            # visible instead of silent.
            info["matches_won_delta"] = mw_delta
            info["enemy_matches_won_delta"] = emw_delta
            # True when the round was decided on the clock rather than by a KO
            # (no HP word went negative). Lets the outcome rates be split by
            # cause without changing the win/loss/draw/timeout partition.
            info["time_over"] = bool(terminated
                                     and not (self.p1_ko or self.p2_ko))
            info["round_timer"] = self.round_timer
            attach_episode_spacing(info, self._ep_rel_dists, self._ep_air_steps)

        return self._get_obs(), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Same state-selection contract as base_env.reset: a random draw from
        # active_training_states, overridable per call via options={"state": name}.
        state = options.get("state") if options else None
        if state is None:
            state = random.choice(self.active_training_states)
        self.current_state_file = state

        if state != self._loaded_state:
            self._env.load_state(state, self._integrations.CUSTOM)
            self._loaded_state = state
        self._env.reset(seed=seed)

        # A savestate load is this backend's "boot": velocity and projectile
        # trackers restart exactly like the Lua script's locals do.
        self._track = RamTrack()
        observation = self._read_ram_frame(is_reset=True)

        self.prev_my_hp = float(observation[0]) if observation[0] > 0 else 176.0
        self.prev_enemy_hp = float(observation[1]) if observation[1] > 0 else 176.0

        self._steps = 0
        self.footsie_steps = 0
        self.prev_rel_dist = float(observation[9])
        self.combo_counter = 0
        self.frames_since_last_hit = 0
        self.reward_state = RewardState(
            prev_my_hp=self.prev_my_hp,
            prev_enemy_hp=self.prev_enemy_hp,
            prev_rel_dist=self.prev_rel_dist,
            combo_counter=0,
            frames_since_last_hit=0,
            prev_airborne=bool(observation[13]),
        )
        self.sticky_counter = 0
        self.sticky_direction = None
        self._ep_rel_dists = []
        self._ep_air_steps = 0
        # Re-baseline the round bookkeeping on the post-savestate-load frame
        # that _read_ram_frame just parsed: counter baseline, clock arming,
        # and -- if that frame is itself inside a KO window, i.e. the savestate
        # was captured mid-KO -- the latch, so the stale result is swallowed
        # instead of terminating the new episode on step 1.
        self._round.reset(matches_won=self.matches_won,
                          enemy_matches_won=self.enemy_matches_won,
                          timer=self.round_timer,
                          ko=bool(self.p1_ko or self.p2_ko))
        # Match base_env.reset: a fresh episode never starts FLAGGED sentinel.
        # (Clearing p1_ko/p2_ko here would be theatre -- the next _ingest
        # re-derives them from the same still-negative HP word. The latch
        # above is what actually protects step 1.)
        self.hp_sentinel = False
        self.p1_sentinel = False
        self.p2_sentinel = False

        self.frames.clear()
        for _ in range(NUM_FRAMES):
            self.frames.append(observation)

        return self._get_obs(), {}

    def close(self):
        self._env.close()
