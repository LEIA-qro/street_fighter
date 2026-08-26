# fake_bizhawk.py
#
# A StreetFighterEnvV3 whose socket layer is replaced by a scripted list of
# payload strings. Lets the reward, termination and parsing logic be tested
# with no EmuHawk.exe, no TCP socket and no ROM file.

import os
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from envs.sf2_v3 import StreetFighterEnvV3
from envs.sf2_v4 import StreetFighterEnvV4, V4_CONT_DIM, V4_FLAG_DIM, V4_ID_DIM


def make_payload(p1_hp, p2_hp, p1_x=100, p2_x=200, p1_y=0, p2_y=0,
                 p1_act=0, p2_act=0, p1_proj=-1, p2_proj=-1,
                 p1_char=0, p2_char=1, rel_dist=100,
                 extended=False,
                 p1_act_lo=0, p2_act_lo=0, p1_btn=0, p2_btn=0,
                 p1_air=0, p2_air=0, rel_y_dist=0,
                 p1_chest=192, p1_head=192, p2_chest=192, p2_head=192) -> str:
    """Builds a payload in the exact Lua wire format.

    13 fields by default (legacy), 24 when extended=True. Field order matches
    lua/v2.0/training_env_client.lua.
    """
    fields = [p1_hp, p2_hp, p1_x, p2_x, p1_y, p2_y,
              p1_act, p2_act, p1_proj, p2_proj, p1_char, p2_char, rel_dist]
    if extended:
        fields += [p1_act_lo, p2_act_lo, p1_btn, p2_btn, p1_air, p2_air,
                   rel_y_dist, p1_chest, p1_head, p2_chest, p2_head]
    return "0 " + ",".join(str(int(f)) for f in fields)


def _bootstrap_common_fields(env, player, trainable, ground_gate=False):
    """Sets every BizHawkBaseEnv / StreetFighterBaseEnv field the real
    __init__ chain would set, minus observation_space/action_space (those
    differ per obs layout and are set by each fake subclass).
    """
    import gymnasium as gym
    from collections import deque
    import core.config as config
    from envs.reward import RewardConfig, RewardState

    gym.Env.__init__(env)

    # --- BizHawkBaseEnv fields ---
    env.bizhawk_path = None
    env.rom_path = None
    env.lua_path = None
    env.host, env.port = config.HOST, config.PORT
    env.trainable = trainable
    env.verbose = False
    env.debug_mode = False
    env.step_count = 0
    env.step_debug_interval = 10 ** 9
    env.server_socket = None
    env.conn = None
    env.emulator_process = None
    env.stream_buffer = ""

    # --- StreetFighterBaseEnv fields ---
    env.player = player
    env.active_training_states = ["FAKE.State"]
    env.prev_my_hp = 176
    env.prev_enemy_hp = 176
    env.prev_p1_x = 0
    env.prev_p2_x = 0
    env.frames = deque(maxlen=config.NUM_FRAMES)
    env._steps = 0
    env.footsie_steps = 0
    env.prev_rel_dist = 80.0
    env.combo_counter = 0
    env.frames_since_last_hit = 0
    env.corrupt_payload_count = 0
    env.sticky_direction = None
    env.sticky_counter = 0
    env.sticky_enabled = True
    env.hp_sentinel = False
    env.p1_sentinel = False
    env.p2_sentinel = False
    env.extra_ram = {}
    env._ep_rel_dists = []
    env._ep_air_steps = 0
    env.reward_cfg = RewardConfig(ground_gate_shaping=ground_gate)
    env.reward_state = RewardState(
        prev_my_hp=176.0, prev_enemy_hp=176.0, prev_rel_dist=80.0,
        combo_counter=0, frames_since_last_hit=0,
    )


# The payload the fake pretends Lua sent unsolicited at boot. The real client
# sends its telemetry BEFORE waiting for a command, so exactly one payload is
# always pending when reset() runs; reset() drains it and then reads the real
# post-savestate-load frame. Every fake starts with this in its queue, so
# `FakeBizHawkEnv([make_payload(...)])` still means "reset() returns an
# observation built from make_payload(...)". A test that resets a second time
# must queue TWO payloads for that reset: the in-flight stale one, then the
# post-load one.
BOOT_STALE_PAYLOAD = make_payload(176, 176)


class _FakeSocketLayerMixin:
    """Socket layer replacement shared by every FakeBizHawkEnv* variant."""

    def queue(self, payloads):
        """Appends payloads the fake will return from receive_payload()."""
        self._queue.extend(payloads)

    def send_command(self, command: str):
        self.sent.append(command)

    def receive_payload(self) -> str:
        if not self._queue:
            raise AssertionError(
                f"{type(self).__name__} ran out of scripted payloads. "
                "Call env.queue([...]) before stepping."
            )
        return self._queue.pop(0)

    def close(self):
        pass

    def _start_emulator_bridge(self):
        # StreetFighterBaseEnv.reset() has a self-healing retry path that
        # catches RuntimeError/OSError and calls this to respawn the
        # emulator. If a test ever triggers that path, fail loudly instead
        # of silently reaching socket.bind()/subprocess.Popen(EmuHawk.exe).
        raise AssertionError(
            f"{type(self).__name__} must never launch a real emulator. "
            "Something reached the self-healing respawn path in "
            "StreetFighterBaseEnv.reset()."
        )


class FakeBizHawkEnv(_FakeSocketLayerMixin, StreetFighterEnvV3):
    """StreetFighterEnvV3 with the emulator bridge stubbed out."""

    def __init__(self, payloads, **kwargs):
        self._queue = [BOOT_STALE_PAYLOAD] + list(payloads)
        self.sent = []
        # Bypass StreetFighterEnvV3.__init__ -> ... -> BizHawkBaseEnv.__init__,
        # which would bind a socket and launch EmuHawk.exe.
        self._bootstrap_without_emulator(**kwargs)

    def _bootstrap_without_emulator(self, player=1, trainable=True, ground_gate=False):
        from gymnasium import spaces
        import numpy as np
        import core.config as config
        from envs.base_env import ONE_HOT_ACT_DIM, ONE_HOT_CHAR_DIM

        _bootstrap_common_fields(self, player, trainable, ground_gate)

        # --- Spaces (mirrors sf2_v2 / sf2_v3) ---
        cont_low = [0., 0., -500., -200., 0., -1., -1., -100., -100., 0.]
        cont_high = [176., 176., 500., 200., 250., 500., 500., 100., 100., 187.]
        single_low = cont_low + [0.] * (ONE_HOT_ACT_DIM + ONE_HOT_CHAR_DIM)
        single_high = cont_high + [1.] * (ONE_HOT_ACT_DIM + ONE_HOT_CHAR_DIM)
        self.observation_space = spaces.Box(
            low=np.array(single_low * config.NUM_FRAMES, dtype=np.float32),
            high=np.array(single_high * config.NUM_FRAMES, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([9, 7])


class FakeBizHawkEnvV4(_FakeSocketLayerMixin, StreetFighterEnvV4):
    """StreetFighterEnvV4 with the emulator bridge stubbed out."""

    def __init__(self, payloads, **kwargs):
        self._queue = [BOOT_STALE_PAYLOAD] + list(payloads)
        self.sent = []
        # Bypass StreetFighterEnvV4.__init__ -> ... -> BizHawkBaseEnv.__init__,
        # which would bind a socket and launch EmuHawk.exe.
        self._bootstrap_without_emulator(**kwargs)

    def _bootstrap_without_emulator(self, player=1, trainable=True, ground_gate=False):
        from gymnasium import spaces
        import numpy as np
        import core.config as config
        from envs.base_env import ACT_CATEGORIES, CHAR_CATEGORIES

        _bootstrap_common_fields(self, player, trainable, ground_gate)

        # --- Spaces (mirrors sf2_v4's compact 23-dim frame) ---
        cont_low = [0., 0., -500., -200., 0., -1., -1., -100., -100., 0.,
                    0., 0., 0.]
        cont_high = [176., 176., 500., 200., 250., 500., 500., 100., 100., 187.,
                     255., 192., 192.]
        flag_low, flag_high = [0., 0.], [1., 1.]
        id_low = [0.] * V4_ID_DIM
        id_high = [float(ACT_CATEGORIES - 1)] * 6 + [float(CHAR_CATEGORIES - 1)] * 2
        single_low = cont_low + flag_low + id_low
        single_high = cont_high + flag_high + id_high
        self.observation_space = spaces.Box(
            low=np.array(single_low * config.NUM_FRAMES, dtype=np.float32),
            high=np.array(single_high * config.NUM_FRAMES, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([9, 7])
