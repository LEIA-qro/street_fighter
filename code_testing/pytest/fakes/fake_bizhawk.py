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


def make_payload(p1_hp, p2_hp, p1_x=100, p2_x=200, p1_y=0, p2_y=0,
                 p1_act=0, p2_act=0, p1_proj=-1, p2_proj=-1,
                 p1_char=0, p2_char=1, rel_dist=100) -> str:
    """Builds one 13-field CSV payload in the exact Lua wire format.

    Field order matches lua/v2.0/training_env_client.lua:112 —
    hp1, hp2, x1, x2, y1, y2, act1, act2, proj1, proj2, char1, char2, rel_dist
    """
    fields = [p1_hp, p2_hp, p1_x, p2_x, p1_y, p2_y,
              p1_act, p2_act, p1_proj, p2_proj, p1_char, p2_char, rel_dist]
    return "0 " + ",".join(str(int(f)) for f in fields)


class FakeBizHawkEnv(StreetFighterEnvV3):
    """StreetFighterEnvV3 with the emulator bridge stubbed out."""

    def __init__(self, payloads, **kwargs):
        self._queue = list(payloads)
        self.sent = []
        # Bypass StreetFighterEnvV3.__init__ -> ... -> BizHawkBaseEnv.__init__,
        # which would bind a socket and launch EmuHawk.exe.
        self._bootstrap_without_emulator(**kwargs)

    def _bootstrap_without_emulator(self, player=1, trainable=True):
        import gymnasium as gym
        from collections import deque
        from gymnasium import spaces
        import numpy as np
        import core.config as config
        from envs.base_env import ONE_HOT_ACT_DIM, ONE_HOT_CHAR_DIM

        gym.Env.__init__(self)

        # --- BizHawkBaseEnv fields ---
        self.bizhawk_path = None
        self.rom_path = None
        self.lua_path = None
        self.host, self.port = config.HOST, config.PORT
        self.trainable = trainable
        self.verbose = False
        self.debug_mode = False
        self.step_count = 0
        self.step_debug_interval = 10 ** 9
        self.server_socket = None
        self.conn = None
        self.emulator_process = None
        self.stream_buffer = ""

        # --- StreetFighterBaseEnv fields ---
        self.player = player
        self.active_training_states = ["FAKE.State"]
        self.prev_my_hp = 176
        self.prev_enemy_hp = 176
        self.prev_p1_x = 0
        self.prev_p2_x = 0
        self.frames = deque(maxlen=config.NUM_FRAMES)
        self._steps = 0
        self.footsie_steps = 0
        self.prev_rel_dist = 80.0
        self.combo_counter = 0
        self.frames_since_last_hit = 0
        self.corrupt_payload_count = 0
        self.sticky_direction = None
        self.sticky_counter = 0
        self.hp_sentinel = False

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

    # --- Socket layer replacement ---

    def queue(self, payloads):
        """Appends payloads the fake will return from receive_payload()."""
        self._queue.extend(payloads)

    def send_command(self, command: str):
        self.sent.append(command)

    def receive_payload(self) -> str:
        if not self._queue:
            raise AssertionError(
                "FakeBizHawkEnv ran out of scripted payloads. "
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
            "FakeBizHawkEnv must never launch a real emulator. "
            "Something reached the self-healing respawn path in "
            "StreetFighterBaseEnv.reset()."
        )
