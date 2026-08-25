# test_dqn_config.py
#
# Guards the DQN replay buffer size against reallocating more memory than a
# machine running 16 live EmuHawk.exe instances can spare.

import os
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from agents.dqn.config import BUFFER_SIZE
from envs.base_env import TOTAL_OBS_DIM
import core.config as config


def test_replay_buffer_footprint_fits_alongside_16_live_emulators():
    """SB3's ReplayBuffer.__init__ allocates (buffer_size // n_envs, n_envs,
    *obs_shape) for BOTH observations and next_observations
    (optimize_memory_usage defaults to False), so n_envs cancels out of the
    total footprint. Against the v2/v3 observation (554 floats x 4 stacked
    frames = 2216 float32) the old BUFFER_SIZE=1_000_000 allocated:

        1_000_000 x 2216 x 4 bytes x 2 = 17.7 GB

    which cannot start alongside 16 live emulators on any normal machine.
    """
    obs_floats = TOTAL_OBS_DIM * config.NUM_FRAMES
    footprint_gb = (BUFFER_SIZE * obs_floats * 4 * 2) / 1e9
    assert footprint_gb < 6.0, (
        f"DQN replay buffer would allocate {footprint_gb:.1f} GB "
        f"(BUFFER_SIZE={BUFFER_SIZE}); this cannot start alongside 16 live "
        "emulators"
    )
