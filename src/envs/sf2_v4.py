# sf2_v4.py
#
# V4 = V3's MultiDiscrete([9, 7]) action space with a COMPACT observation built
# on the expanded 24-field payload.
#
# V2/V3 emit 554 floats per frame, of which 512 are one-hot action IDs and 32
# are one-hot character IDs. Stacked four deep that is 2216 inputs, 2048 of them
# almost always zero, feeding a Linear(2216, 512) whose first layer is 1.13M
# parameters of mostly dead weight -- while simultaneously omitting the airborne
# flags, both players' raw inputs, and the low byte of the state word.
#
# V4 carries strictly MORE information in 23 floats per frame. Categorical IDs
# stay as raw integers and are embedded by core/sf2_extractor.py -- the standard
# treatment for large categorical game state (cf. AlphaStar, Nature 575:350-354).
#
# Frame layout (23):
#   [0-9]   continuous, unchanged from v2/v3:
#           HP(2), RelX, RelY, WallDist, ProjX(2), VelX(2), RelDist
#   [10]    rel_y_dist    (0x834E, engine-native vertical separation)
#   [11]    p1_head_clear (0x80E0)
#   [12]    p2_head_clear (0x8360)
#   [13]    p1_airborne   (0/1)
#   [14]    p2_airborne   (0/1)
#   [15-16] p1_act_hi, p2_act_hi   (0-255)
#   [17-18] p1_act_lo, p2_act_lo   (0-255, recovered low byte)
#   [19-20] p1_btn, p2_btn         (0-255, raw inputs)
#   [21-22] p1_char, p2_char       (0-15)
#
# NOTE on p2_btn (0x845E, index 20): this is the P2 controller port, and it
# reads constant 0 when training against the built-in CPU -- the CPU drives
# its character through game logic, never through the input port. Verified
# live over a 3000-step run against level-1 CPU states: every other new field
# showed real variation (rel_y_dist 100 distinct values, p1/p2_head 111/105,
# p1/p2_chest 107/108, p1_act_lo/p2_act_lo 6 each, p1_btn 0..64) while p2_btn
# showed exactly 1 distinct value across the whole run. It is regime-dependent,
# not a bad address: it carries real signal in PvP/league play, where P2's
# actions are injected via joypad.set the same way P1's are. Left in the
# layout deliberately -- see the note beside self.extra_ram in base_env.py.

import numpy as np
from gymnasium import spaces

import core.config as config
from envs.sf2_v3 import StreetFighterEnvV3
from envs.base_env import ACT_CATEGORIES, CHAR_CATEGORIES, TOTAL_OBS_DIM

V4_CONT_DIM = 13
V4_FLAG_DIM = 2
V4_ID_DIM = 8
V4_FRAME_DIM = V4_CONT_DIM + V4_FLAG_DIM + V4_ID_DIM  # 23


class StreetFighterEnvV4(StreetFighterEnvV3):
    """V3 action space, compact ID-based observation over the wide payload."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

    def _parse_payload(self, data, is_reset=False):
        """Reuses V3's parser, then rebuilds a compact frame from it.

        Keeping V3's parser as the single source of truth for the wire format
        means the perspective flip, velocity deltas and sentinel handling stay
        in exactly one place.
        """
        full = super()._parse_payload(data, is_reset=is_reset)

        # The base failsafe repeats the last good frame verbatim. For v4 that
        # frame is already compact (V4_FRAME_DIM), not the 554-dim v2/v3 layout,
        # so re-running the one-hot argmax extraction on it would read empty
        # slices and raise. Pass it through unchanged.
        if full.shape[0] != TOTAL_OBS_DIM:
            return full.astype(np.float32)

        x = self.extra_ram

        base_cont = full[:10]
        act_start = 10
        p1_act = float(np.argmax(full[act_start:act_start + ACT_CATEGORIES]))
        p2_act = float(np.argmax(
            full[act_start + ACT_CATEGORIES:act_start + 2 * ACT_CATEGORIES]))
        char_start = act_start + 2 * ACT_CATEGORIES
        p1_char = float(np.argmax(full[char_start:char_start + CHAR_CATEGORIES]))
        p2_char = float(np.argmax(
            full[char_start + CHAR_CATEGORIES:char_start + 2 * CHAR_CATEGORIES]))

        tail = np.array([
            float(x.get("rel_y_dist", 0)),
            float(x.get("p1_head", 192)),
            float(x.get("p2_head", 192)),
            float(x.get("p1_air", 0)),
            float(x.get("p2_air", 0)),
            p1_act, p2_act,
            float(x.get("p1_act_lo", 0)), float(x.get("p2_act_lo", 0)),
            # p2_btn (0x845E) reads constant 0 vs. the built-in CPU (it drives
            # its character through game logic, not the input port) -- see the
            # module docstring above for the 3000-step measurement. Real
            # signal only in PvP/league play; kept for layout stability.
            float(x.get("p1_btn", 0)), float(x.get("p2_btn", 0)),
            p1_char, p2_char,
        ], dtype=np.float32)

        return np.concatenate((base_cont, tail)).astype(np.float32)
