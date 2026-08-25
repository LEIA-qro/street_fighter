# action_macros.py
#
# Temporally-extended action vocabulary for Street Fighter II.
#
# WHY THIS EXISTS
# ---------------
# With MultiDiscrete([9, 7]) at FRAME_SKIP=4, a Hadouken is three consecutive
# correct agent steps (Down, Down-Forward, Forward + Punch). That is ~1 in 1,700
# under a uniform policy, so it DOES happen -- the problem is not reachability
# but credit assignment:
#
#   1. The two setup steps have negative local advantage. Crouching in neutral
#      whiffs and gets counter-poked, and it occurs constantly in non-Hadouken
#      contexts where it is genuinely bad, so its average advantage stays
#      negative and PPO suppresses it.
#   2. MultiDiscrete puts direction in ONE 9-way softmax, so raising P(Down)
#      for step 1 mechanically lowers P(Down-Right) and P(Right) for steps 2
#      and 3. The three steps of a single motion compete inside one head.
#      (MultiBinary in v1/v2 factorized into independent Bernoullis, whose
#      marginals reinforce each other -- the likely reason v3 regressed here.)
#
# Collapsing the motion into one atomic action removes both: no setup steps in
# the rollout buffer to suppress, no intra-head competition. This is the
# options / macro-action formulation of Sutton, Precup & Singh, "Between MDPs
# and semi-MDPs: A framework for temporal abstraction in reinforcement
# learning", Artificial Intelligence 112(1-2):181-211, 1999.
#
# Index space:
#   [0, 63)              primitives, identical to the v3 MultiDiscrete([9,7]) grid
#   [63, 63 + n_macros)  macros, executed over len(macro) consecutive agent steps

from typing import List, Tuple

N_DIRECTIONS = 9
N_BUTTONS = 7
N_PRIMITIVES = N_DIRECTIONS * N_BUTTONS  # 63

# Direction indices, matching DIRECTION_MAP in envs/sf2_v3.py:
#   0 neutral   1 Up      2 Down     3 Left     4 Right
#   5 Up+Left   6 Up+Right  7 Down+Left  8 Down+Right
#
# Button indices, matching BUTTON_MAP in envs/sf2_v3.py:
#   0 none  1 A(LK)  2 B(MK)  3 C(HK)  4 X(LP)  5 Y(MP)  6 Z(HP)

_MIRROR = {0: 0, 1: 1, 2: 2, 3: 4, 4: 3, 5: 6, 6: 5, 7: 8, 8: 7}

# All macros are written FACING RIGHT (opponent to the agent's right) and are
# mirrored on decode when the agent is on the right side of the screen.
MACROS = {
    # Hadouken -- quarter-circle forward + punch
    "hadouken_lp": [(2, 0), (8, 0), (4, 4)],
    "hadouken_hp": [(2, 0), (8, 0), (4, 6)],
    # Shoryuken -- forward, down, down-forward + punch
    "shoryuken_lp": [(4, 0), (2, 0), (8, 4)],
    "shoryuken_hp": [(4, 0), (2, 0), (8, 6)],
    # Tatsumaki Senpukyaku -- quarter-circle back + kick
    "tatsumaki_lk": [(2, 0), (7, 0), (3, 1)],
    "tatsumaki_mk": [(2, 0), (7, 0), (3, 2)],
    # Common movement macros that also span several steps
    "jump_forward": [(6, 0), (6, 0)],
    "jump_back": [(5, 0), (5, 0)],
    "dash_block": [(3, 0), (3, 0), (3, 0)],
}

MACRO_NAMES: List[str] = list(MACROS.keys())
N_ACTIONS = N_PRIMITIVES + len(MACRO_NAMES)


def mirror_direction(direction: int) -> int:
    """Swaps the left/right component of a direction index."""
    return _MIRROR[direction]


def mirror_macro(steps: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Mirrors a macro's directions while leaving its buttons untouched."""
    return [(mirror_direction(d), b) for d, b in steps]


def decode(action: int, facing_right: bool) -> List[Tuple[int, int]]:
    """Expands one action index into the (direction, button) steps to execute.

    Primitives return a single step. Macros return len(macro) steps, mirrored
    when the agent is facing left.
    """
    if not 0 <= action < N_ACTIONS:
        raise ValueError(
            f"action {action} out of range for action space of size {N_ACTIONS}"
        )
    if action < N_PRIMITIVES:
        # Same divmod bijection the DQN wrapper uses: btn = a % 7, dir = a // 7.
        return [(action // N_BUTTONS, action % N_BUTTONS)]
    steps = MACROS[MACRO_NAMES[action - N_PRIMITIVES]]
    return list(steps) if facing_right else mirror_macro(steps)
