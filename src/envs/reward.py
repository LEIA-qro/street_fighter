# reward.py
#
# Pure reward function for Street Fighter II. No environment, no sockets, no
# global state -- everything the function needs arrives as arguments and
# everything it changes leaves as a new RewardState.
#
# The distance term is potential-based shaping in the form
#     F(s, s') = gamma * Phi(s') - Phi(s)
# which is policy-invariant (Ng, Harada & Russell, "Policy invariance under
# reward transformations", ICML 1999).

from dataclasses import dataclass, replace

from core.rl_constants import AGENT_GAMMA


@dataclass
class RewardConfig:
    damage_clamp: float = 100.0
    damage_scale: float = 1.0
    damage_taken_penalty: float = 0.77
    combo_window: int = 6
    combo_step: float = 0.5
    combo_cap: float = 4.0
    # Was 0.015, which cost ~-8.6 over a measured 570-step round -- comparable
    # to landing a light punch, and strong pressure to end rounds fast.
    time_penalty: float = 0.002
    win_bonus: float = 65.0
    loss_penalty: float = 50.0
    # Spacing potential. peak_dist sits in Ryu's poke range where combos start;
    # max_dist is the measured saturation point of rel_dist (0x834C).
    peak_dist: float = 70.0
    max_dist: float = 187.0
    spacing_weight: float = 2.5
    # Must equal the acting agent's discount -- see core/rl_constants.py's
    # docstring for why a mismatch breaks the potential-based shaping
    # guarantee. Do NOT hardcode a second literal here; if the agent's
    # gamma ever needs to differ from AGENT_GAMMA, update the shared
    # constant (or pass gamma= explicitly at both call sites) rather than
    # letting the two drift apart again.
    gamma: float = AGENT_GAMMA


@dataclass
class RewardState:
    prev_my_hp: float
    prev_enemy_hp: float
    prev_rel_dist: float
    combo_counter: int
    frames_since_last_hit: int


def spacing_potential(dist: float, cfg: RewardConfig) -> float:
    """Phi(s): peaks at poke range, decays linearly toward both extremes.

    THE OLD VERSION WAS `0.05 * max(0, 1 - d/80)`, which is identically zero
    for every d >= 80. Telemetry measured 52.2% of all training steps in that
    range, with a median distance of 83 -- so over half of training had no
    shaping gradient at all, and crossing from 187 to 81 earned exactly
    nothing. Against a flat time penalty, closing distance was net-negative
    and the policy correctly learned to stand still.

    Two-sided rather than monotone on purpose: a "closer is always better"
    potential teaches rushdown, but Ryu wins by holding a spacing band.

    The magnitude is safe to tune freely. Potential-based shaping is
    policy-invariant for ANY Phi and any coefficient (Ng, Harada & Russell,
    ICML 1999) because gamma*Phi(s') - Phi(s) telescopes; scaling changes the
    learning dynamics, never the optimum.
    """
    d = min(max(dist, 0.0), cfg.max_dist)
    if d <= cfg.peak_dist:
        return cfg.spacing_weight * (d / cfg.peak_dist)
    return cfg.spacing_weight * (cfg.max_dist - d) / (cfg.max_dist - cfg.peak_dist)


def compute_reward(state: RewardState, my_hp: float, enemy_hp: float,
                   rel_dist: float, terminated: bool,
                   cfg: RewardConfig) -> tuple[float, RewardState, dict]:
    """Returns (total_reward, next_state, component_breakdown).

    The component dict always sums exactly to total_reward, which makes both
    the unit tests and the TensorBoard breakdown trustworthy.
    """
    damage_dealt = min(max(0.0, state.prev_enemy_hp - enemy_hp), cfg.damage_clamp)
    damage_taken = min(max(0.0, state.prev_my_hp - my_hp), cfg.damage_clamp)

    combo_counter = state.combo_counter
    frames_since_last_hit = state.frames_since_last_hit

    if damage_dealt > 0:
        if frames_since_last_hit == 0:
            pass  # continuous damage on consecutive steps is one hit
        elif frames_since_last_hit <= cfg.combo_window:
            combo_counter += 1
        else:
            combo_counter = 1
        frames_since_last_hit = 0
        combo = min(combo_counter * cfg.combo_step, cfg.combo_cap)
        time = 0.0
    else:
        frames_since_last_hit += 1
        if frames_since_last_hit > cfg.combo_window:
            combo_counter = 0
        combo = 0.0
        time = -cfg.time_penalty

    shaping = (cfg.gamma * spacing_potential(rel_dist, cfg)
               - spacing_potential(state.prev_rel_dist, cfg))

    terminal = 0.0
    if terminated:
        if enemy_hp <= 0:
            terminal += cfg.win_bonus
        if my_hp <= 0:
            terminal -= cfg.loss_penalty

    components = {
        "damage": cfg.damage_scale * damage_dealt,
        "taken": -cfg.damage_taken_penalty * damage_taken,
        "combo": combo,
        "shaping": shaping,
        "time": time,
        "terminal": terminal,
    }

    next_state = replace(
        state,
        prev_my_hp=my_hp,
        prev_enemy_hp=enemy_hp,
        prev_rel_dist=rel_dist,
        combo_counter=combo_counter,
        frames_since_last_hit=frames_since_last_hit,
    )
    return sum(components.values()), next_state, components
