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


def hp_to_signed(raw) -> int:
    """Decodes one raw HP word as the SIGNED 16-bit value the ROM stores.

    Lives here because it is the only pure module both backends can import
    (base_env pulls in core.config, which is fatal off-Windows; retro_env
    therefore duplicates every other base_env constant rather than import it).
    Death detection has to agree bit-for-bit across the two rigs, so it gets
    exactly one implementation.

    MEASURED on this ROM (Street Fighter II' SCE, Genesis, 0xFF8042 / 0xFF82C2;
    several independent live runs totalling >500,000 emulator frames): the HP
    words hold 0..176 when alive and a SMALL NEGATIVE number when dead. The
    negative set is NOT just -1 -- {-1, -4, -5, -6, -7, -8, -9, -10, -11, -13,
    -27} have all been observed, because the killing blow's damage is
    subtracted past zero before the ROM freezes the word. An earlier revision
    of this docstring claimed "only 0..176 and -1, not a single reading in
    177..65525"; that is FALSE as stated -- read unsigned, -13 is 65523 and
    -27 is 65509, both inside that interval. The invariant that actually holds,
    and the only one this module relies on, is:

        alive  =>  0 <= hp <= 176        (never 177..32767)
        dead   =>  -256 < hp < 0         (i.e. 65280..65535 unsigned)

    so the SIGN separates the two classes with a wide margin either way. Do NOT
    harden any death test to `== -1` / `== 65535` on the strength of a comment:
    it would miss most KOs. Likewise HP_SENTINEL_THRESHOLD (200) survives only
    as an obs clamp for synthetic/corrupt values; it is not a death test.

    So the "HP sentinel" the code was built around was never a sentinel at all
    -- it was a negative HP read through the wrong type, i.e. the KO signal
    itself, inverted into "this frame is unreadable, refuse to terminate".

    Both transports deliver the same 16 bits and both need this decode:
      * BizHawk/Lua sends `mainmemory.read_u16_be` UNSIGNED over the socket,
        so a KO arrives as 65535.
      * stable-retro types the field in data.json; the repo's custom
        integration said ">u2" (fixed to ">i2" alongside this change), while
        the integration shipped with stable-retro has always said ">i2".
    Normalizing here makes the death test independent of which one produced
    the number, and keeps working if either side is re-typed later.
    """
    value = int(raw)
    return value - 65536 if value > 32767 else value


def resolve_round_result(my_ko: bool, enemy_ko: bool,
                         my_award_delta: int = 0,
                         enemy_award_delta: int = 0,
                         time_over: bool = False,
                         my_hp: float | None = None,
                         enemy_hp: float | None = None) -> tuple[bool, bool]:
    """Who lost the round: (my_loss, enemy_loss). Shared by both backends.

    A SF2 round ends in one of THREE ways, and only the first is visible in HP:

    1. KO -- the loser's HP word goes negative and stays there for >= 33
       emulator frames (measured min over 52 KO windows; median 33, max 457).
       Exact on the frame it fires, so it wins every disagreement below.
    2. TIME OVER, DECISIVE -- the round clock expires and the ROM awards the
       round to whoever has more health, ticking that side's win counter.
       NEITHER HP word ever goes negative, so HP alone is blind to it.
    3. TIME OVER, EQUAL HP ("DRAW GAME") -- the clock expires with both bars
       exactly level. The ROM ends the round and ticks NEITHER counter.
       Measured live (both HP pinned to 120, clock allowed to run out): the
       round ended, the bars refilled ~95 agent steps later, and both counters
       stayed at 0. An env that watches only HP and the counters does not
       terminate at all here -- it plays a whole extra round and truncates at
       MAX_STEPS_PER_ROUND as a TIMEOUT worth 0, which is precisely the Run A
       pathology this whole change set exists to kill.

    Case 3 is why the ROUND CLOCK (0xFF972B, BCD, 0x99 -> 0x00) is the primary
    time-over signal and the counters are only the fallback:

      * the clock reads 0 for 91-131 agent steps (364-524 emulator frames) at
        every time over, so no sampling cadence can miss it;
      * it fires ~10 agent steps EARLIER than the winner's counter does;
      * it is present on a DRAW GAME, where no counter ever moves;
      * with the clock at 0 the outcome is decided by comparing the two live
        HP words -- both fighters cap at 176, so "more health" and "higher
        percentage" are the same comparison -- and equal HP is a genuine draw.

    The caller is responsible for only asserting `time_over` on a frame whose
    HP words are actually readable (not sentinel, not the [0, 0] blank the ROM
    paints between rounds) and only after the clock has been seen RUNNING in
    this episode -- the clock also reads 0 on the inter-match/continue screens.

    Award deltas are measured from the episode's reset, never absolute: the
    counters are monotone within a match but RESET to 0 when a match ends,
    which makes a delta negative -- and only a strictly positive delta may
    award a round. Both ticking at once resolves to a draw. Every optional
    argument defaults to "absent", so a transport that can supply neither the
    clock nor the counters (the 24-field BizHawk payload, at time of writing)
    degrades cleanly to KO-only detection.
    """
    if my_ko or enemy_ko:
        return bool(my_ko), bool(enemy_ko)
    if time_over and my_hp is not None and enemy_hp is not None:
        if my_hp > enemy_hp:
            return False, True          # I had more health: the enemy lost.
        if enemy_hp > my_hp:
            return True, False
        return True, True               # DRAW GAME: level bars on the buzzer.
    if my_award_delta > 0 or enemy_award_delta > 0:
        # "my award" means I won the round, i.e. the OPPONENT lost it.
        return enemy_award_delta > 0, my_award_delta > 0
    return False, False


class RoundTracker:
    """Per-episode round-result bookkeeping. Shared by ALL THREE backends.

    The round rules themselves live in resolve_round_result(); this owns the
    three pieces of per-episode STATE around them, each of which produced the
    same class of bug independently in base_env, retro_env and league_env:

    1. **Counter baseline.** The two round-win counters are absolute, monotone
       within a match, and reset to 0 between matches. Only the delta since
       this episode's reset may award a round, and the baseline has to follow
       a counter DOWN across a match boundary or the delta stays negative and
       silently disables counter detection for the rest of the episode.

    2. **Clock arming.** The round clock reads 0 at a TIME OVER *and* on the
       inter-match / continue screens. It may only decide a round after it has
       been seen RUNNING in this episode and on a frame whose HP is readable.

    3. **The edge latch -- the important one.** A round result is not an event,
       it is a STATE that stays asserted for a long time: the KO window is
       33-457 emulator frames and the clock reads 0 for 364-524. On an env
       with trainable=False (model eval, AI-vs-AI, any harness that wants a
       continuous stream) `terminated` is forced False, so nothing consumes
       the result -- and the pre-latch code paid win_bonus/loss_penalty on
       EVERY step of that window. Measured on the live core before this class
       existed: 1,773 terminal payments in 2,500 steps, episode return
       -22,290. The counter path was worse still: its delta never returns to
       zero, so ONE time over paid a terminal on every remaining step of the
       run, forever. The latch pays a result exactly once and re-arms only
       when the result clears, which is what the pre-fix code accidentally got
       right (a sentinel self-cleared after one frame).

    It also fixes the mid-KO savestate hole for free: reset(ko=True) starts
    latched, so a savestate captured inside a KO window cannot terminate the
    new episode on step 1. (base_env used to "clear" p1_ko/p2_ko in reset()
    for that, which did nothing at all -- step() re-derives them from the same
    still-negative HP word on the very next payload.)
    """

    __slots__ = ("_mw_base", "_emw_base", "_latched", "_clock_armed",
                 "mw_delta", "emw_delta", "time_over", "suppressed")

    def __init__(self) -> None:
        self.reset()

    def reset(self, matches_won: int = 0, enemy_matches_won: int = 0,
              timer: int | None = None, ko: bool = False) -> None:
        """Re-baselines on the post-savestate-load frame."""
        self._mw_base = int(matches_won)
        self._emw_base = int(enemy_matches_won)
        self._latched = bool(ko)
        self._clock_armed = bool(timer is not None and timer > 0)
        self.mw_delta = 0
        self.emw_delta = 0
        self.time_over = False
        self.suppressed = False

    def resolve(self, my_ko: bool, enemy_ko: bool, *,
                my_hp: float | None = None, enemy_hp: float | None = None,
                hp_readable: bool = True,
                matches_won: int = 0, enemy_matches_won: int = 0,
                timer: int | None = None) -> tuple[bool, bool]:
        """(my_loss, enemy_loss) for THIS frame, at most once per round.

        Every optional argument absent => KO-only detection, which is exactly
        what the 24-field BizHawk payload deployed on the rig today supports.
        """
        matches_won, enemy_matches_won = int(matches_won), int(enemy_matches_won)
        # Follow a counter DOWN across a match boundary; never up (that is the
        # award we are trying to detect).
        self._mw_base = min(self._mw_base, matches_won)
        self._emw_base = min(self._emw_base, enemy_matches_won)
        self.mw_delta = matches_won - self._mw_base
        self.emw_delta = enemy_matches_won - self._emw_base

        if timer is not None and timer > 0:
            self._clock_armed = True
        self.time_over = bool(timer is not None and timer == 0
                              and self._clock_armed and hp_readable)

        my_loss, enemy_loss = resolve_round_result(
            my_ko, enemy_ko,
            my_award_delta=self.mw_delta, enemy_award_delta=self.emw_delta,
            time_over=self.time_over, my_hp=my_hp, enemy_hp=enemy_hp)

        if not (my_loss or enemy_loss):
            self._latched = False
            self.suppressed = False
            return False, False

        # A result is asserted: swallow the counter delta for the rest of the
        # window so it cannot keep re-firing after the HP words recover, and
        # report the result to exactly one caller.
        self._mw_base, self._emw_base = matches_won, enemy_matches_won
        if self._latched:
            self.suppressed = True
            return False, False
        self._latched = True
        self.suppressed = False
        return my_loss, enemy_loss


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
    # Payoff for a DRAW (both fighters KO'd on the same frame). This used to
    # not exist: the terminal block ran two INDEPENDENT `if`s, so a double KO
    # collected win_bonus AND loss_penalty and settled at +65 - 50 = +15 --
    # NET POSITIVE, and +65 better than losing cleanly. Break-even against
    # fighting for the win was p(win) = (draw - (-loss)) / (win + loss)
    # = 65/115 = 0.565, and Run A's measured win share among decisive rounds
    # was 0.566: the outcome channel contributed almost exactly ZERO gradient
    # toward winning. The three outcomes are now mutually exclusive branches
    # (see compute_reward) and a draw is scored as "you did not win", never as
    # the algebraic sum of a bonus and a penalty.
    draw_penalty: float = 50.0
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
    # Anti-jump gate. When True, the spacing potential is defined over the
    # extended state (dist, airborne): Phi = spacing_potential(dist) while
    # grounded, 0 while airborne. Jump-approach then stops collecting shaping
    # (leaving the ground forfeits the accumulated Phi; walking keeps it).
    # This is still PURE potential-based shaping, just over the extended
    # state space, so the policy-invariance guarantee (Ng, Harada & Russell,
    # ICML 1999) holds unchanged -- gamma*Phi(s') - Phi(s) telescopes over
    # (dist, air) exactly as it did over dist alone. Default False =>
    # bit-identical rewards for every existing caller.
    ground_gate_shaping: bool = False


@dataclass
class RewardState:
    prev_my_hp: float
    prev_enemy_hp: float
    prev_rel_dist: float
    combo_counter: int
    frames_since_last_hit: int
    # Carrier for the airborne half of the extended shaping state: written by
    # compute_reward (next_state.prev_airborne = this step's airborne) so the
    # env can hand it back as the prev_airborne argument on the next step.
    # Defaulted so existing positional constructions keep working.
    prev_airborne: bool = False


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
                   cfg: RewardConfig, airborne: bool = False,
                   prev_airborne: bool = False,
                   my_ko: bool | None = None,
                   enemy_ko: bool | None = None) -> tuple[float, RewardState, dict]:
    """Returns (total_reward, next_state, component_breakdown).

    The component dict always sums exactly to total_reward, which makes both
    the unit tests and the TensorBoard breakdown trustworthy.

    airborne / prev_airborne only matter under cfg.ground_gate_shaping; the
    arguments are authoritative (state.prev_airborne is just the carrier the
    env reads prev_airborne back out of between steps). Both default False, so
    every pre-gate call site computes exactly what it always did.

    my_ko / enemy_ko carry the AUTHORITATIVE death flags. They exist because
    `hp <= 0` is not a death test on this ROM: the KO'd fighter's HP word goes
    to -1 (measured: the only negative value the RAM ever holds), while HP == 0
    is a perfectly ordinary live reading -- during round transitions BOTH words
    sit at exactly 0 for hundreds of consecutive frames with nobody dead. The
    envs derive these flags from the signed HP word and pass them in. Left at
    None they fall back to the historical `hp <= 0` test, so every caller that
    predates the fix computes bit-identical rewards.
    """
    my_dead = (my_hp <= 0) if my_ko is None else bool(my_ko)
    enemy_dead = (enemy_hp <= 0) if enemy_ko is None else bool(enemy_ko)
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

    if cfg.ground_gate_shaping:
        # Extended-state potential: Phi(d, air) = spacing_potential(d) if
        # grounded else 0. Still F(s, s') = gamma*Phi(s') - Phi(s), i.e. pure
        # potential-based shaping over (dist, airborne) -- policy-invariant by
        # Ng/Harada/Russell exactly as the ungated form is over dist alone.
        phi_next = 0.0 if airborne else spacing_potential(rel_dist, cfg)
        phi_prev = 0.0 if prev_airborne else spacing_potential(state.prev_rel_dist, cfg)
    else:
        phi_next = spacing_potential(rel_dist, cfg)
        phi_prev = spacing_potential(state.prev_rel_dist, cfg)
    shaping = cfg.gamma * phi_next - phi_prev

    # Three MUTUALLY EXCLUSIVE outcomes. Draw is its own payoff, not the sum
    # of the other two -- see RewardConfig.draw_penalty for the +15 reward hack
    # this replaces. Draw is checked first so that neither of the other two
    # branches can ever fire alongside it.
    terminal = 0.0
    if terminated:
        if my_dead and enemy_dead:
            terminal = -cfg.draw_penalty
        elif enemy_dead:
            terminal = cfg.win_bonus
        elif my_dead:
            terminal = -cfg.loss_penalty

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
        prev_airborne=airborne,
    )
    return sum(components.values()), next_state, components
