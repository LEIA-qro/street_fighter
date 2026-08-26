# base_env.py
import os
import random
import time
import numpy as np
from gymnasium import spaces
from collections import deque

import core.config as config
from core.bizhawk_base import BizHawkBaseEnv
from envs.reward import (RewardConfig, RewardState, RoundTracker,
                         compute_reward, hp_to_signed)

CONTINUOUS_DIM = config.OBS_DIM  # HP(2), RelX(1), RelY(1), WallDist(1), Proj_X(2), Vel_X(2), RelDist(1) = 10
ACT_CATEGORIES = 256
CHAR_CATEGORIES = 16
ONE_HOT_ACT_DIM = ACT_CATEGORIES * 2
ONE_HOT_CHAR_DIM = CHAR_CATEGORIES * 2
TOTAL_OBS_DIM = CONTINUOUS_DIM + ONE_HOT_ACT_DIM + ONE_HOT_CHAR_DIM  # 10+512+32 = 554

# Boundary of the OLD shaping potential's dead zone (Phi was identically zero
# for d >= 80). The measured baseline had 52.2% of steps at or past it; the
# per-episode fraction reported in info["ep_rel_dist_frac_far"] is judged
# against that number.
FAR_DIST_THRESHOLD = 80.0

# Lua payload widths this parser accepts. 13 = legacy, 24 = expanded,
# 26 = + round-win counters, 27 = + round clock. Anything else is treated as a
# corrupt frame. Only the SET is exact-matched; every optional field block
# inside _parse_payload is gated with `>=` so widening the payload again can
# never silently drop the blocks below it.
ACCEPTED_PAYLOAD_WIDTHS = (13, 24, 26, 27)


class StreetFighterBaseEnv(BizHawkBaseEnv):
    """Abstract Base Gym Environment for Street Fighter II.

    Handles connection, telemetry bridge, frame stacking, state parsing,
    reward calculation, and socket failure auto-recovery.
    Concrete subclasses override the action spaces and parsing helpers.
    """

    # HP reads above this are round-transition sentinels (0xFF etc.), not health.
    HP_SENTINEL_THRESHOLD = 200

    # debug_mode defaults to False: the every-10k-steps payload print is a
    # diagnostic, and building its f-string on every step of every worker is
    # measurable waste. Pass debug_mode=True to re-enable it.
    def __init__(self, rank=0, lua_path=config.TRAINING_ENV_CLIENT_LUA_PATH, trainable=True, debug_mode=False, player=1, verbose=True,
                 ground_gate=False):
        assigned_port = config.PORT + rank

        super().__init__(
            bizhawk_path=config.BIZHAWK_PATH,
            rom_path=config.ROM_PATH,
            lua_path=lua_path,
            host=config.HOST,
            port=assigned_port,
            trainable=trainable,
            debug_mode=debug_mode,
            verbose=verbose
        )

        self.player = player
        self.active_training_states = config.TRAINING_STATES

        self.prev_my_hp    = 176
        self.prev_enemy_hp = 176
        self.prev_p1_x     = 0
        self.prev_p2_x     = 0
        self.frames        = deque(maxlen=config.NUM_FRAMES)

        # ground_gate also reaches here via set_ground_gate(): sf2_v2's
        # explicit __init__ signature does not forward unknown kwargs, so
        # SFv2_make_env flips the flag post-construction instead.
        self.reward_cfg = RewardConfig(ground_gate_shaping=ground_gate)
        self.reward_state = RewardState(
            prev_my_hp=176.0, prev_enemy_hp=176.0, prev_rel_dist=80.0,
            combo_counter=0, frames_since_last_hit=0,
        )

        # Counter systems
        self._steps = 0
        self.footsie_steps = 0
        self.prev_rel_dist = 80.0
        self.combo_counter = 0
        self.frames_since_last_hit = 0
        self.corrupt_payload_count = 0
        self.sticky_direction = None
        self.sticky_counter = 0
        self.hp_sentinel = False
        self.p1_sentinel = False
        self.p2_sentinel = False
        # Death flags: p1_/p2_ in raw payload order, my_/enemy_ in the acting
        # player's perspective. Set together by _parse_payload; on a corrupt
        # payload they stay stale from the last good frame, the same discipline
        # extra_ram follows (the repeated frame is internally consistent).
        self.p1_ko = False
        self.p2_ko = False
        self.my_ko = False
        self.enemy_ko = False
        # Round-win counters, in the acting player's perspective, plus the
        # same pair in RAW p1/p2 order (league_env re-parses one payload from
        # both perspectives, so it can only trust the raw-order copies).
        # Present only in a 26-field payload; narrower clients leave them at 0.
        self.matches_won = 0
        self.enemy_matches_won = 0
        self.p1_matches_won = 0
        self.p2_matches_won = 0
        # Round clock (field 27). None on every payload the rig sends today,
        # which switches the clock rule off rather than misfiring it.
        self.round_timer = None
        # Counter baseline, clock arming and the once-per-round edge latch.
        # Shared implementation with retro_env and league_env.
        self._round = RoundTracker()

        # Per-episode rel_dist samples (non-sentinel frames only). Summarized
        # into info on the terminal step so the metrics callback can log the
        # spacing distribution without any per-step info traffic.
        self._ep_rel_dists: list = []
        # Non-sentinel steps spent airborne (p1_air truthy) this episode. The
        # denominator is len(self._ep_rel_dists) -- both counters gate on the
        # same sentinel check, so the fraction is over the same step set.
        self._ep_air_steps = 0

        # The post-savestate-load payload of the most recent reset. League
        # play re-parses it from the P2 perspective.
        self._last_reset_payload: str = ""

        # Extra RAM fields from the 24-field payload. Empty when the Lua client
        # is an older 13-field build. Only v4 reads these.
        self.extra_ram: dict = {}

        # Macro actions need exact, unmodified input sequences; MacroActionWrapper
        # turns this off. See src/envs/macro_wrapper.py.
        self.sticky_enabled = True

    def set_training_states(self, new_states):
        """Receives broadcast from the Main Process and updates local memory."""
        self.active_training_states = new_states

    def set_ground_gate(self, enabled: bool) -> None:
        """Flips the anti-jump shaping gate (see RewardConfig.ground_gate_shaping).

        Exists because sf2_v2's explicit __init__ signature (owned by another
        track) does not forward a ground_gate kwarg down to this base class;
        SFv2_make_env calls this right after construction instead.
        """
        self.reward_cfg.ground_gate_shaping = bool(enabled)

    def _get_obs(self):
        return np.concatenate(self.frames)

    def _one_hot(self, val, num_classes):
        """Universal One-Hot Encoder"""
        arr = np.zeros(num_classes, dtype=np.float32)  # float32 to match obs dtype
        safe_val = max(0, min(int(val), num_classes - 1))
        arr[safe_val] = 1.0
        return arr

    def _action_to_string(self, action) -> str:
        """Converts step action object to standard 10-bit binary command string."""
        raise NotImplementedError("Subclasses must implement _action_to_string")

    def step(self, action):
        try:
            # 1. Convert action and Send Action via Parent Method
            action_string = self._action_to_string(action)

            # --- STICKY MOVEMENT LOGIC ---
            # Holds a fresh directional input for two extra agent steps so the
            # policy can walk instead of jittering. Disabled by MacroActionWrapper,
            # whose macros are exact multi-step input sequences.
            action_list = list(action_string)

            if not self.sticky_enabled:
                self.sticky_counter = 0
                self.sticky_direction = None
            else:
                agent_left = action_list[2] == '1'
                agent_right = action_list[3] == '1'
                agent_crouch = action_list[1] == '1'

                # Cancel stickiness if the agent is crouching or inputs the opposite direction
                opposite_input = ((self.sticky_direction == 'L' and agent_right)
                                  or (self.sticky_direction == 'R' and agent_left))

                if agent_crouch or opposite_input:
                    self.sticky_counter = 0
                    self.sticky_direction = None

                if self.sticky_counter > 0:
                    if self.sticky_direction == 'L':
                        action_list[2] = '1'
                        action_list[3] = '0'  # Prevent conflicting Left+Right inputs
                    elif self.sticky_direction == 'R':
                        action_list[3] = '1'
                        action_list[2] = '0'  # Prevent conflicting Left+Right inputs
                    self.sticky_counter -= 1
                elif not agent_crouch:
                    # Initiate stickiness on fresh directional inputs
                    if agent_left:
                        self.sticky_direction = 'L'
                        self.sticky_counter = 2
                    elif agent_right:
                        self.sticky_direction = 'R'
                        self.sticky_counter = 2

            action_string = "".join(action_list)
            # -----------------------------

            full_command = (action_string + "0000000000\n") if self.player == 1 else ("0000000000" + action_string + "\n")
            self.send_command(full_command)
            # 2. Receive State via Parent Method.
            #
            # PROTOCOL PHASE NOTE: the Lua client sends its payload BEFORE
            # waiting for our command, so the payload received here was
            # produced before `action` was applied -- observations (and the
            # HP diffs the reward is computed on) lag actions by exactly one
            # agent step (4 emulator frames). This is a deliberate, consistent
            # property: it pipelines the policy forward pass with emulation
            # (per-step wall time is max(emulation, python), not the sum).
            # Receiving after sending without that pending payload would
            # serialize the loop. reset() keeps the offset primed.
            data = self.receive_payload()

            if self.debug_mode:
                self.debug_print(
                    f"Command Sent: '{full_command}' | Raw Payload: '{data}'"
                )

        except RuntimeError as e:
            # Socket is dead. Return a terminal state so SB3 calls reset().
            # Do NOT let this propagate - it kills the SubprocVecEnv
            print(f"[Rank {self.port - config.PORT}] Socket error in step: {e}. Returning terminal obs.")
            obs = self._get_obs() if len(self.frames) > 0 else np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, 0.0, True, False, {
                "socket_death": True,
                "hp_sentinel": self.hp_sentinel,
                "reward_parts": {},
            }

        observation = self._parse_payload(data, is_reset=False)
        self.frames.append(observation)

        # =====================================================
        # 3. Calculate Reward
        current_my_hp, current_enemy_hp = float(observation[0]), float(observation[1])
        rel_dist = float(observation[9])
        self._steps += 1

        if rel_dist <= self.reward_cfg.peak_dist:
            self.footsie_steps += 1
        else:
            self.footsie_steps = 0

        # p1_air comes from the 24-field payload; a legacy 13-field client
        # leaves extra_ram empty, so airborne reads permanently False and the
        # gate (if enabled) degrades to the ungated shaping. On a corrupt
        # payload extra_ram is stale from the last good frame -- but so are
        # rel_dist and HP (same repeated frame), so the (d, air) pair the
        # potential is evaluated on stays internally consistent.
        airborne = bool(self.extra_ram.get("p1_air", 0))

        # BOTH words reading exactly 0 is the ROM blanking the bars between
        # rounds; one side at 0 is an ordinary live reading, both at once never
        # is, and diffing that against the last real HP invents ~+73 of damage
        # out of a blank screen. Together with the sentinel flag this is the
        # single "these are not health values" predicate that the round rules,
        # the reward and the spacing aggregates all share.
        blanked = bool(current_my_hp == 0 and current_enemy_hp == 0)
        hp_readable = not (self.hp_sentinel or blanked)

        # A round ends by KO -- the loser's HP word goes NEGATIVE and stays
        # there for 33-457 emulator frames (measured; a normal round is 33,
        # only the match-ending KO runs into the hundreds), so no sampling
        # cadence can miss it -- or on the CLOCK. The previous test
        # (`hp <= 0 and not hp_sentinel`) could never fire on a real KO: the
        # Lua client sends HP unsigned, so a KO arrives as 65535, which tripped
        # the sentinel and BLOCKED termination for the whole KO window. By the
        # time the flag cleared the ROM had reset both HP words and the
        # identity of the winner was gone.
        #
        # The clock (field 27) and the win counters (fields 25-26) are BOTH
        # absent from the 24-field Lua client deployed on the rig today, so
        # there time-over detection is simply inactive and the rules degrade
        # to KO-only. agent/stage0-runbook.md 6.5 carries the exact, additive
        # Lua edit that closes the gap; until it lands, a rig episode that
        # runs out the clock still truncates as a TIMEOUT.
        my_ko, enemy_ko = self._round.resolve(
            self.my_ko, self.enemy_ko,
            my_hp=current_my_hp, enemy_hp=current_enemy_hp,
            hp_readable=hp_readable,
            matches_won=self.matches_won,
            enemy_matches_won=self.enemy_matches_won,
            timer=self.round_timer)
        ko = bool(my_ko or enemy_ko)

        # A KO frame carries a sentinel HP word (that IS the negative value),
        # but it is the single most informative frame of the round: both
        # fighters are on screen at real positions and the winner's HP is
        # intact. "Unreadable" means "not a health value AND no round result"
        # -- a menu or round-transition frame.
        unreadable = bool(not hp_readable and not ko)

        if not unreadable:
            self._ep_rel_dists.append(rel_dist)
            if airborne:
                self._ep_air_steps += 1

        if unreadable:
            # Do NOT diff a real previous HP against a fabricated
            # sentinel-derived zero. Skip reward computation and leave
            # reward_state untouched so the next real frame diffs against the
            # last real HP.
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
        truncated = (bool(self._steps >= config.MAX_STEPS_PER_ROUND) and not terminated) if self.trainable else False

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
            # Same audit trail retro_env emits, so outcome-by-cause splitting
            # exists on BOTH backends rather than only on the one nobody
            # trains on. On a 24-field payload these are constant (0, 0,
            # False, None) -- which is itself the signal that the rig cannot
            # see a time over yet.
            info["matches_won_delta"] = self._round.mw_delta
            info["enemy_matches_won_delta"] = self._round.emw_delta
            info["time_over"] = bool(terminated
                                     and not (self.p1_ko or self.p2_ko))
            info["round_timer"] = self.round_timer
            if hasattr(self, "current_state_file"):
                info["state_file"] = self.current_state_file
            self._attach_episode_spacing(info)

        return self._get_obs(), reward, terminated, truncated, info

    def _attach_episode_spacing(self, info: dict) -> None:
        """Summarizes this episode's rel_dist samples into the terminal info.

        Baseline to beat (random policy, 2026-08-24 telemetry run): median 83,
        52.2% of steps at rel_dist >= 80. A working spacing fix shifts the
        distribution toward peak_dist=70 and drops that fraction.
        """
        if self._ep_rel_dists:
            arr = np.asarray(self._ep_rel_dists, dtype=np.float32)
            info["ep_rel_dist_mean"] = float(arr.mean())
            info["ep_rel_dist_median"] = float(np.median(arr))
            info["ep_rel_dist_frac_far"] = float((arr >= FAR_DIST_THRESHOLD).mean())
            # Fraction of non-sentinel steps spent airborne. THE metric for
            # "approaches by jumping": uniform random over MultiDiscrete([9,7])
            # picks one of the 3 jump directions a third of the time and sits
            # at ~0.33 here; a policy that actually walks stays well under
            # 0.15. (Reads 0.0 under a legacy 13-field Lua client, which has
            # no p1_air field.)
            info["ep_air_frac"] = self._ep_air_steps / len(self._ep_rel_dists)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        MAX_RETRIES = 2
        for attempt in range(MAX_RETRIES):
            try:
                # Random Domain Selection
                # Phase selection
                chosen_state_file = random.choice(self.active_training_states)
                self.current_state_file = chosen_state_file
                full_state_path = os.path.join(config.STATES_DIR, chosen_state_file)

                # Send Reset via Parent Method
                if self.trainable:
                    self.send_command(f"RESET {full_state_path}\n")

                    # PROTOCOL PHASE NOTE: exactly one payload is always
                    # pending when reset() runs -- the unsolicited boot
                    # payload on a fresh emulator, or the in-flight payload
                    # produced after the previous step's action. It describes
                    # the PREVIOUS episode's final frame (or the ROM boot
                    # screen), not the state we just asked Lua to load.
                    # Drain it, then read the real post-savestate-load frame.
                    self.receive_payload()          # stale pre-RESET payload
                    data = self.receive_payload()   # fresh post-load payload

                    # Re-prime the one-message offset that pipelines the
                    # policy forward pass with emulation (see step()): send a
                    # neutral 20-bit command so Lua emulates 4 no-op frames
                    # and queues the payload step() will consume first.
                    self.send_command("0" * (2 * config.ACTION_DIM) + "\n")
                else:
                    data = self.receive_payload()

                self._last_reset_payload = data
                observation = self._parse_payload(data, is_reset=True)

                self.prev_my_hp    = float(observation[0]) if observation[0] > 0 else 176.0
                self.prev_enemy_hp = float(observation[1]) if observation[1] > 0 else 176.0

                # Internal absolute coordinates are still tracked for velocity calculation
                # (These are NOT in the final observation anymore, but we need them for deltas)
                csv_string = data.strip().split(" ")[-1]
                parts = csv_string.split(",")
                if len(parts) == 13:
                    raw = [int(x) for x in parts]
                    if self.player == 2:
                        self.prev_p1_x, self.prev_p2_x = raw[3], raw[2]
                    else:
                        self.prev_p1_x, self.prev_p2_x = raw[2], raw[3]

                # Reset reward tracking
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
                    # _parse_payload above populated extra_ram from the
                    # post-load frame; the shaping gate's Phi(s0) must match
                    # whether that frame is actually airborne (False for a
                    # legacy 13-field client, whose extra_ram is empty).
                    prev_airborne=bool(self.extra_ram.get("p1_air", 0)),
                )
                self.sticky_counter = 0
                self.sticky_direction = None
                self.hp_sentinel = False
                self.p1_sentinel = False
                self.p2_sentinel = False
                # Re-baseline the round bookkeeping on the post-load frame
                # _parse_payload just read: counter baseline, clock arming,
                # and -- if that frame is itself inside a KO window, i.e. the
                # savestate was captured mid-KO -- the latch, so the stale
                # result is swallowed instead of terminating on step 1.
                # (The previous code "cleared" p1_ko/p2_ko here and claimed
                # that protected step 1. It did not: step() re-derives them
                # from the same still-negative HP word on the next payload,
                # so the flags were clear for exactly zero steps.)
                self._round.reset(matches_won=self.matches_won,
                                  enemy_matches_won=self.enemy_matches_won,
                                  timer=self.round_timer,
                                  ko=bool(self.p1_ko or self.p2_ko))
                self._ep_rel_dists = []
                self._ep_air_steps = 0

                self.frames.clear()
                for _ in range(config.NUM_FRAMES): self.frames.append(observation)

                return self._get_obs(), {}

            except (RuntimeError, OSError) as e:
                rank = self.port - config.PORT
                if attempt < MAX_RETRIES - 1:
                    print(f"[Rank {rank}] RESET FAILED (Attempt {attempt+1}): {e}. Attempting self-healing respawn...")
                    try:
                        self.close()
                    except Exception: pass
                    time.sleep(5)
                    try:
                        self._start_emulator_bridge()
                    except Exception as bridge_err:
                        print(f"[Rank {rank}] Critical: Self-healing respawn failed: {bridge_err}")
                else:
                    raise RuntimeError(f"[Rank {rank}] BizHawk DEAD on reset after {MAX_RETRIES} attempts: {e}")

    def _parse_payload(self, data, is_reset=False):
        """Builds a 554-dimensional float32 observation.

        Layout per frame:
          [0-9]   Continuous: HP(2), RelX(1), RelY(1), WallDist(1), ProjX(2), VelX(2), RelDist(1)
          [10-265] P1 action one-hot (256)
          [266-521] P2 action one-hot (256)
          [522-537] P1 char one-hot (16)
          [538-553] P2 char one-hot (16)
        """
        # Grab strictly the CSV string, stripping off the leading zero
        csv_string = data.strip().split(" ")[-1]
        parts = csv_string.split(",")

        # 13 = legacy payload, 24 = expanded, 26 = expanded + the two round-win
        # counters, 27 = + the round clock. Fields 1-13 are identical in all
        # of them, so v1/v2/v3 are unaffected by the wider ones. The widths
        # above 13 are accepted AHEAD of the Lua client that emits them: the
        # rig is production hardware owned by another track and cannot be
        # tested here, so the Python side is made ready first and the Lua
        # change stays an additive edit (agent/stage0-runbook.md 6.5 has it
        # verbatim).
        #
        # EVERY optional block below is gated with `>=`, never `==`. It was
        # `if len(raw) == 24:` for one revision, which meant the 26-field
        # payload this code was written to accept fell straight through to
        # `extra_ram = {}` -- silently zeroing 8 of the 23 v4 observation dims
        # (rel_y_dist, p1/p2 chest+head, p1/p2 air, both act_lo, p1_btn) and
        # pinning `airborne` False so the anti-jump ground gate stopped
        # gating. It would have detonated on the day the Lua edit landed, on
        # the rig, with nothing on this machine able to reproduce it.
        if len(parts) in ACCEPTED_PAYLOAD_WIDTHS:
            try:
                raw = [int(x) for x in parts]

                if len(raw) >= 24:
                    p1_lo, p2_lo = raw[13], raw[14]
                    p1_btn, p2_btn = raw[15], raw[16]
                    p1_air, p2_air = raw[17], raw[18]
                    if self.player == 2:
                        p1_lo, p2_lo = p2_lo, p1_lo
                        p1_btn, p2_btn = p2_btn, p1_btn
                        p1_air, p2_air = p2_air, p1_air
                    # NOTE: p2_btn (0x845E) is the P2 controller port. It reads
                    # constant 0 when training against the built-in CPU, which
                    # drives its character through game logic rather than the
                    # input port -- confirmed over a 3000-step live run (single
                    # distinct value observed). It carries real signal only in
                    # PvP/league play, where P2's actions are injected via
                    # joypad.set the same way P1's are. Kept in the layout
                    # anyway: dropping it would churn the observation shape
                    # that Tasks 7-9 build on for the sake of one embedding
                    # lookup the network already learns to ignore.
                    self.extra_ram = {
                        "p1_act_lo": p1_lo, "p2_act_lo": p2_lo,
                        "p1_btn": p1_btn, "p2_btn": p2_btn,
                        "p1_air": p1_air, "p2_air": p2_air,
                        "rel_y_dist": raw[19],
                        "p1_chest": raw[20], "p1_head": raw[21],
                        "p2_chest": raw[22], "p2_head": raw[23],
                    }
                    if self.player == 2:
                        self.extra_ram["p1_chest"], self.extra_ram["p2_chest"] = \
                            self.extra_ram["p2_chest"], self.extra_ram["p1_chest"]
                        self.extra_ram["p1_head"], self.extra_ram["p2_head"] = \
                            self.extra_ram["p2_head"], self.extra_ram["p1_head"]
                else:
                    self.extra_ram = {}

                # Fields 25-26: matches_won (0xFF81DA) / enemy_matches_won
                # (0xFF845A), in RAW P1/P2 order, flipped to the acting
                # player's perspective like every other paired field. The raw
                # order is ALSO kept, because league_env parses one payload
                # from both perspectives through this same method and the
                # perspective-flipped copies are whichever parse ran last.
                if len(raw) >= 26:
                    self.p1_matches_won, self.p2_matches_won = raw[24], raw[25]
                    mw, emw = raw[24], raw[25]
                    if self.player == 2:
                        mw, emw = emw, mw
                    self.matches_won, self.enemy_matches_won = mw, emw

                # Field 27: the round clock (0xFF972A, one BCD byte,
                # 0x99 -> 0x00). Perspective-free. See
                # envs.reward.resolve_round_result for why it, and not the
                # counters, is the primary time-over signal.
                if len(raw) >= 27:
                    self.round_timer = raw[26]

                # The Lua client sends mainmemory.read_u16_be, i.e. UNSIGNED,
                # so a KO'd fighter (-1 in RAM) arrives here as 65535. Decode
                # the sign before anything else: that is the death signal.
                p1_dead = hp_to_signed(raw[0]) < 0
                p2_dead = hp_to_signed(raw[1]) < 0
                p1_sentinel = p1_dead or raw[0] > self.HP_SENTINEL_THRESHOLD
                p2_sentinel = p2_dead or raw[1] > self.HP_SENTINEL_THRESHOLD
                self.p1_sentinel = p1_sentinel
                self.p2_sentinel = p2_sentinel
                # Per-side death flags, in RAW P1/P2 order. step() flips them
                # to the acting player's perspective the same way the HP words
                # are flipped below.
                self.p1_ko = p1_dead
                self.p2_ko = p2_dead
                # A sentinel on either side means that side's HP is not a live
                # health value this frame. step() still skips reward on such a
                # frame -- UNLESS it is a KO, which is the one case where the
                # frame is fully informative and must terminate the episode.
                self.hp_sentinel = p1_sentinel or p2_sentinel
                raw[0] = 0 if p1_sentinel else raw[0]
                raw[1] = 0 if p2_sentinel else raw[1]

                # Death flags in the ACTING player's perspective, flipped by
                # the same rule as the HP words just below.
                if self.player == 2:
                    self.my_ko, self.enemy_ko = p2_dead, p1_dead
                else:
                    self.my_ko, self.enemy_ko = p1_dead, p2_dead

                # PERSPECTIVE FLIP
                if self.player == 2:
                    p1_hp, p2_hp, p1_x, p2_x, p1_y, p2_y = raw[1], raw[0], raw[3], raw[2], raw[5], raw[4]
                    p1_act, p2_act, p1_proj, p2_proj, p1_char, p2_char = raw[7], raw[6], raw[9], raw[8], raw[11], raw[10]
                else:
                    p1_hp, p2_hp, p1_x, p2_x, p1_y, p2_y = raw[0], raw[1], raw[2], raw[3], raw[4], raw[5]
                    p1_act, p2_act, p1_proj, p2_proj, p1_char, p2_char = raw[6], raw[7], raw[8], raw[9], raw[10], raw[11]

                rel_dist = raw[12]

                # --- FIX C: TRANSLATION INVARIANCE ---
                rel_x = int(np.clip(p2_x - p1_x, -500, 500))
                rel_y = int(np.clip(p2_y - p1_y, -200, 200))
                p1_corner_dist = min(p1_x, 500 - p1_x) # Wall awareness

                p1_vel_x = 0 if is_reset else int(np.clip(p1_x - self.prev_p1_x, -100, 100))
                p2_vel_x = 0 if is_reset else int(np.clip(p2_x - self.prev_p2_x, -100, 100))

                # CRITICAL: Preserve internal tracking for velocity deltas
                self.prev_p1_x, self.prev_p2_x = p1_x, p2_x

                # 1. Continuous Features (10 dims)
                cont_obs = np.array([p1_hp, p2_hp, rel_x, rel_y, p1_corner_dist, p1_proj, p2_proj, p1_vel_x, p2_vel_x, rel_dist], dtype=np.float32)

                # 2. Categorical Features (One-Hot Encoded)
                p1_act_oh  = self._one_hot(p1_act,  ACT_CATEGORIES)
                p2_act_oh  = self._one_hot(p2_act,  ACT_CATEGORIES)
                p1_char_oh = self._one_hot(p1_char, CHAR_CATEGORIES)
                p2_char_oh = self._one_hot(p2_char, CHAR_CATEGORIES)

                # 3. Smash them all together
                return np.concatenate((cont_obs, p1_act_oh, p2_act_oh, p1_char_oh, p2_char_oh))

            except ValueError: pass

        # Failsafe: Return 554 zeros if the string is corrupted
        self.corrupt_payload_count += 1
        if self.corrupt_payload_count % 100 == 0:
            print(f"[WARNING] {self.corrupt_payload_count} corrupt payloads received. Check socket integrity.")
        # Return last known good observation instead of zeros:
        return self.frames[-1][:TOTAL_OBS_DIM] if len(self.frames) > 0 else np.zeros(TOTAL_OBS_DIM, dtype=np.float32)
