# base_env.py
import os
import random
import time
import numpy as np
from gymnasium import spaces
from collections import deque

import core.config as config
from core.bizhawk_base import BizHawkBaseEnv
from envs.reward import RewardConfig, RewardState, compute_reward

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

        if not self.hp_sentinel:
            self._ep_rel_dists.append(rel_dist)
            if airborne:
                self._ep_air_steps += 1

        ko = bool(current_my_hp <= 0 or current_enemy_hp <= 0) and not self.hp_sentinel

        if self.hp_sentinel:
            # HP is unknown on this frame (round transition, menu, KO
            # animation on at least one side) -- do NOT diff a real previous
            # HP against a fabricated sentinel-derived zero. Skip reward
            # computation and leave reward_state untouched so the next real
            # frame diffs against the last real HP.
            reward, reward_parts = 0.0, {}
        else:
            reward, self.reward_state, reward_parts = compute_reward(
                self.reward_state, current_my_hp, current_enemy_hp,
                rel_dist, ko, self.reward_cfg,
                airborne=airborne,
                prev_airborne=self.reward_state.prev_airborne,
            )

            # Mirror into the legacy attributes other modules still read.
            self.prev_my_hp = self.reward_state.prev_my_hp
            self.prev_enemy_hp = self.reward_state.prev_enemy_hp
            self.prev_rel_dist = self.reward_state.prev_rel_dist
            self.combo_counter = self.reward_state.combo_counter
            self.frames_since_last_hit = self.reward_state.frames_since_last_hit

        # A frame where either HP read is a sentinel is a round transition,
        # not a KO. Terminating there fabricates episodes out of menu frames.
        terminated = ko if self.trainable else False
        truncated = (bool(self._steps >= config.MAX_STEPS_PER_ROUND) and not terminated) if self.trainable else False

        info = {
            "my_hp": current_my_hp,
            "enemy_hp": current_enemy_hp,
            "hp_sentinel": self.hp_sentinel,
            "reward_parts": reward_parts,
        }
        if terminated or truncated:
            double_ko = bool(terminated and current_my_hp <= 0 and current_enemy_hp <= 0)
            info["double_ko"] = double_ko
            info["timeout"] = bool(truncated)
            info["episode_steps"] = self._steps
            info["win"] = 1 if (terminated and current_enemy_hp <= 0 and current_my_hp > 0) else 0
            info["loss"] = 1 if (terminated and current_my_hp <= 0 and current_enemy_hp > 0) else 0
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

        # 13 = legacy payload, 24 = expanded payload. Fields 1-13 are identical
        # in both, so v1/v2/v3 are unaffected by the wider one.
        if len(parts) in (13, 24):
            try:
                raw = [int(x) for x in parts]

                if len(raw) == 24:
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

                p1_sentinel = raw[0] > self.HP_SENTINEL_THRESHOLD
                p2_sentinel = raw[1] > self.HP_SENTINEL_THRESHOLD
                self.p1_sentinel = p1_sentinel
                self.p2_sentinel = p2_sentinel
                # A sentinel on EITHER side means that side's HP is unreadable
                # this frame (round transition, menu, KO animation) -- not
                # that HP is actually zero. step() treats any sentinel frame
                # as "HP unknown": it refuses to terminate and skips reward
                # computation entirely, rather than diffing a real HP against
                # a fabricated zero (a false KO when only one side sentinels,
                # or ~+23 reward of pure noise when both do).
                self.hp_sentinel = p1_sentinel or p2_sentinel
                raw[0] = 0 if p1_sentinel else raw[0]
                raw[1] = 0 if p2_sentinel else raw[1]

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
