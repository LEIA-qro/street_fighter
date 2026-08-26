# retro_smoke.py -- live smoke test of the stable-retro v4 backend.
#
#     .venv/bin/python tools/retro_smoke.py [--steps N]
#
# Boots RetroSF2Env (real emulator, real ROM, headless), runs N random agent
# steps, and prints: obs shape/bounds check, one decoded 23-float frame with
# field labels, reward statistics with component totals, episode outcomes, and
# throughput (agent steps/s and emulator fps). Needs stable-retro + the ROM
# imported; the unit tests in code_testing/pytest/test_retro_env.py cover the
# pure logic offline instead.

import argparse
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import numpy as np  # noqa: E402

from envs.retro_env import (  # noqa: E402
    RetroSF2Env, FRAME_SKIP, NUM_FRAMES, V4_FRAME_DIM,
)

FRAME_LABELS = [
    "p1_hp", "p2_hp", "rel_x", "rel_y", "corner_dist", "p1_proj", "p2_proj",
    "p1_vel_x", "p2_vel_x", "rel_dist", "rel_y_dist", "p1_head", "p2_head",
    "p1_air", "p2_air", "p1_act_hi", "p2_act_hi", "p1_act_lo", "p2_act_lo",
    "p1_btn", "p2_btn", "p1_char", "p2_char",
]


def main():
    parser = argparse.ArgumentParser(description="Smoke test RetroSF2Env")
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    env = RetroSF2Env()
    print(f"[smoke] retro buttons order: {env._buttons}")
    print(f"[smoke] action_space: {env.action_space}")

    obs, _ = env.reset()
    print(f"[smoke] obs shape: {obs.shape} dtype: {obs.dtype} "
          f"(expect ({V4_FRAME_DIM * NUM_FRAMES},) float32)")

    frame = obs[-V4_FRAME_DIM:]
    print("[smoke] decoded reset frame:")
    for name, val in zip(FRAME_LABELS, frame):
        print(f"    {name:>10} = {val:g}")

    rewards, part_totals = [], {}
    episodes = wins = losses = timeouts = sentinel_steps = 0
    rng = np.random.default_rng(0)

    t0 = time.perf_counter()
    for _ in range(args.steps):
        action = rng.integers([9, 7])
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        sentinel_steps += int(info["hp_sentinel"])
        for k, v in info["reward_parts"].items():
            part_totals[k] = part_totals.get(k, 0.0) + v
        if terminated or truncated:
            episodes += 1
            wins += info["win"]
            losses += info["loss"]
            timeouts += int(info["timeout"])
            print(f"[smoke] episode end @ step {info['episode_steps']}: "
                  f"win={info['win']} loss={info['loss']} timeout={info['timeout']} "
                  f"my_hp={info['my_hp']:g} enemy_hp={info['enemy_hp']:g} "
                  f"spacing(mean={info.get('ep_rel_dist_mean', float('nan')):.1f}, "
                  f"frac_far={info.get('ep_rel_dist_frac_far', float('nan')):.2f})")
            obs, _ = env.reset()
    elapsed = time.perf_counter() - t0

    r = np.asarray(rewards, dtype=np.float64)
    print(f"[smoke] {args.steps} agent steps in {elapsed:.2f}s = "
          f"{args.steps / elapsed:.0f} agent steps/s "
          f"({args.steps * FRAME_SKIP / elapsed:.0f} emulator fps)")
    print(f"[smoke] reward: sum={r.sum():.2f} mean={r.mean():.4f} "
          f"min={r.min():.2f} max={r.max():.2f}")
    print(f"[smoke] reward part totals: "
          + ", ".join(f"{k}={v:.2f}" for k, v in sorted(part_totals.items())))
    print(f"[smoke] episodes={episodes} wins={wins} losses={losses} "
          f"timeouts={timeouts} sentinel_steps={sentinel_steps}")

    last = obs[-V4_FRAME_DIM:]
    lo = np.array(env.observation_space.low[:V4_FRAME_DIM])
    hi = np.array(env.observation_space.high[:V4_FRAME_DIM])
    out = [(FRAME_LABELS[i], float(last[i]))
           for i in range(V4_FRAME_DIM) if not (lo[i] <= last[i] <= hi[i])]
    print(f"[smoke] final frame within obs-space bounds: "
          f"{'yes' if not out else f'NO -- {out}'}")

    env.close()


if __name__ == "__main__":
    main()
