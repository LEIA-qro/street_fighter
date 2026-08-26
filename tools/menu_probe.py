# menu_probe.py -- exploratory driver for SF2'SCE menus under stable-retro.
#
# Boots the Genesis core from POWER-ON (state=stable_retro.State.NONE -- no
# savestate at all) and executes a tiny input script frame by frame, so a
# human/agent can map the boot -> title -> OPTIONS path and locate the
# difficulty setting's RAM address by diffing full-RAM dumps taken at two
# different settings. Throwaway probe: tools/farm_states.py owns the real
# navigation once the map is known.
#
# Script DSL (comma-separated segments, executed in order):
#     "240:., 8:START, 120:., 8:START"
# Each segment is FRAMES:BUTTONS where BUTTONS is "." (nothing held) or
# button names joined by "+" (names from env.buttons, e.g. START, UP, A).
# A segment holds those buttons down for all FRAMES frames; menu screens
# only register edges, so taps are written as "8:START,8:." pairs.
#
# Snapshots are RAW core states (em.get_state() bytes, no gzip) kept in
# --workdir; they let a probe resume where the previous one stopped instead
# of replaying the whole boot. Screenshots are PNGs of em.get_screen().
#
#     .venv/bin/python tools/menu_probe.py --script "240:.,8:START" \
#         --snap-out title --shot title --vars
#     .venv/bin/python tools/menu_probe.py --snap-in title --script "..." ...
#     .venv/bin/python tools/menu_probe.py --diff ram_a.npy ram_b.npy

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import numpy as np  # noqa: E402

DEFAULT_WORKDIR = os.environ.get(
    "MENU_PROBE_WORKDIR",
    os.path.join(REPO_ROOT, "benchmarks", "menu_probe_work"))


def make_poweron_env():
    """retro.make from power-on: no savestate, our custom integration."""
    import stable_retro as retro
    from stable_retro.data import Integrations

    integration_root = os.path.join(REPO_ROOT, "retro_integration")
    if integration_root not in Integrations.CUSTOM_ONLY.paths:
        Integrations.add_custom_path(integration_root)
    # Actions.ALL is load-bearing: the default FILTERED mode masks the action
    # array against the scenario's combat button combos, which silently drops
    # START -- menus become unnavigable and the failure looks like the game
    # ignoring input.
    env = retro.make(
        game="StreetFighterIISpecialChampionEdition-Genesis-v0",
        state=retro.State.NONE, inttype=Integrations.CUSTOM,
        obs_type=retro.Observations.RAM, render_mode=None,
        use_restricted_actions=retro.Actions.ALL,
    )
    env.reset()
    return env


def parse_script(script, buttons):
    """DSL string -> list of (frames, action_array). Validates button names."""
    segments = []
    for raw in script.split(","):
        raw = raw.strip()
        if not raw:
            continue
        frames_s, _, names_s = raw.partition(":")
        frames = int(frames_s)
        arr = np.zeros(len(buttons), dtype=np.uint8)
        if names_s.strip() != ".":
            for name in names_s.split("+"):
                arr[buttons.index(name.strip().upper())] = 1
        segments.append((frames, arr))
    return segments


def run_script(env, segments, shot_every=0, shot_dir=None, tag="probe"):
    frame_no = 0
    for frames, action in segments:
        for _ in range(frames):
            env.step(action)
            frame_no += 1
            if shot_every and frame_no % shot_every == 0:
                save_shot(env, os.path.join(
                    shot_dir, f"{tag}_f{frame_no:05d}.png"))
    return frame_no


def save_shot(env, path):
    from PIL import Image
    Image.fromarray(env.em.get_screen()).save(path)
    return path


def diff_rams(path_a, path_b, context=0):
    a, b = np.load(path_a), np.load(path_b)
    assert a.shape == b.shape, (a.shape, b.shape)
    idx = np.nonzero(a != b)[0]
    print(f"{len(idx)} differing bytes of {len(a)}")
    for i in idx:
        print(f"  offset 0x{i:04X} (68k 0xFF{i:04X}): "
              f"{int(a[i]):3d} (0x{int(a[i]):02X}) -> "
              f"{int(b[i]):3d} (0x{int(b[i]):02X})")
    return idx


def main():
    ap = argparse.ArgumentParser(description="SF2 menu probe (power-on boot)")
    ap.add_argument("--script", default="", help="input DSL, see module docstring")
    ap.add_argument("--snap-in", default=None, help="raw snapshot to resume from")
    ap.add_argument("--snap-out", default=None, help="save raw snapshot at end")
    ap.add_argument("--shot", default=None, help="save screenshot PNG at end")
    ap.add_argument("--shot-every", type=int, default=0)
    ap.add_argument("--ram-out", default=None, help="save full work RAM .npy at end")
    ap.add_argument("--vars", action="store_true", help="print data.lookup_all()")
    ap.add_argument("--workdir", default=DEFAULT_WORKDIR)
    ap.add_argument("--diff", nargs=2, metavar=("A.npy", "B.npy"),
                    help="diff two RAM dumps and exit (no emulator)")
    args = ap.parse_args()

    if args.diff:
        diff_rams(*args.diff)
        return

    os.makedirs(args.workdir, exist_ok=True)
    env = make_poweron_env()
    print(f"[probe] buttons: {env.buttons}")

    if args.snap_in:
        with open(os.path.join(args.workdir, args.snap_in + ".rawstate"), "rb") as f:
            env.em.set_state(f.read())
        print(f"[probe] resumed from snapshot '{args.snap_in}'")

    if args.script:
        segments = parse_script(args.script, env.buttons)
        total = run_script(env, segments, shot_every=args.shot_every,
                           shot_dir=args.workdir, tag=args.snap_out or "probe")
        print(f"[probe] ran {total} frames")

    if args.snap_out:
        path = os.path.join(args.workdir, args.snap_out + ".rawstate")
        with open(path, "wb") as f:
            f.write(env.em.get_state())
        print(f"[probe] snapshot -> {path}")

    if args.shot:
        path = save_shot(env, os.path.join(args.workdir, args.shot + ".png"))
        print(f"[probe] shot -> {path}")

    if args.ram_out:
        path = os.path.join(args.workdir, args.ram_out)
        np.save(path, env.get_ram())
        print(f"[probe] ram ({env.get_ram().shape}) -> {path}")

    if args.vars:
        for k, v in sorted(env.data.lookup_all().items()):
            print(f"    {k:>20} = {v}")

    env.close()


if __name__ == "__main__":
    main()
