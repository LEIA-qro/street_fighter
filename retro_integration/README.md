# Custom stable-retro integration for SF2'SCE (Genesis)

Registered by `src/envs/retro_env.py` via `Integrations.add_custom_path(<this dir>)`
and used with `inttype=Integrations.CUSTOM`, which searches here FIRST and falls
back to the shipped package dir -- that fallback is where `rom.md` lives, so this
dir needs no ROM copy.

`StreetFighterIISpecialChampionEdition-Genesis-v0/data.json` exposes the full
24-field RAM table the BizHawk Lua client reads (the shipped integration only has
health/score/timer). The authoritative address list is
`lua/v2.0/training_env_client.lua` lines 73-129; every `data.json` address is that
Lua offset plus `0xFF0000` (Genesis 68k work RAM base -- confirmed by the shipped
integration's `health` at 16744514 == 0xFF8042). Types: `>u2` for the Lua
`read_u16_be` fields, `|u1` for `read_u8`.

Two variables cover two payload fields each: `p1_state_word`/`p2_state_word`
(0x804E/0x82CE) hold action id in the hi byte and act_lo in the low byte --
stable-retro can only read raw RAM, so `envs/retro_env.assemble_v4_frame()` does
the split (plus the air-flag normalization, `p2_y & 0xFF` mask and projectile
freshness test the Lua client performs before sending its payload).

`scenario.json` is deliberately inert (no done condition, no reward variables):
reward and termination are computed in Python against the v4 contract.
`metadata.json` and the `.state` file are copied from the shipped integration so
this dir is self-sufficient for state loading.

## FightLadder state import (2026-08-26)

143 savestates raided from [FightLadder](https://github.com/wenzhe-li/FightLadder)
(MIT, ICML 2024 -- they run this exact game/core on gym-retro) now live next to
the shipped state, all prefixed `FL_` and cataloged in `../states_manifest.json`
(shared catalog; other state-farming tracks append to the same file via
read-merge-write). `tools/verify_states.py` is the permanent linter: it boots
every cataloged state in RetroSF2Env, waits out screen transitions and the
round-intro freeze (~100 frames where nothing can move), then checks
p1/p2 chars, full HP (176/176) and 30 random steps of liveness. `--write`
stores results into the manifest. All 144 pass as of import.

**Curriculum states** (`FL_Level{N}.{i}`, 126 files): saved by FightLadder's
finetune eval the moment the agent reached arcade-ladder level N (episode i),
i.e. at the round-1 start of that level's fight, Ryu as P1, full HP. Levels
4/8/12 are the bonus stages (absent). The ladder order proved deterministic --
every instance of a level showed the same `p2_char` in RAM:

| Level | 1 | 2 | 3 | 5 | 6 | 7 | 9 | 10 | 11 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Opponent | GUILE | KEN | CHUNLI | ZANGIEF | DHALSIM | RYU | EHONDA | BLANKA | BALROG | VEGA | SAGAT | MBISON |

That is all 12 opponents with Ryu as P1. **Difficulty is null for every
FightLadder state**: the curriculum inherits whatever setting the shipped
`Champion.Level1.RyuVsGuile` was recorded at (undocumented), no documented
RAM variable exposes it, and guessing is banned. These states therefore do
NOT slot into the `RYU_{OPP}_R1_lvl{N}` BizHawk naming (which encodes a
verified difficulty) -- hence the `FL_` convention.

**Stars states** (`FL_Champion.Level1.RyuVsRyu.{left,right}_star{1..8}`, 16
files): RyuVsRyu at 8 "star" settings per side. FightLadder's README calls the
star a "difficulty level", but the states' embedded gzip filenames reveal they
were carved from a 2-player base state (`Champion.Level1.unknown.2Player`), so
the stars may be VS-mode handicap rather than options-menu difficulty --
difficulty stays null. Verification hints: on all 8 `left_star` states P2
moves without being hit (CPU-driven opponent, trainable); on `right_star`
2/4/7/8 P2 only reacts to hits (likely a passive human port -- FightLadder
drove the RIGHT side there; treat as league material, not CPU opponents).

**2P state** (`FL_Champion.RyuVsRyu.2Player.align`): both ports human, marked
`"players": 2` in the manifest, load-verified only. Future league use.
