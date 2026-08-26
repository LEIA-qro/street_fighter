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
