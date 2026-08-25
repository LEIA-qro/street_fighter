# Handoff — SF2 RL Pipeline

**Last updated:** 2026-08-25
**Branch:** `sf2-sota-rl-upgrade` (branched from `main` at `11552e8c`, **not yet merged**)
**State:** 24 commits, 110 tests passing, working tree clean, compiles on Python 3.11 / 3.13 / 3.14.

---

## 1. Goal

Bring the Street Fighter II RL pipeline up to current best practice, targeting the defects that were silently degrading training rather than adding new capability for its own sake.

The concrete objective driving everything below: **the trained agent never learned to move**, despite millions of steps. Spacing is the core skill in this game, so an agent that will not walk cannot improve past poke-trading. The plan diagnosed why, fixed it, and fixed everything else the same code review turned up along the way.

Plan: `docs/superpowers/plans/2026-08-24-sf2-sota-rl-upgrade.md` (3,354 lines — read its *Global Constraints*, *Measured Baseline*, and *Agent Logic Review* sections, not the whole thing).
Execution ledger: `.superpowers/sdd/progress.md` (git-ignored; recovery map if context is lost).

---

## 2. Current state of the code

### 2a. Prior session (compressed — was the entire previous handoff)

An earlier agent session did CLI/packaging hardening, unrelated to RL behaviour. Summary of what it established and left behind:

- Added `test_all_cli_scripts_help_execution` to `code_testing/pytest/test_model_testing_config.py`, spawning all 9 CLI entrypoints with `--help` in subprocesses to catch import-time crashes. Also `test_all_readme_cli_command_examples`.
- Fixed a missing trailing newline in `src/scripts/train.py`.
- Established the project standard of compiling cleanly on Python 3.10–3.14 and re-verified README / `doc/DEVELOPER_CLI_GUIDE.md` CLI examples against the actual argparse definitions.
- Verified GPU detection (RTX 5070 Ti Laptop, CUDA available).
- Noted `train_pbt.py` needs `ray` / `ray[tune]`, deliberately excluded from `requirements.txt` to keep the PPO pipeline lean; it fails gracefully with install instructions.
- Its test count was 29. That suite is now folded into the 110 below.
- Its closing claim — "Task Complete… Ready for deployment" — was about CLI conformance only. It did not examine RL semantics, which is where every defect in this session was found.

### 2b. This session — what landed

Nine planned tasks plus two review-driven fix passes. Commit ranges:

| Task | Commits | What |
|---|---|---|
| 1 | `270ce13`..`bdacfbe` | Offline test harness. `FakeBizHawkEnv` subclasses the real env but replaces the socket layer with scripted payloads — no emulator, socket, or ROM. **Previously zero tests touched the env.** |
| 2 | `a7f54ce` | Episode diagnostics: `double_ko`, `timeout`, `episode_steps`, `loss`, `hp_sentinel` in `info`. |
| 3 | `29f8656` | **The movement fix.** Reward extracted to a pure `src/envs/reward.py`; shaping potential rebuilt. |
| 4 | `b836cb5`..`20ad6b1` | PPO implementation details → pure `src/agents/ppo/hyperparams.py`. |
| 5 | `2eb58e1` | Macro action space: `Discrete(63+M)`, specials as atomic multi-step options. |
| 6 | `d597e77`..`ca68501` | Lua payload 13→24 fields (backward compatible); new env `v4` with embedded categorical IDs. |
| 7 | `e397be1` | `RecurrentPPO` backend behind `--recurrent` (sb3-contrib). |
| 8 | `59ea7a3`..`394d92a` | DQN → QR-DQN; shared action wrapper; SAC guarded. |
| 9 | `2d6fc93` | `src/core/elo.py` — Elo + PFSP, standalone. |
| fixes | `0670f31`..`13dc4a9` | Final-review Criticals + the regression the first fix pass caused. |

### 2c. The core diagnosis, with measured evidence

The shaping potential was `Φ(d) = 0.05·max(0, 1 − d/80)` over `rel_dist ∈ [0, 187]`.

`max(0, …)` clamps Φ to **exactly zero for every d ≥ 80**. A live telemetry run (3,000 steps, random policy, all twelve level-1 states) measured:

- `rel_dist` range **0–187**, saturating at 187 (6.5% of samples pinned there)
- median **83** — *inside* the dead zone
- **52.2% of all steps had d ≥ 80**, i.e. zero shaping gradient

Against a flat −0.015/step time penalty over a measured ~570-step round (≈ −8.6), versus a maximum shaping payoff of +0.05, **approaching the opponent was net-negative by more than two orders of magnitude**. The agent was behaving correctly given the reward it was handed. This was a reward bug, not an exploration failure.

Now: two-sided potential peaking at `peak_dist = 70` (poke range, where combos start), non-constant across the whole `[0, 187]` domain, `spacing_weight = 2.5` (×50). Safe to scale because potential-based shaping is policy-invariant for any Φ and any coefficient (Ng, Harada & Russell, ICML 1999) — `γΦ(s′) − Φ(s)` telescopes. Time penalty dropped 0.015 → 0.002.

### 2d. Other verified findings

- **RAM verification (3,000 live steps).** All five addresses the project notes hedged on ("Possible distance of…", "Not sure why it repeats") are **real and informative**: `rel_y_dist` (0x834E) 0–107 with 100 distinct values; `p1_head`/`p2_head`/`p1_chest`/`p2_chest` 69–192, dropping on crouch/jump exactly as documented. The recovered low byte of `0x804E`/`0x82CE` shows 6 distinct values — discarding it *was* costing real information.
- **`p2_btn` (0x845E) reads constant 0** against the CPU. It is the P2 *controller port*; the built-in AI drives its character through game logic and never writes there. Live only in PvP/league. **Deliberately kept** in the v4 layout (documented in `sf2_v4.py`) rather than churning the observation contract — one always-zero embedding the network ignores immediately.
- **Throughput baseline:** ~165 agent steps/s on a single emulator. Mean episode ~570 steps.

---

## 3. Everything tried that failed

Kept deliberately — these are the dead ends and wrong turns, so nobody repeats them.

### 3a. Errors in the plan itself (all caught in review, all fixed)

| What was wrong | Why it mattered |
|---|---|
| Claimed specials were "unreachable" at ~1 in 250,000 | **Wrong.** Assumed all 3 steps needed an exact 63-way match, but the button is free on the setup steps. Real odds ≈ **1 in 1,700** (v3) and 1 in 4,700 (v1/v2 MultiBinary). Specials *did* fire — `logs/action_state_mappings.json` has the recorded QCF sequences. The barrier is **credit assignment**, not reachability. |
| `BUFFER_SIZE = 1_000_000` | Allocates **17.7 GB** — SB3 allocates both `observations` and `next_observations`, and the v2/v3 obs is 2216 float32. `--algo dqn` could not have started. Now 250_000 (4.4 GB) with the arithmetic documented. |
| E4 described only the *both-sided* HP sentinel | The **single-sided** case is more likely and worse: it zeroed one side's HP, produced `ko=True`, and fabricated an `info["loss"]=1` worth **−127 reward from a menu frame**, labelled a clean terminal. |
| Task 3 set reward γ=0.99; Task 4 set PPO γ=0.995 | Broke PBRS invariance. Residual is a real per-step penalty **maximal at exactly `peak_dist=70`** — ≈ −7.1/round for holding poke range vs 0 for camping at max distance, 6× the time penalty, **pointing backwards**. The telescoping test passed only because it constructs `gamma=1.0`. |
| Asserted Task 8 would clean up DQN/SAC's `> 0.0` sentinels | It doesn't. Task 4's switch to `None` sentinels broke `--algo dqn` / `--algo sac` with `TypeError`. Would have survived to the end of the plan. |
| SAC guard scoped to `train()` only | `tune.py --algo sac` would still spin up 16 emulators on the broken objective. A guard covering one of two entrypoints reads as protection that isn't there. |
| Task 8 set `learning_starts=10_000` in the trainer only | The Optuna study fell back to QRDQN's default of 100. Would have returned a full study of hyperparameters tuned under a warmup regime production never uses. No error, no warning. |
| Test files were gitignored (`/code_testing/pytest`) | Every `git add` of a test would have silently done nothing. |
| Task 1's fake didn't mirror `bizhawk_path`/`rom_path`/`lua_path` | `reset()`'s self-healing path catches `RuntimeError`/`OSError` and calls `_start_emulator_bridge()` → live `socket.bind()` + `Popen(EmuHawk.exe)`. `_start_emulator_bridge` is now overridden to raise, so the offline guarantee is enforced rather than intended. |

### 3b. A crash introduced during implementation

**v4 corrupt-payload path killed the worker.** `_parse_payload`'s failsafe returns `frames[-1][:554]`, which for v4 is a 23-element array. `sf2_v4._parse_payload` then ran `np.argmax` over slices of it including `full[266:522]` — **empty** → `ValueError`, raised *outside* `step()`'s try block, taking down a `SubprocVecEnv` rank. Dormant only until the first successful step; the code already counts corrupt payloads and warns every 100, so it would have fired in production. Fixed with a pass-through guard + regression test.

### 3c. A regression caused by one of the fixes

**Fixing the γ mismatch broke four other entrypoints.** Raising `RewardConfig.gamma` 0.99 → 0.995 desynced `pbt_orchestrator.py:148`, `train_exploiter.py:243` and `train_league.py:246`, which all hardcode 0.99 and had been *consistent before*. Sign-flipped the residual into a **+7.1/round stall incentive** at `peak_dist` — it paid the agent to camp and do nothing. Arguably worse than the original.

Now routed through `src/core/rl_constants.py::AGENT_GAMMA`, with an **AST guard test** that fails if anyone reintroduces a discount literal anywhere in `src/`. Point fixes don't stop the next one.

### 3d. Process lesson

**Every Critical finding was a seam between two individually-correct tasks.** Nothing inside any single task's diff was wrong. Per-task review structurally cannot see these. `code_testing/pytest/test_cross_module_contracts.py` exists specifically to guard those seams (reward γ == PPO γ; v4 emitted frame indices == extractor slice indices; production QRDQN kwargs == Optuna's).

---

## 4. Files actively edited

**New (25).** Core: `src/envs/reward.py`, `src/envs/action_macros.py`, `src/envs/macro_wrapper.py`, `src/envs/sf2_v4.py`, `src/core/sf2_extractor.py`, `src/core/rl_constants.py`, `src/core/elo.py`, `src/agents/ppo/hyperparams.py`, `src/agents/common/action_wrappers.py` (+ `__init__.py`).
Tests: `code_testing/pytest/fakes/fake_bizhawk.py`, `test_env_reward.py`, `test_action_macros.py`, `test_macro_wrapper.py`, `test_sf2_extractor.py`, `test_ppo_hyperparams.py`, `test_action_wrappers.py`, `test_elo.py`, `test_cross_module_contracts.py`, `test_dqn_config.py`, `test_dqn_qrdqn_warmup_parity.py`.

**Modified.** `lua/v2.0/training_env_client.lua` (payload 13→24), `src/envs/base_env.py` (heaviest — `step()` and `_parse_payload` were touched by Tasks 2, 3, 5, 6 and both fix passes), `src/core/env_tools.py`, `src/scripts/train.py`, `src/agents/ppo/agent.py`, `src/agents/dqn/{agent,config,optuna_study}.py`, `src/agents/sac/{agent,optuna_study}.py`, `src/agents/pbt/pbt_orchestrator.py`, `src/scripts/{train_league,train_exploiter}.py`, `requirements.txt`, `.gitignore`.

**Deliberately untouched** (would break saved models or was out of scope): `src/envs/sf2_v1.py`, `sf2_v2.py`, `sf2_v3.py`, `src/core/config.py`, `src/envs/league_env.py`, `README.md`.

---

## 5. Next step

**Do this first, before any long run:** add a TensorBoard logging callback for `info["reward_parts"]`, `info["macro_action"]`, and per-episode mean `rel_dist`.

Nothing currently consumes those keys — the curriculum callbacks read only `win`. **Three of the plan's four success metrics are unmeasurable as shipped.** You could spend 6M steps and be unable to tell whether the movement fix worked. ~40 lines, highest value-per-line work remaining.

**Then** follow the plan's own advice: run **Task 3's reward fix alone against `v3`** before layering on macros or v4.

```bash
.venv/Scripts/python.exe src/scripts/train.py --algo ppo --env v3 --auto_curriculum --steps 1000000 --device cpu
```

Watch **mean `rel_dist` per episode**. Baseline is median 83 with 52.2% of steps at d ≥ 80. If the fix worked, the distribution shifts toward the `peak_dist = 70` band and that 52.2% falls. Small diff, large predicted effect, unambiguous attribution — it tells you whether the core diagnosis was right before you spend anything else.

### Known open items (none are committed-broken; all are scope the plan didn't cover)

1. **`src/envs/league_env.py` still carries the old broken reward** — dead-zone potential, −0.015/step, no sentinel handling, divergent constants (0.70 vs 0.77 damage penalty, ±50 vs +65/−50 terminal). Self-play currently trains against the thing this branch fixed. `league_env.step` already delegates parsing to the base `_parse_payload`, so porting it to `src/envs/reward.py` is small.
2. **No `v4` / `--macros` support in evaluation, tuning or league paths.** `test_agent_v2.py`, `test_ai_vs_ai_v2.py`, `tune.py`, `train_league.py` all still gate on `choices=["v2","v3"]`. The flagship config is trainable but you cannot watch, ladder, or tune it.
3. **`pbt_orchestrator.py:148` still has `n_epochs=10, target_kl=0.03`** — Task 4's rollout-truncation defect, never applied to the PBT path.
4. **Optuna tunes `gamma ∈ [0.95, 0.9999]` while the shaping γ is pinned** to `AGENT_GAMMA`. Up to ±0.045 mismatch → up to ±64/round, dominating every other reward term and making γ trial scores incomparable. Documented in `rl_constants.py`; fixing it means threading the trial's γ into `RewardConfig`.
5. **`README.md` not updated** — no `sf2_v4.py` / `reward.py` / `action_macros.py` / `elo.py` in the file tree, and no `--env v4` / `--macros` / `--recurrent` / `--no_anneal_lr` in the command list. The widened Lua payload is a deployment-relevant fact that appears nowhere user-facing.
6. **Sentinel frames still carry a fabricated HP of 0 in the *observation*.** The fix applied "sentinel means HP unknown" to the reward and termination sides only; `_parse_payload` still zeroes `raw[0]`/`raw[1]`, so the policy sees 0 HP for up to `NUM_FRAMES` steps. Pre-existing, not a regression, but the principle is only half-applied.
7. **`.venv` drifts from `requirements.txt`.** `pytest` was missing despite being pinned; `sb3-contrib` had to be installed during Task 7. Run `pip install -r requirements.txt`.

### Landing the branch

Not yet merged. `git log --oneline 11552e8c..HEAD` for the 24 commits. Options: merge to `main`, open a PR, or leave for review — was awaiting that decision when the session ended.
