# SF2 RL — Project Handoff & Optimization Charter

> **Addendum 2026-08-25 (noche), rama `stage0-metrics-and-semantics`:** el P0 de
> §5.1 está resuelto — `MetricsCallback` registra en TensorBoard reward_parts,
> spacing por episodio, uso de macros y throughput; hay scripts de medición
> (`benchmark_throughput.py`, `measure_spinlock.py`) y quedaron corregidos los
> open items 1 (league reward), 3 (PBT) y 9 (docstring), el atexit de workers,
> la γ=0.99 de `selective_norm`, el replay ratio de QR-DQN y la observación
> rancia de `reset()` (ver la nota de fase del protocolo en `base_env.step()` —
> el lag de un paso durante el episodio es deliberado y pipelinea emulación con
> inferencia). Suite: 126 tests. Instrucciones de ejecución en la 13900K:
> `agent/stage0-runbook.md`.
>
> **Addendum 2026-08-26:** el estado VIVO del proyecto se mantiene ahora en
> **`agent/memory/`** (INDEX.md primero) — arquitectura de dos backends + flota ES,
> bugs cazados, decisiones, infra y bitácora de runs. Este handoff queda como
> referencia histórica profunda; para retomar trabajo, empieza por la memoria.

**Last updated:** 2026-08-25
**Branch:** `sf2-sota-rl-upgrade` (from `main` at `11552e8c`, 24 commits, **not merged**)
**Verified state:** 110 pytest tests passing (re-run 2026-08-25, 81 s, fully offline).
**Predecessor document:** `agent/archive/handoff-2026-08-25-branch-session.md` — the per-session execution ledger this file supersedes.
**Governing plan:** `docs/superpowers/plans/2026-08-24-sf2-sota-rl-upgrade.md` (3,354 lines).

---

## 0. Read this first

This document hands a **complete, verified mental model** of the project to a stronger model, so it can act without re-deriving anything.

Three things to internalise before reading further:

1. **The project's headline defect was an environment bug, not an algorithm bug.** The agent never learned to move because the reward function paid it not to. That is fixed in code and **has never been validated by a training run.**
2. **Every metric that fix would be judged by is currently unlogged.** The env emits the diagnostic keys; nothing reads them. Fixing that is ~40 lines and gates everything else.
3. **The strategic objective is throughput.** The stated direction is to optimise *everything* — environment, wire protocol, observation, network, host configuration — until evolutionary methods at the scale of **EGGROLL** become viable. Section 7 sets numeric gates for when that becomes true. Section 8 explains why it is not true today.

Sections 1–3 are context and architecture. Sections 4–6 are the problem set. Sections 7–8 are the forward plan. Sections 9–11 are the ledger, open items and reference material.

---

## 1. Core objective

### 1.1 The game-level goal

Train an agent to play **Street Fighter II' — Special Champion Edition (Genesis)** as **Ryu**, against the built-in CPU across all 12 opponents and 8 difficulty levels, and eventually against other trained agents in a self-play league.

The behavioural target is **spacing**. Street Fighter is a game of distance management — walking in and out of poke range, whiff-punishing, controlling the neutral. An agent that will not walk cannot improve past trading pokes, which is exactly the plateau every run to date has hit.

### 1.2 The engineering goal

Bring the pipeline to current best practice, in strict priority order:

1. **Correctness** — no silent reward, termination, or observation defects. *(Largely achieved this branch; see §9.)*
2. **Measurability** — every claim about agent behaviour backed by a logged metric. ***Not achieved.*** *(See §5.1.)*
3. **Throughput** — maximise agent steps/second on the given hardware. *(Barely started; §6, §7.)*
4. **Algorithmic headroom** — only after 1–3, because algorithm changes on an unmeasured, slow, buggy pipeline are unattributable. *(§8.)*

### 1.3 The strategic goal: make EGGROLL viable

**EGGROLL** = *Evolution Guided General Optimization via Low-rank Learning*, from **"Evolution Strategies at the Hyperscale"** (Oxford / Mila / NVIDIA, Nov 2025, [arXiv:2511.16652](https://arxiv.org/abs/2511.16652)). It is an Evolution Strategies variant in which each population member's perturbation is a **rank-r matrix `ABᵀ`** rather than a dense per-parameter perturbation. The mean update stays full-rank, but arithmetic intensity rises enormously — the paper reports ~100× throughput for billion-parameter models at large population sizes, reaching 91 % of pure batch-inference throughput, and **parity with OpenES on tabula-rasa and multi-agent RL**.

ES is **gradient-free**. That is the appeal here: it can optimise the objective actually cared about — match win rate, Elo — rather than a differentiable shaped surrogate. It is also trivially parallel and indifferent to reward sparsity, credit-assignment depth, and non-differentiable action spaces.

Its cost is **rollouts**. One ES generation requires a full episode evaluation *per population member*. That makes environment throughput the single gating variable, which is why the whole programme in §7 is a ladder toward a measurable throughput threshold with an explicit go/no-go gate (§7.6).

**Current honest verdict: not viable today, by roughly two orders of magnitude.** The arithmetic is in §8.4. The purpose of this document is to specify what would have to change for that verdict to flip — and to ensure every intermediate step is valuable *regardless of whether it ever does*, since every item in §7 also speeds up PPO, Rainbow and Dreamer.

---

## 2. Hardware and runtime environment

### 2.1 Target training machine (the one that matters)

| Component | Spec | Relevance |
|---|---|---|
| CPU | **Intel Core i9-13900K** — 8 P-cores (16 threads) + 16 E-cores = **24 physical cores / 32 threads** | The binding constraint. Emulator instances are single-threaded CPU work. |
| GPU | **NVIDIA RTX 4090, 24 GB VRAM** | Massively underused. The policy is a small MLP; the GPU idles during rollout collection. |
| RAM | **64 GB DDR5-6400 CL32** | Comfortably holds a 1 M-transition replay buffer even at the wide v2/v3 observation (17.7 GB); trivial at v4 (0.74 GB). |
| OS | Windows 11 | EmuHawk is the Windows build; `env_tools.failsafe_env` uses PowerShell / `taskkill` for orphan cleanup. |

**Heterogeneous-core note (P vs E cores).** The 13900K's E-cores are substantially slower per-clock than its P-cores. Sixteen identical emulator workers scheduled indiscriminately across both will run at the speed of the slowest, and `SubprocVecEnv` is **lock-step** — every `step()` waits for the slowest worker. **Explicit CPU affinity (emulators pinned to E-cores, learner + PyTorch pinned to P-cores) is a real, unexploited win** and should be measured early (§7.3).

### 2.2 Development machine (where all current measurements were taken)

Python 3.13.12, **RTX 5070 Ti Laptop GPU**, 24 logical CPUs.

> **⚠ CRITICAL CAVEAT.** Every throughput number in this document and in the plan — including the **~165 agent steps/s** baseline — was measured on the *laptop*, not on the 13900K/4090. **All performance baselines must be re-measured on the training machine before any optimisation decision is made.** Treat existing figures as order-of-magnitude only.

### 2.3 Software stack (pinned, verified installed)

```
python 3.13.12          gymnasium 1.3.0        stable-baselines3 2.9.0
sb3-contrib 2.9.0       torch 2.13.0+cu130     optuna 4.9.0
tensorboard 2.21.0      gradio 6.25.0          pytest 8.x
numpy >= 1.26           opencv-python 5.0.0.93 ale-py 0.12.1
```

Emulator: **BizHawk 2.8** (`EmuHawk.exe`), Genesis core, Lua 5.4.
The project targets clean compilation on **Python 3.10–3.14**. `sb3-contrib` must track SB3's minor version exactly.
`ray` / `ray[tune]` are **deliberately excluded** from `requirements.txt` (needed only by `train_pbt.py`, which fails gracefully with install instructions).

**The project directory must live inside the BizHawk folder** — `config.BIZHAWK_PATH` resolves as `dirname(PROJECT_ROOT)/EmuHawk.exe` and raises `FileNotFoundError` at import time otherwise.

---

## 3. Architecture

### 3.1 Topology

```
                ┌──────────────────────────────────────────────┐
                │  src/scripts/train.py   (CLI entry point)     │
                │  --algo --env --macros --recurrent --steps    │
                └───────────────────┬──────────────────────────┘
                                    │ dynamic dispatch: agents.{ppo,dqn,sac}
                ┌───────────────────▼──────────────────────────┐
                │  Agent (PPO / RecurrentPPO / QR-DQN)         │
                │  + AutoCurriculumCallback                    │
                └───────────────────┬──────────────────────────┘
                ┌───────────────────▼──────────────────────────┐
                │  SelectiveVecNormalize (obs + reward norm)   │
                └───────────────────┬──────────────────────────┘
                ┌───────────────────▼──────────────────────────┐
                │  SubprocVecEnv  ×  N_ENVS (16)               │
                └───────────────────┬──────────────────────────┘
                    ┌───────────────┴────────────────┐
                    │  per rank: Monitor →           │
                    │  [MacroActionWrapper] →        │
                    │  StreetFighterEnvV{2,3,4}      │
                    └───────────────┬────────────────┘
                                    │  TCP 127.0.0.1 : (9999 + rank)
                                    │  strict lock-step, 1 round-trip per agent step
                ┌───────────────────▼──────────────────────────┐
                │  EmuHawk.exe + lua/v2.0/training_env_client   │
                │  invisible, no sound, no vsync, unthrottled   │
                │  FRAME_SKIP = 4 emulator frames per agent step│
                └──────────────────────────────────────────────┘
```

### 3.2 The lock-step wire protocol — the performance critical path

This is the most important section for §7.

**Per agent step, exactly one round trip:**

1. **Lua → Python:** reads RAM, formats a CSV payload, `comm.socketServerSend(payload)`.
   Format: `"0 f1,f2,…,f24\n"` — a leading `0`, a space, then **24 comma-separated integers**.
2. **Lua spinlock:** busy-polls `comm.socketServerResponse()` until non-empty. A **dead-man's switch** kills EmuHawk after 600 s of silence.
3. **Python → Lua:** a **20-character command string** + `\n` — 10 bits for P1, 10 for P2 (`"1"`/`"0"` per button), prefixed by Python's `{len} ` framing.
   Bit order: `[Up, Down, Left, Right, A(LK), B(MK), C(HK), X(LP), Y(MP), Z(HP)]`.
4. **Lua:** applies the input via `joypad.set` and runs `emu.frameadvance()` **4 times** (action repeat), holding the input across all four frames.

**Special commands:** `RESET <absolute_state_path>` loads a savestate and advances one frame without injecting input; `EXIT` restores emulator defaults and calls `client.exit()`.

**Hard invariants — do not break these:**

- `config.ACTION_DIM = 10`; a two-player command is exactly **20 chars + newline**.
- `config.NUM_FRAMES = 4` (frame stack).
- `FRAME_SKIP = 4` in [training_env_client.lua:49](lua/v2.0/training_env_client.lua:49).
- Payload fields **1–13 must never change position** — `v1`/`v2`/`v3` and every saved model parse them. Fields 14–24 were added additively; the parser gates on `len(parts) in (13, 24)`.

**Emulator performance flags already set** ([training_env_client.lua:7-11](lua/v2.0/training_env_client.lua:7)):
`client.setwindowsize(1)`, `client.invisibleemulation(true)`, `emu.displayvsync(false)`, `client.displaymessages(false)`, `client.SetSoundOn(false)` — and because `config.ENABLE_THROTTLING = False`, `emu.limitframerate(false)`. **There is no artificial frame-rate cap.** (`config.THROTTLE_SPEED = 250` is dead code unless throttling is re-enabled — do not "fix" it thinking it is a bottleneck.)

**Known un-tuned aspects of this path** (all candidates for §7.2):

- No `TCP_NODELAY` on the socket ([bizhawk_base.py:48-50](src/core/bizhawk_base.py:48)).
- The payload is transferred as **decimal ASCII** and parsed with `str.split` + per-field `int()` every step.
- The Lua spinlock **burns a full core while waiting** for Python — with 16 workers that is 16 cores spent on busy-waiting.

### 3.3 RAM map — the 24-field payload

Read every agent step by [training_env_client.lua:73-128](lua/v2.0/training_env_client.lua:73). Addresses are Genesis main memory.

| # | Field | Address | Access | Notes |
|---|---|---|---|---|
| 1 | `p1_hp` | `0x8042` | u16 be | > 200 ⇒ **sentinel** (round transition / menu), not health |
| 2 | `p2_hp` | `0x82C2` | u16 be | same |
| 3 | `p1_x` | `0x8006` | u16 be | absolute; used only for velocity deltas |
| 4 | `p2_x` | `0x8286` | u16 be | |
| 5 | `p1_y` | `0x800A` | u16 be | |
| 6 | `p2_y` | `0x828A` | u16 be & 0xFF | |
| 7 | `p1_action_id` | `0x804E >> 8` | hi byte | move / state identity, 0–255 |
| 8 | `p2_action_id` | `0x82CE >> 8` | hi byte | |
| 9 | `p1_proj_x` | `0x8506` | u16 be | `-1` when frozen; activity inferred from frame-to-frame change |
| 10 | `p2_proj_x` | `0x8586` | u16 be | same |
| 11 | `p1_char_id` | `0x81DB` | u8 | 0–11 |
| 12 | `p2_char_id` | `0x845B` | u8 | 0–11 |
| 13 | `rel_dist` | `0x834C` | u16 be | **engine-native separation. Range 0–187, clips at 187.** The quantity all spacing shaping is built on. |
| 14 | `p1_action_lo` | `0x804E & 0xFF` | low byte | **recovered this branch**; 6 distinct values measured — was being discarded |
| 15 | `p2_action_lo` | `0x82CE & 0xFF` | low byte | |
| 16 | `p1_btn` | `0x81E2` | u8 | P1 raw controller port; 0–64 observed |
| 17 | `p2_btn` | `0x845E` | u8 | **reads constant 0 vs. the built-in CPU** — it is the P2 controller *port*, and the CPU drives its character through game logic, never the port. Live only in PvP/league. Kept for shape stability. |
| 18 | `p1_air` | `0x80C0 != 0` | → 0/1 | airborne flag |
| 19 | `p2_air` | `0x86F4 == 13` | → 0/1 | airborne flag (different encoding) |
| 20 | `rel_y_dist` | `0x834E` | u16 be | engine-native vertical separation; 0–107, 100 distinct values |
| 21 | `p1_chest` | `0x80DC` | u16 be | posture clearance, 69–192, drops on crouch/jump |
| 22 | `p1_head` | `0x80E0` | u16 be | |
| 23 | `p2_chest` | `0x835C` | u16 be | |
| 24 | `p2_head` | `0x8360` | u16 be | |

All addresses **verified live over a 3,000-step run** on 2026-08-24 across all twelve level-1 states. Source notes: `doc/RAM locations of Street Fighter II.txt`.

**Not yet located: the round timer.** Its absence forces the `MAX_STEPS_PER_ROUND = 1500` artificial truncation. Finding it is cheap now that the payload is widened.

**Unexploited RAM frontier:** BizHawk's `memory.usememorydomain("VRAM")` exposes the Genesis Sprite Attribute Table — up to 80 sprites × 8 bytes (Y, size+link, tile index + palette/priority/flip, X). Parsing it yields real on-screen geometry (limb extension, exact projectile position, whether two attacks will trade) as **~20–40 scalars instead of ~28,000 pixels**. This is the cheap way to get what pixels would give, and the natural step after v4 if the observation still feels insufficient.

### 3.4 Environment versions

All inherit `StreetFighterBaseEnv` ([base_env.py](src/envs/base_env.py), 408 lines) → `BizHawkBaseEnv` ([bizhawk_base.py](src/core/bizhawk_base.py), socket + process lifecycle).

| Env | Action space | Obs / frame | Obs total | Status |
|---|---|---|---|---|
| `v1` | `MultiBinary(10)` | 554 | 2216 | Frozen. Do not modify. |
| `v2` | `MultiBinary(10)` | 554 | 2216 | Frozen. Default in most scripts. |
| `v3` | `MultiDiscrete([9, 7])` = 63 combos | 554 | 2216 | Current production. |
| `v4` | `MultiDiscrete([9, 7])` | **23** | **92** | **Flagship.** New this branch. |

**v2/v3 observation layout (554 floats/frame):**
`[0-9]` continuous — `p1_hp, p2_hp, rel_x, rel_y, p1_corner_dist, p1_proj_x, p2_proj_x, p1_vel_x, p2_vel_x, rel_dist`; `[10-265]` P1 action one-hot (256); `[266-521]` P2 action one-hot (256); `[522-537]` P1 char one-hot (16); `[538-553]` P2 char one-hot (16).

**2,048 of the 2,216 inputs are one-hot and almost always zero**, feeding a `Linear(2216, 512)` first layer of 1.13 M parameters, ~92 % of which see a dead input — while simultaneously *omitting* the airborne flags, both players' raw inputs, and the low byte of the state word.

**v4 observation layout (23 floats/frame):**
`[0-9]` same continuous block; `[10]` `rel_y_dist`; `[11-12]` `p1_head`, `p2_head`; `[13-14]` `p1_air`, `p2_air`; `[15-16]` `p1_act_hi`, `p2_act_hi`; `[17-18]` `p1_act_lo`, `p2_act_lo`; `[19-20]` `p1_btn`, `p2_btn`; `[21-22]` `p1_char`, `p2_char`.

**v4 carries strictly more information in 4 % of the width.** Categorical IDs stay raw and are embedded by `SF2FeaturesExtractor` (§3.6) — the standard treatment for large categorical game state (AlphaStar, *Nature* 575:350-354).

**Perspective flip:** when `player == 2`, `_parse_payload` swaps every P1/P2 field pair, so one policy can train from either side.

**Sentinel handling:** an HP read above `HP_SENTINEL_THRESHOLD = 200` means that side's HP is *unreadable this frame* (round transition, menu, KO animation) — not zero. On any sentinel frame `step()` **refuses to terminate and skips reward computation entirely**, leaving `reward_state` untouched so the next real frame diffs against the last real HP.

**Corrupt-payload failsafe:** returns the last good frame rather than zeros, increments `corrupt_payload_count`, warns every 100.

### 3.5 Reward — [src/envs/reward.py](src/envs/reward.py) (pure, 135 lines, no I/O)

`compute_reward(state, my_hp, enemy_hp, rel_dist, terminated, cfg) -> (total, next_state, components)`. The component dict **always sums exactly to the total**, which is what makes both the unit tests and the (unbuilt) TensorBoard breakdown trustworthy.

| Component | Formula | Config |
|---|---|---|
| `damage` | `damage_scale × clamp(prev_enemy_hp − enemy_hp, 0, 100)` | `damage_scale=1.0`, `damage_clamp=100` |
| `taken` | `−0.77 × clamp(prev_my_hp − my_hp, 0, 100)` | `damage_taken_penalty=0.77` |
| `combo` | `min(counter × 0.5, 4.0)` within a 6-step window | `combo_step=0.5`, `combo_cap=4.0` |
| `shaping` | `γ·Φ(d′) − Φ(d)` | `γ = AGENT_GAMMA = 0.995` |
| `time` | `−0.002` on any step with no damage dealt | was `0.015` |
| `terminal` | `+65` win, `−50` loss | |

**The spacing potential Φ — the heart of the project's core fix:**

```
Φ(d) = spacing_weight · d / peak_dist                        for d ≤ 70
Φ(d) = spacing_weight · (max_dist − d) / (max_dist − 70)     for d > 70
   with spacing_weight = 2.5, peak_dist = 70, max_dist = 187
```

Two-sided on purpose: a monotone "closer is always better" potential teaches rushdown, but Ryu wins by *holding* a spacing band. Magnitude is free to tune — potential-based shaping is policy-invariant for **any** Φ and any coefficient (Ng, Harada & Russell, ICML 1999) because `γΦ(s′) − Φ(s)` telescopes.

### 3.6 Action space, macros, and the feature extractor

**Primitives:** `MultiDiscrete([9, 7])` — 9 directions × 7 buttons = **63** combinations, mapped to the 10-bit string by `DIRECTION_MAP` / `BUTTON_MAP` in [sf2_v3.py](src/envs/sf2_v3.py).

**Sticky movement** ([base_env.py:107-144](src/envs/base_env.py:107)): a fresh directional input is held for **2 extra agent steps** so the policy can walk instead of jittering. Cancelled by crouching or by the opposite direction. **Disabled automatically by `MacroActionWrapper`**, whose macros are exact input sequences.

**Macros** ([action_macros.py](src/envs/action_macros.py) + [macro_wrapper.py](src/envs/macro_wrapper.py), behind `--macros`): flattens to `Discrete(63 + 9) = Discrete(72)`. Indices `[0,63)` are primitives; `[63,72)` are temporally-extended options executed over consecutive agent steps.

```
hadouken_lp / hadouken_hp    ↓, ↘, →+P     (3 steps)
shoryuken_lp / shoryuken_hp  →, ↓, ↘+P     (3 steps)
tatsumaki_lk / tatsumaki_mk  ↓, ↙, ←+K     (3 steps)
jump_forward / jump_back                    (2 steps)
dash_block                                  (3 steps)
```

Macros are authored **facing right** and mirrored on decode via the sign of `rel_x`. Reward over a macro is the **undiscounted sum** of its inner steps — standard semi-MDP treatment (Sutton, Precup & Singh 1999); intra-option discounting error at γ=0.995 over ≤3 steps is under 1.5 %.

**Why macros exist — and the corrected reasoning.** The plan originally claimed specials were "unreachable" at ~1 in 250,000. **That was wrong, and the correction matters:** the button is free on the setup steps, so real odds are ≈ **1 in 1,700** (v3). Specials *do* fire — `logs/action_state_mappings.json` contains recorded QCF sequences. **The barrier is credit assignment, not reachability:**

- (a) The two setup steps have **negative local advantage**. Crouching in neutral whiffs and gets counter-poked, and it occurs constantly in non-Hadouken contexts where it is genuinely bad, so its average advantage stays negative and PPO suppresses it.
- (b) `MultiDiscrete` puts direction in **one 9-way softmax**, so raising `P(↓)` for step 1 mechanically lowers `P(↘)` and `P(→)` for steps 2–3 — the three steps of one motion **compete inside one head**. (MultiBinary in v1/v2 factorised into independent Bernoullis whose marginals reinforce each other — the likely reason v3 regressed here.)

Collapsing the motion into one atomic action removes both failure modes.

**Feature extractor for v4** — [sf2_extractor.py](src/core/sf2_extractor.py): continuous + flags pass through; `p1/p2_act_hi` → `nn.Embedding(256, 32)`; `act_lo` and `btn` → `nn.Embedding(256, 16)`; chars → `nn.Embedding(16, 8)`. IDs are `.round().long().clamp(...)` so a corrupt payload cannot index out of bounds and kill a 16-worker run.

**Resulting network widths** (`net_arch = dict(pi=[512,512,256], vf=[512,512,256])` — separate π and V trunks):

| Env | Features dim | Params per trunk | Total policy params |
|---|---|---|---|
| v2 / v3 | 2216 (flat) | ≈ 1.53 M | **≈ 3.1 M** |
| v4 | 636 (embedded) | ≈ 0.72 M | **≈ 1.45 M** (incl. 12.4 k embeddings) |

*Retain this number — it is the parameter budget that decides ES feasibility in §7.6.*

### 3.7 Agents

| Path | Algorithm | State |
|---|---|---|
| [src/agents/ppo/](src/agents/ppo/) | **PPO** (SB3) — main workflow | Production. `hyperparams.py` is pure and unit-tested. |
| same, `--recurrent` | **RecurrentPPO** (PPO-LSTM, sb3-contrib) | Works. Forces `n_steps=512, batch_size=256` — full 2048×16 LSTM BPTT will not fit alongside 16 emulators. |
| [src/agents/dqn/](src/agents/dqn/) | **QR-DQN** (sb3-contrib, `n_quantiles=51`) | Works. Was plain DQN. |
| [src/agents/sac/](src/agents/sac/) | **SAC** | **Deliberately dead.** Both `train()` and `tune()` raise `NotImplementedError` — SB3's SAC is continuous-only. |
| [src/agents/pbt/](src/agents/pbt/) | Ray Tune PBT orchestrator | Works, but **never received this branch's PPO fixes**. |
| [src/agents/league/](src/agents/league/) | `pool_manager.py` — self-play pool | Works, but trains against the **old broken reward** (§10.1). |
| [src/core/elo.py](src/core/elo.py) | Elo + PFSP weighting | **Written, tested, wired into nothing.** |

**PPO configuration** ([hyperparams.py](src/agents/ppo/hyperparams.py) + [config.py](src/agents/ppo/config.py)), with the reasoning that produced it:

```
n_steps      2048   (× N_ENVS=16 → 32,768-sample rollouts)   FIXED after model creation
batch_size   1024                                            FIXED after model creation
n_epochs     4      (was 10)
gamma        0.995  ← core.rl_constants.AGENT_GAMMA
gae_lambda   0.95
target_kl    None   (was 0.03)
vf_coef      0.5    max_grad_norm 0.5    normalize_advantage True
learning_rate  linear anneal to 0 by default (--no_anneal_lr disables)
lr 2.108e-05   ent_coef 0.01536   clip_range 0.2603   (from Optuna)
```

- `target_kl=0.03` was **discarding most of every rollout**: SB3 aborts the epoch loop once `approx_kl > 1.5 × target_kl`, which at `clip_range ≈ 0.26` fired within 1–2 of the 10 configured epochs. Lower `n_epochs` instead of truncating them.
- `gamma` 0.99 → 0.995: at FRAME_SKIP=4, γ=0.99 gives a ~100-step ≈ 6.7-second horizon, far short of a round. 0.995 roughly doubles it so the value head can see the KO from mid-round.
- `resolve_override(cli, phase)` exists because the previous `if x > 0.0` sentinel made `--ent_coef 0` a **silent no-op**. 0.0 is a real value.

**QR-DQN configuration** ([dqn/agent.py:120-136](src/agents/dqn/agent.py:120)): `buffer_size=250_000`, `batch_size=256`, `learning_starts=10_000`, `train_freq=4`, `gradient_steps=1`, `target_update_interval=10_000`, ε 1.0→0.05 over 10 % of training.

Buffer sizing arithmetic ([dqn/config.py](src/agents/dqn/config.py)): SB3 allocates **both** `observations` and `next_observations`, so `1_000_000 × 2216 × 4 B × 2 = 17.7 GB` — the old value could not start. 250 k = 4.4 GB. **On v4 (92 floats) even 1 M is 0.74 GB.** With 64 GB DDR5 on the target machine this constraint largely evaporates.

### 3.8 Curriculum

[auto_curriculum_callback.py](src/agents/auto_curriculum_callback.py) (675 lines), enabled by `--auto_curriculum`:

- **8 difficulty levels × 12 opponents** (`config.DIFFICULTY_LEVELS`), from `RYU_*_lvl1` to `RYU_*_HARD`.
- **Rehearsal Weighted Lottery** — the sampling pool is built by *replication*: prior levels ×1, current level ×3, newly introduced states ×5.
- **Micro-steps** — 2 states from the next level introduced at a time.
- **Gated advancement** — ≥ 75 % win rate (`WIN_RATE_THRESHOLD`) over a rolling window (`WIN_RATE_WINDOW = 250`), requiring **3 consecutive** stable evaluations before promotion.
- **JSON persistence** for crash recovery; `train.py` re-reads `auto_curriculum_state_{MODEL_NAME}.json` on retry to resync steps and level.
- States are broadcast to all workers via `env.env_method("set_training_states", …)` — a fix for a multiprocessing inheritance bug where workers kept the parent's stale list.

`manual_curriculum_callback.py` is the 4-phase predecessor, used without `--auto_curriculum`.

**The callbacks read only `info["win"]`.** They do not read `reward_parts`, `macro_action`, `rel_dist`, `double_ko`, `timeout`, or `hp_sentinel`. This is the measurement gap in §5.1.

### 3.9 Normalisation — [selective_norm.py](src/core/selective_norm.py)

`SelectiveVecNormalize` normalises **only the leading continuous dimensions of each stacked frame**, passing one-hot / categorical dims through unchanged; it also normalises reward. Persisted as `*_vecnorm.pkl` alongside every model `.zip`.

- v1/v2/v3 → `n_continuous_dims = config.OBS_DIM = 10`
- **v4 → `n_continuous_dims = V4_CONT_DIM = 13`** ([ppo/agent.py:105-110](src/agents/ppo/agent.py:105)). `config.OBS_DIM` must stay 10; v1–v3 depend on it.

**Any saved model plus its `.pkl` must keep loading.** Observation or action-space changes ship as a **new env version**, never as an in-place edit.

### 3.10 Entry points

| Script | Purpose | v4 / macros aware? |
|---|---|---|
| [train.py](src/scripts/train.py) | Single-agent training | ✅ `v1,v2,v3,v4`, `--macros`, `--recurrent` |
| [resume.py](src/scripts/resume.py) | Resume from checkpoint | partial |
| [tune.py](src/scripts/tune.py) | Optuna study driver | ❌ |
| [train_league.py](src/scripts/train_league.py) | Self-play league | ❌ `choices=["v2","v3"]` |
| [train_exploiter.py](src/scripts/train_exploiter.py) | Exploiter training | ❌ `choices=["v2","v3"]` |
| [train_pbt.py](src/scripts/train_pbt.py) | Ray PBT (needs `ray[tune]`) | ❌ |
| [test_agent_v2.py](src/scripts/test_agent_v2.py) | Watch agent vs CPU | ❌ |
| [test_ai_vs_ai_v2.py](src/scripts/test_ai_vs_ai_v2.py) | Agent vs agent | ❌ |
| [web_dashboard.py](src/scripts/web_dashboard.py) | Gradio control panel (1,667 lines) | ❌ (4 dropdowns hardcode `["v2","v3"]`) |

**Robustness machinery worth knowing about.** `train.py` wraps training in a **10-retry loop** that catches socket death, runs `failsafe_env()`, sleeps 10 s, resyncs progress from the curriculum JSON, and restarts. `failsafe_env()` kills child processes, then uses PowerShell `Get-CimInstance Win32_Process` to snipe **orphaned `EmuHawk.exe` grandchildren** whose command line contains `street_fighter`, then GCs and empties the CUDA cache; registered via `atexit`. Env boots are **staggered by `rank × 3.5 s`** to avoid 16 simultaneous emulator launches. `prevent_sleep()` blocks Windows standby during training. `torch.set_num_threads(2)` keeps PyTorch from hijacking cores from live emulators.

### 3.11 Test harness — `code_testing/pytest/` (110 tests, fully offline)

**No emulator, no socket, no ROM.** `fakes/fake_bizhawk.py` subclasses the real env and replaces the socket layer with scripted payload strings; `_start_emulator_bridge` is **overridden to raise**, so the offline guarantee is *enforced* rather than intended (`reset()`'s self-healing path would otherwise `socket.bind()` and `Popen(EmuHawk.exe)`).

`test_cross_module_contracts.py` is the important one — it guards **seams between individually-correct modules**: reward γ == PPO γ; v4 emitted frame indices == extractor slice indices; production QRDQN kwargs == Optuna's. There is also an **AST guard test that fails if any discount-factor literal is reintroduced anywhere in `src/`**.

```bash
.venv/Scripts/python.exe -m pytest code_testing/pytest -q
```

### 3.12 Constants that must never drift

[rl_constants.py](src/core/rl_constants.py) exists solely to prevent a class of bug that already cost this project two regressions.

```python
AGENT_GAMMA = 0.995   # the ONLY discount-factor literal in the codebase
```

`envs/reward.py::RewardConfig.gamma` and `agents/ppo/hyperparams.py::build_ppo_kwargs` both read it. Potential-based shaping is policy-invariant **only when the shaping γ equals the acting agent's discount**. If they drift, the shaping term stops telescoping and becomes a real per-step reward — see §9.3 for what that cost.

---

## 4. Current state — done, proven, unmeasured

### 4.1 What landed on this branch (verified)

Nine planned tasks plus two review-driven fix passes, 24 commits.

| Task | Commits | What |
|---|---|---|
| 1 | `270ce13`..`bdacfbe` | Offline test harness. **Previously zero tests touched the env.** |
| 2 | `a7f54ce` | Episode diagnostics: `double_ko`, `timeout`, `episode_steps`, `loss`, `hp_sentinel` in `info`. |
| 3 | `29f8656` | **The movement fix.** Reward extracted to a pure module; shaping potential rebuilt. |
| 4 | `b836cb5`..`20ad6b1` | PPO implementation details → pure `hyperparams.py`. |
| 5 | `2eb58e1` | Macro action space: `Discrete(63+M)`, specials as atomic multi-step options. |
| 6 | `d597e77`..`ca68501` | Lua payload 13→24 fields (backward compatible); env `v4` with embedded categoricals. |
| 7 | `e397be1` | `RecurrentPPO` backend behind `--recurrent`. |
| 8 | `59ea7a3`..`394d92a` | DQN → QR-DQN; shared action wrapper; SAC guarded. |
| 9 | `2d6fc93` | `core/elo.py` — Elo + PFSP, standalone. |
| fixes | `0670f31`..`13dc4a9` | Final-review Criticals, plus the regression the first fix pass caused. |

> **Note on the plan file:** all 84 of its checkboxes are still `- [ ]`. They were never ticked during execution. The work *is* done — verified by file existence, commit range, and the passing suite — but the plan document reads as 0 % complete. Reconcile it in one commit to avoid misleading the next reader.

### 4.2 The core diagnosis, with measured evidence

The shaping potential was `Φ(d) = 0.05·max(0, 1 − d/80)` over `rel_dist ∈ [0, 187]`.

`max(0, …)` clamps Φ to **exactly zero for every d ≥ 80**. A live telemetry run (3,000 steps, random policy, all twelve level-1 states, 2026-08-24) measured:

- `rel_dist` range **0–187**, saturating at 187 (6.5 % of samples pinned there)
- median **83** — *inside* the dead zone; quartiles p25 = 31, p75 = 136
- **52.2 % of all steps had d ≥ 80**, i.e. **zero shaping gradient**
- per-opponent medians: ZANGIEF 48, KEN 58, VEGA 87, RYU 98, BLANKA 108

Against a flat −0.015/step time penalty over a measured ~570-step round (≈ −8.6), versus a maximum shaping payoff of +0.05, **approaching the opponent was net-negative by more than two orders of magnitude**. The agent was behaving correctly given the reward it was handed. **This was a reward bug, not an exploration failure.**

Raw report: `scratchpad/reldist_report.json`.

### 4.3 Measured baseline

| Quantity | Value | Caveat |
|---|---|---|
| Throughput, single emulator | **~165 agent steps/s** (≈ 660 emulator fps ≈ 11× real time) | Laptop, not the 13900K |
| Throughput, 16 emulators aggregate | **NEVER MEASURED** | The single most important missing number |
| Mean episode length | **~570 agent steps** (≈ 2,280 frames) | `MAX_STEPS_PER_ROUND=1500` rarely truncates under a random policy |
| Test suite | 110 tests, 81 s | Offline |

---

## 5. The problems, ranked

### 5.1 P0 — Nothing measures the thing the branch was built to fix

`base_env.py` **writes** `info["reward_parts"]` (lines 164, 189, 191, 212) and `MacroActionWrapper` writes `info["macro_action"]`. A repository-wide grep finds **no reader for either**. The curriculum callbacks consume only `info["win"]`.

**Three of the plan's four success metrics are unmeasurable as shipped.** You can spend 6 M steps and be unable to tell whether the movement fix worked.

The four metrics the plan defines:

1. **Does it move?** Mean `rel_dist` per episode. Baseline: median 83, 52.2 % of steps at d ≥ 80. If the fix worked the distribution shifts toward the `peak_dist = 70` band and that 52.2 % falls. **This is the single clearest signal in the whole project** — it is the behaviour that has never been obtained.
2. `train/n_updates` — should now be `4 × (32768/1024) = 128` per rollout, versus the 1–2 epochs' worth `target_kl=0.03` allowed.
3. Per-component reward breakdown from `info["reward_parts"]` — confirms `shaping` is now the same order as `damage` over a round rather than 0.2 % of it.
4. Macro usage frequency (`info["macro_action"] >= 63`) — if the policy never selects a macro after 1 M steps, raise `ent_coef` before concluding macros do not help.

**Fix: a TensorBoard logging callback, ~40 lines. Highest value-per-line work in the repository. Do it before any long run.**

### 5.2 P1 — Throughput is unmeasured and unoptimised

See §6 and §7. The aggregate 16-emulator figure does not exist; the P/E-core split is unexploited; the wire protocol has never been profiled.

### 5.3 P2 — The flagship configuration is trainable but not observable

`--env v4 --macros` can be trained by `train.py` and by nothing else. You cannot **watch** it, **ladder** it, or **tune** it (§3.10). `core/elo.py` — the module written specifically to make progress measurable across a heterogeneous opponent pool — is wired into nothing.

### 5.4 P3 — Documented, deliberate conflicts

- **Optuna tunes γ ∈ [0.95, 0.9999] while the shaping γ is pinned** to `AGENT_GAMMA`, in **all three** studies (`ppo`, `dqn`, `sac`). A ±0.045 mismatch yields a residual up to ±0.1125/step ≈ **±64 per round**, dominating `win_bonus=65` and making γ trial scores **incomparable**. Documented at length in `rl_constants.py`. **γ-tuning results from all three studies are currently untrustworthy.** The fix is to thread the trial's γ into `RewardConfig(gamma=trial_gamma)` at the same call site that builds the model.
- **`QR-DQN` replay ratio is ~0.016.** `train_freq=4` counts *calls to `collect_rollouts`*, so with `N_ENVS=16` that is **64 transitions collected per gradient step**. Off-policy sample efficiency comes entirely from replaying each transition many times; at 0.016 the only reason to use a value-based method has been discarded — while the 4090 sits idle. **One-line experiment, do it before evaluating any DQN-family algorithm.**
- **`pbt_orchestrator.py:148`** still constructs PPO with `n_epochs=10, target_kl=0.03` — Task 4's rollout-truncation defect, never applied to the PBT path.

---

## 6. Where the time actually goes

**This section is a hypothesis set, not a measurement.** Nobody has profiled the loop. Attribution below must be established before it is acted on.

One agent step costs: 4 × Genesis frame emulation + 1 TCP round trip + payload format/parse + Python-side observation assembly + (amortised) a policy forward pass.

| Candidate cost | Status | Cheap test |
|---|---|---|
| Genesis emulation, 4 frames | Irreducible without leaving BizHawk | Measure Lua-side frames/s with Python replaced by a stub responder |
| **Lua spinlock busy-wait** | **Unquantified. Suspected large** — 16 workers × 1 core each spent waiting | Compare wall time at N_ENVS = 4, 8, 16, 24 |
| TCP round trip (loopback, no `TCP_NODELAY`) | Unquantified; strict ping-pong rarely triggers Nagle, so possibly nothing | One `setsockopt` line, then re-measure |
| ASCII payload format + parse | Unquantified; 24 `int()` calls + `split` per step per worker | Profile `_parse_payload` in isolation |
| One-hot construction (v2/v3) | 554-float assembly with two 256-wide `_one_hot` calls per step | **Already solved by v4** (23 floats) — quantify the delta |
| Policy forward pass | Small; batched across 16 envs on GPU | `torch.profiler` |
| `SubprocVecEnv` lock-step sync | **Suspected significant on a P/E machine** — every step waits for the slowest worker | Per-rank step-time histogram |
| Savestate load on reset | ~570 steps between resets, so amortised | Time `RESET` round trips separately |

**The one number that matters most and does not exist: aggregate agent steps/s at N_ENVS = 16 on the 13900K.** Everything in §7 is scored against it.

---

## 7. The optimisation roadmap

Strictly ordered. Each stage is independently valuable; each is a prerequisite for the next stage's conclusions to mean anything.

### Stage 0 — Instrument (blocking, ~1 day)

1. **TensorBoard callback** for `reward_parts` (per component), `macro_action` (usage histogram + fraction ≥ 63), per-episode mean/median `rel_dist`, fraction of steps with `rel_dist ≥ 80`, `double_ko`, `timeout`, `hp_sentinel` rate, `episode_steps`. *(~40 lines. §5.1.)*
2. **Throughput counter** — aggregate agent steps/s, per-rank step-time histogram, and a `SubprocVecEnv` wait-time metric.
3. **Re-measure the baseline on the 13900K/4090.** Every existing number is from the laptop.

**Exit criterion:** you can read mean `rel_dist` and aggregate steps/s off a dashboard.

### Stage 1 — Validate the core hypothesis (blocking, ~1 day of compute)

Run **Task 3's reward fix alone against `v3`**, no macros, no v4 — the plan's own advice. Small diff, large predicted effect, unambiguous attribution.

```bash
.venv/Scripts/python.exe src/scripts/train.py --algo ppo --env v3 --auto_curriculum --steps 1000000 --device cpu
```

**Watch mean `rel_dist` per episode.** Baseline is median 83 with 52.2 % of steps at d ≥ 80. If the fix worked, the distribution shifts toward the `peak_dist = 70` band and that 52.2 % falls.

**This tells you whether the core diagnosis was right before you spend anything else.** If it did not work, everything downstream is built on a wrong model of the problem, and the correct response is to re-diagnose, not to add algorithms.

### Stage 2 — Host and configuration throughput (cheap, high certainty)

Everything here is hours of work with no architectural risk.

1. **Scale `N_ENVS`.** Currently 16; the 13900K has 24 physical cores. Sweep 16 / 20 / 24 / 28 and plot aggregate steps/s. Expect the curve to bend where emulators start contending with the learner.
2. **CPU affinity.** Pin emulator workers to E-cores, the learner + PyTorch to P-cores. Lock-step `SubprocVecEnv` runs at the pace of the slowest worker, so **making the workers homogeneous may matter more than making them fast**. Also try P-cores-only at lower `N_ENVS` — fewer, faster, uniform workers can beat more, uneven ones.
3. **`torch.set_num_threads`.** Currently hard-coded to 2 ([train.py](src/scripts/train.py)). Retune against the affinity plan.
4. **`TCP_NODELAY`** on the bridge socket. One line; measure, keep or discard.
5. **Confirm v4's narrow payload is actually on the wire.** If Lua still serialises all 24 fields for v4 when fewer are consumed, that is free bandwidth and parse time. (Note: the wide payload is *required* for backward compatibility with v1–v3, so any narrowing must be a **new protocol mode**, negotiated at handshake — do not break the 13/24-field contract.)
6. **Kill the busy-wait if it profiles hot.** A blocking read instead of a Lua spinlock would return ~16 cores.
7. **Buffer sizes.** With 64 GB, `BUFFER_SIZE` for QR-DQN can go back to 1 M on v4 (0.74 GB) or even v2/v3 (17.7 GB) if that path is revived.

**Target: measure, then state a number.** A 1.5–2× aggregate improvement here would be unsurprising and costs almost nothing.

### Stage 3 — Architectural throughput: a second, headless backend

This is the step that changes the order of magnitude, and it directly implements the "strip the environment down" plan.

**Assessment of the three proposed steps:**

| Proposed step | Verdict |
|---|---|
| **1. Ditch the GUI for headless C++ (stable-retro)** | **Directionally right, number optimistic.** stable-retro (Farama's maintained gym-retro fork) runs Genesis through libretro in-process with no window, and **SF2 Special Champion Edition is an integrated game** with a `data.json` RAM-variable schema. Porting the 24 addresses into `data.json` is straightforward and arguably cleaner than the Lua bridge. **But "50,000+ FPS on an 8–16 core consumer box" is off by roughly an order of magnitude for Genesis.** That figure belongs to the JAX-native environment world (Brax, Gymnax, Craftax) where the entire env is a pure function on the GPU. A Genesis emulator is sequential, branch-heavy, cache-hostile CPU work. Realistic: **~10 k–40 k emulator frames/s across 8–16 processes → ~2.5 k–10 k *agent* steps/s at frame-skip 4.** A genuine 2–4× over the current estimate, not 20×. |
| **2. Read RAM, not pixels** | **Already done, to a higher standard than the advice describes.** This project has never used pixels; `base_env.py` parses a RAM payload, and v4 encodes 24 verified addresses compactly. Note also that retro's own `obs_type=RAM` hands you the whole 64 KB address space, which is *worse* — you would want `data.json` variables plus a custom observation, i.e. you would be reimplementing v4. |
| **3. Shrink the CNN to an MLP** | **Already true — and for ES the framing is inverted.** There is no CNN anywhere in this project; it is `[512,512,256]` MLP already. But for evolution strategies **~1.5 M parameters is the problem, not the solution.** Naive ES perturbs every parameter per population member; that is precisely why EGGROLL's low-rank trick exists. A serious consumer-PC ES run wants something like `[64,64]` ≈ **45 k parameters**, at which point plain OpenES or CMA-ES handles it fine and **EGGROLL is solving a problem you do not have**. |

**What to actually build:** a **stable-retro backend as an alternative fast environment for experiments and cross-validation**, with BizHawk retained as ground truth for evaluation, PvP and league play. This is a cheaper version of the DIAMBRA cross-check the plan already lists as future scope.

**What breaks in the move, and must be planned for:**

- The 96-savestate curriculum must be re-authored as retro `.state` files.
- **2-player self-play is much less well-supported in retro** than the BizHawk PvP path. The league infrastructure is the part that does not port cleanly.
- The reward, macro, extractor and curriculum modules are already **pure and env-agnostic** — `reward.py`, `action_macros.py`, `macro_wrapper.py`, `sf2_extractor.py`, `elo.py` port unchanged. This is the payoff from Task 3/5/6's refactors and it makes the backend swap far cheaper than it would otherwise be.

**Keep the abstraction honest:** a new backend must satisfy the same `StreetFighterBaseEnv` observation and action contracts, so a policy trained on one can be evaluated on the other. That cross-check is itself a bug detector.

### Stage 4 — Algorithm

Only after Stages 0–3. See §8 for the full evaluation. Short version: **fix the QR-DQN replay ratio first (one line), then n-step returns, then PER.** Treat DreamerV3 as a cross-check to run on the fast backend, not on the BizHawk rig.

### Stage 5 — Recurrence and observation depth

If Stage 1 succeeds and the agent moves but still cannot read move startup/recovery: a 4-frame stack at frame-skip 4 covers only 16 emulator frames, while a whiffed Shoryuken is 50+. `--recurrent` (RecurrentPPO) already exists — compare it against feed-forward **on the same seed**. If the observation still feels thin after that, the order of attack is: **expanded RAM → embeddings → VRAM sprite table → pixels. Do not skip to the end.**

### 7.6 The EGGROLL viability gate

Do not attempt ES until **all five** of these hold. Each is measurable.

| # | Gate | Threshold | Why |
|---|---|---|---|
| 1 | **Aggregate throughput** | **≥ 25,000 agent steps/s sustained** (≈ 100 k emulator fps at frame-skip 4) | §8.4 arithmetic: ES needs ~300 M steps for a run; 300 M ÷ 25 k ≈ 3.3 h. Below this, one ES run costs days. |
| 2 | **Policy size** | **≤ ~100 k parameters**, or an architecture where low-rank `ABᵀ` perturbation is natural | At 1.45 M (v4) dense ES is memory- and bandwidth-bound; at 45 k (`[64,64]` on the v4 features) plain OpenES suffices and EGGROLL is unnecessary. **EGGROLL only earns its place once the network is large enough to make dense ES the bottleneck** — which argues for going *bigger* deliberately, not smaller, if ES is the goal. |
| 3 | **Episode cost** | **No per-episode process boot**; resets must be savestate loads inside a live process, forkable and deterministic | Currently a 3.5 s staggered boot per rank and a 10-retry socket-death loop. ES evaluates thousands of episodes per generation; any per-episode fixed cost dominates. |
| 4 | **Objective** | A **non-differentiable objective worth optimising directly** — Elo or match win rate, from `core/elo.py`, actually wired up | ES's real advantage over PPO is optimising the thing you care about instead of a shaped surrogate. Without this, ES is just a worse gradient estimator. |
| 5 | **PPO plateau** | A **documented PPO ceiling** — a run that has stopped improving with metrics to prove it | Otherwise you cannot tell whether ES helped or merely differed. |

**If gates 1–5 hold, the recommended first ES use is narrow, not a rewrite:** a **gradient-free fine-tuning pass on a frozen, already-trained policy**, perturbing only the last layer, optimising Elo directly. That is where ES is strongest and where failure is cheap.

---

## 8. Algorithm evaluations

**Applies to all of them: none addresses the defect this branch diagnosed.** The agent did not move because Φ(d) was identically zero for 52.2 % of steps and approaching cost 8.6 reward to earn 0.05. Swapping the optimizer on a broken reward gets you a differently-shaped failure. Everything below assumes Stage 1 has validated the reward fix.

### 8.1 Rainbow DQN — *worth it, but build the 60 % version*

**Where you already are:** QR-DQN ([dqn/agent.py:120](src/agents/dqn/agent.py:120), `n_quantiles=51`) — Rainbow's *distributional* component in a stronger form than C51, plus SB3's target network. **2 of Rainbow's 7 pieces.**

**Missing:** prioritized replay, n-step returns, dueling heads, noisy nets, double-Q. In the Hessel et al. ablation, **prioritized replay and multi-step returns were the two components whose removal hurt most**; noisy nets and dueling were marginal. That is the build order, and it means most of Rainbow can be skipped.

**Cost:** SB3 ships neither PER nor an n-step buffer, so both mean a custom `ReplayBuffer` subclass (sum-tree, importance weights) plus overriding `QRDQN.train()` to thread IS weights into the quantile-Huber loss — realistically 400–600 lines with tests. Dueling and noisy nets each need a custom policy/network. There is **no drop-in**; Tianshou has full Rainbow but adopting it means a second training stack beside the SB3 pipeline, the `SelectiveVecNormalize` contract and the curriculum callbacks.

**Why it is nonetheless the best fit:** you are **sample-starved and compute-rich** — a ~2.6 k steps/s emulator ceiling against an idle 4090. That is precisely the regime where value-based methods should win. **But first fix the replay ratio (§5.4): `train_freq=4, gradient_steps=1, N_ENVS=16` = one gradient step per 64 transitions.** Raising `gradient_steps` to 8–16 is one line and must be measured before any Rainbow work is justified.

**Caveat:** replay buffers interact badly with the auto-curriculum — the savestate distribution shifts underneath you, so old transitions are off-distribution in a way PPO's on-policy rollouts never are.

**Verdict: do it, incrementally.** (a) fix the replay ratio, measure; (b) n-step returns (~150 lines); (c) PER (~300 lines); (d) stop.

### 8.2 Discrete SAC — *skip it*

**Where you are:** the scaffold exists and is deliberately dead — [sac/agent.py:113](src/agents/sac/agent.py:113) and `:334` both raise `NotImplementedError`. Task 8 guarded **both** entrypoints specifically so `tune.py --algo sac` could not spin up 16 emulators on a broken objective. The plan lists it as out of scope.

**Cost:** SB3's SAC is continuous-only. Christodoulou's formulation ([arXiv:1910.07207](https://arxiv.org/abs/1910.07207)) changes the actor to a categorical, has the critics emit Q for all actions, and computes entropy and soft-value expectations exactly rather than by sampling — a custom policy plus a rewritten `train()`, 400–600 lines, none reusable from SB3.

**Why skip:** discrete SAC has a poor reliability record. The original paper's Atari results were notably weak, and the follow-up literature (e.g. Zhou et al., *Revisiting Discrete SAC*, 2022) exists **because** vanilla discrete SAC is unstable — automatic α tuning against a discrete target entropy tends either to collapse to greedy or to refuse to sharpen, needing entropy-penalty and double-average-Q corrections to be usable. You would debug α dynamics instead of fighting-game behaviour.

**And the payoff is already available.** SAC's motivation is maximum-entropy exploration; PPO with `ent_coef` gives entropy regularisation directly, and Task 4 made that coefficient explicit and tunable. If the diagnosis is "not enough exploration," raise `ent_coef` — a config change, not 600 lines. **Keep the guards in place.**

### 8.3 DreamerV3 — *highest ceiling, run it on the fast backend first*

**Why it fits this bottleneck better than anything else:** Dreamer ([arXiv:2301.04104](https://arxiv.org/abs/2301.04104)) learns a world model, then trains the actor-critic almost entirely **inside imagined rollouts on the GPU**. The scarce resource here is emulator steps; the abundant one is a 4090. Dreamer is the one algorithm that converts abundant compute into a substitute for scarce environment interaction. Its default train ratios (tens to hundreds of gradient steps per env step) are absurd for most setups and near-free for this one.

**Task 6 accidentally made it tractable.** Dreamer is usually pitched at pixels, where the decoder reconstructs 64×64×3. **The v4 observation is 23 floats per frame** — a vector Dreamer reconstructs 23 numbers, a dramatically easier world-modelling problem than any published Dreamer benchmark. If model-based was ever going to work here, v4 is why it is now plausible.

**Why it still ranks behind Rainbow:**

- **Adversarial nonstationarity is the classic model-based failure mode.** The opponent's next action is not a function of the observed state — it is the CPU AI's internal logic, or another network in league play. The RSSM will model it as high-entropy noise, imagined rollouts will feature a mush opponent, and the actor will learn to **exploit model error rather than the game**. This is why competitive-game results stay model-free: **FightLadder** ([arXiv:2406.02081](https://arxiv.org/abs/2406.02081)) evaluates five methods on SF2/SF3/KOF and **all five use PPO as the backbone.** There is no published Dreamer fighting-game result.
- **Discontinuous dynamics.** Hit detection, frame-perfect transitions and the `action_id` categoricals are sharp discrete events; reconstruction losses smooth exactly the things that decide rounds.
- **Framework cost.** No SB3 path. Realistically `danijar/dreamerv3` (JAX) or **SheepRL** (PyTorch/Lightning, Gymnasium-native, vector-obs support, actively maintained) — SheepRL is the pragmatic choice, but it is a second stack with its own replay, config system and env plumbing beside the lock-step socket bridge, `SelectiveVecNormalize`, curriculum callbacks and Elo module.

**Verdict:** do not start it on the BizHawk rig. Run it on the Stage-3 stable-retro backend where a result takes hours. **If Dreamer cannot beat PPO there, it will not here.**

### 8.4 EGGROLL — the arithmetic

Assume the optimistic end of Stage 3: **10,000 agent steps/s**.

- One ES generation = one full episode per population member. Mean episode ≈ **570 steps**. A modest population of 256 → **~146,000 steps per single update**.
- At 10 k steps/s that is **~15 s per generation**. ES on control tasks typically needs **thousands** of generations; at 2,000 that is **~292 M environment steps and 8+ hours** for one run.
- PPO extracts ~320 gradient updates from the same 146 k steps, and the plan targets a useful policy at **6 M steps**.

**That is a ~50× sample-efficiency gap, and a 4× throughput win does not close it.** The paper's own claim for EGGROLL in RL is **parity with OpenES, not superiority over PPO**. Note also that the "50,000+ FPS" target, read as emulator frames, is **12,500 agent steps/s — below gate 1's 25,000.** The plan's original verdict — *park it; revisit only at hundreds of parallel emulators* — survives the throughput argument intact.

**Where it could genuinely earn its place:** as a **gradient-free fine-tuning pass on a frozen, already-trained policy**, perturbing only the last layer, optimising Elo or win rate directly. See gate 5 in §7.6.

### 8.5 Also on the radar (from the plan's research section)

- **Group Policy Gradient (GPG)** ([arXiv:2510.03679](https://arxiv.org/abs/2510.03679)) — keeps PPO's clipped objective but drops the critic, computing advantages from grouped Monte-Carlo returns. Shines with many cheap parallel rollouts of the *same* initial state; this project has 16 expensive emulators and per-state savestate randomisation. **Controlled experiment later, not a rewrite.** Becomes more interesting *after* Stage 3.
- **Action masking** with `MaskablePPO` ([arXiv:2006.14171](https://arxiv.org/abs/2006.14171); follow-up [arXiv:2603.09090](https://arxiv.org/abs/2603.09090)) — needs a per-state legality oracle (e.g. mask specials while airborne). The v4 payload now carries the airborne flags that make this possible. Relevant **because** macros exist: you want to mask a macro that cannot execute.
- **Learned action duration** ([arXiv:2605.20911](https://arxiv.org/abs/2605.20911), runs on FightLadder/SF2) — a second head predicting how long to hold each action. Its key empirical finding — **agents perform best with consistently high frame-skip** — validates `FRAME_SKIP = 4`. Natural successor to fixed-length macros.
- **SimbaV2** ([arXiv:2502.15280](https://arxiv.org/abs/2502.15280)) — hyperspherical normalisation for scalable deep RL; cheap architectural change if network scaling ever becomes the goal (relevant to EGGROLL gate 2).

---

## 9. Everything already tried that failed

Kept deliberately. These are the dead ends, so nobody repeats them.

### 9.1 Errors in the plan itself (all caught in review, all fixed)

| What was wrong | Why it mattered |
|---|---|
| Claimed specials were "unreachable" at ~1 in 250,000 | **Wrong.** Assumed all 3 steps needed an exact 63-way match, but the button is free on the setup steps. Real odds ≈ **1 in 1,700**. The barrier is **credit assignment**, not reachability (§3.6). |
| `BUFFER_SIZE = 1_000_000` | Allocates **17.7 GB** — SB3 allocates both `observations` and `next_observations`, and the v2/v3 obs is 2216 float32. `--algo dqn` **could not have started**. Now 250 k with the arithmetic documented. |
| E4 described only the *both-sided* HP sentinel | The **single-sided** case is more likely and worse: it zeroed one side's HP, produced `ko=True`, and fabricated an `info["loss"]=1` worth **−127 reward from a menu frame**, labelled a clean terminal. |
| Task 3 set reward γ=0.99; Task 4 set PPO γ=0.995 | Broke PBRS invariance. Residual is a real per-step penalty **maximal at exactly `peak_dist=70`** — ≈ −7.1/round for holding poke range vs 0 for camping at max distance, **pointing backwards**. The telescoping test passed only because it constructs γ=1.0. |
| Asserted Task 8 would clean up DQN/SAC's `> 0.0` sentinels | It does not. Task 4's switch to `None` sentinels broke `--algo dqn` / `--algo sac` with `TypeError`, and would have survived to the end of the plan. |
| SAC guard scoped to `train()` only | `tune.py --algo sac` would still spin up 16 emulators on the broken objective. A guard covering one of two entrypoints reads as protection that is not there. |
| Task 8 set `learning_starts=10_000` in the trainer only | The Optuna study fell back to QRDQN's default of 100 — a full study tuned under a warmup regime production never uses. **No error, no warning.** |
| Test files were gitignored (`/code_testing/pytest`) | Every `git add` of a test would have silently done nothing. |
| Task 1's fake did not mirror `bizhawk_path`/`rom_path`/`lua_path` | `reset()`'s self-healing path catches `RuntimeError`/`OSError` and calls `_start_emulator_bridge()` → live `socket.bind()` + `Popen(EmuHawk.exe)`. Now overridden to raise. |

### 9.2 A crash introduced during implementation

**v4 corrupt-payload path killed the worker.** `_parse_payload`'s failsafe returns `frames[-1][:554]`, which for v4 is a 23-element array. `sf2_v4._parse_payload` then ran `np.argmax` over slices including `full[266:522]` — **empty** → `ValueError`, raised *outside* `step()`'s try block, taking down a `SubprocVecEnv` rank. Dormant only until the first successful step. Fixed with a pass-through guard + regression test.

### 9.3 A regression caused by one of the fixes

**Fixing the γ mismatch broke four other entrypoints.** Raising `RewardConfig.gamma` 0.99 → 0.995 desynced `pbt_orchestrator.py:148`, `train_exploiter.py:243` and `train_league.py:246`, which all hardcoded 0.99 and had been *consistent before*. This **sign-flipped the residual into a +7.1/round stall incentive** at `peak_dist` — it paid the agent to camp and do nothing. **Arguably worse than the original bug.**

Now routed through `AGENT_GAMMA`, with an **AST guard test** that fails if anyone reintroduces a discount literal anywhere in `src/`. Point fixes do not stop the next one.

### 9.4 The process lesson

**Every Critical finding was a seam between two individually-correct tasks.** Nothing inside any single task's diff was wrong. Per-task review structurally cannot see these. `test_cross_module_contracts.py` exists specifically to guard those seams — **extend it whenever you add a cross-module invariant.**

---

## 10. Open items (none committed-broken; all are scope the plan did not cover)

1. **`src/envs/league_env.py` still carries the old broken reward** — dead-zone potential at [line 271](src/envs/league_env.py:271), hardcoded `0.99` discount at 275, `−0.015`/step at 297, no sentinel handling, divergent constants (±50 terminals vs the fixed +65/−50). **Self-play currently trains against the exact bug this branch fixed.** `league_env.step` already delegates parsing to the base `_parse_payload`, so porting it to `envs/reward.py` is small.
2. **No `v4` / `--macros` support in evaluation, tuning or league paths** — `test_agent_v2.py:45`, `test_ai_vs_ai_v2.py:153/159`, `train_exploiter.py:156`, `train_league.py:156` and **four dropdowns in `web_dashboard.py`** (1297, 1335, 1379, 1393) all gate on `["v2","v3"]`.
3. **`pbt_orchestrator.py:148` still has `n_epochs=10, target_kl=0.03`.**
4. **Optuna tunes γ while the shaping γ is pinned** — all three studies. See §5.4.
5. **`README.md` not updated** — zero occurrences of `sf2_v4`, `--macros`, or `--recurrent`. The widened Lua payload is a deployment-relevant fact that appears nowhere user-facing.
6. **Sentinel frames still carry a fabricated HP of 0 in the *observation*.** The fix applied "sentinel means HP unknown" to the reward and termination sides only; `_parse_payload` still zeroes `raw[0]`/`raw[1]`, so the policy sees 0 HP for up to `NUM_FRAMES` steps. Pre-existing, not a regression, but the principle is only half-applied.
7. **`core/elo.py` is wired into nothing** — no caller anywhere in `src/`.
8. **Plan checkboxes never ticked** — 84 unchecked, 0 checked.
9. **`macro_wrapper.py:26` docstring says `frame_size` is "14 for v4"**; the real value is 23 and `env_tools.py:40` passes `V4_FRAME_DIM` correctly. **Code is right, docstring is stale.**
10. **`.venv` drifts from `requirements.txt`** — `pytest` was missing once despite being pinned; `sb3-contrib` had to be installed mid-task. Run `pip install -r requirements.txt` on the training machine.
11. **Branch not merged.** `git log --oneline 11552e8c..HEAD` for the 24 commits. Options: merge to `main`, open a PR, or leave for review.

---

## 11. Reference

### 11.1 Key numbers

| Quantity | Value |
|---|---|
| `AGENT_GAMMA` | 0.995 (the only discount literal) |
| `FRAME_SKIP` / `NUM_FRAMES` / `ACTION_DIM` | 4 / 4 / 10 |
| `N_ENVS` | 16 |
| `MAX_STEPS_PER_ROUND` | 1500 |
| PPO rollout | 2048 × 16 = 32,768; batch 1024; 4 epochs → 128 updates/rollout |
| Obs width v2/v3 → v4 | 2216 → **92** floats |
| Policy params v2/v3 → v4 | ≈ 3.1 M → **≈ 1.45 M** |
| `rel_dist` domain | 0–187, clips at 187; baseline median **83**, **52.2 %** of steps ≥ 80 |
| Shaping | `peak_dist=70`, `spacing_weight=2.5`, two-sided |
| Mean episode | ~570 agent steps |
| Throughput, 1 emulator | ~165 agent steps/s **(laptop — re-measure)** |
| Throughput, 16 aggregate | **UNMEASURED** |
| Tests | 110, offline, 81 s |

### 11.2 New files this branch (25)

**Core:** `src/envs/reward.py`, `action_macros.py`, `macro_wrapper.py`, `sf2_v4.py`; `src/core/sf2_extractor.py`, `rl_constants.py`, `elo.py`; `src/agents/ppo/hyperparams.py`; `src/agents/common/action_wrappers.py` (+ `__init__.py`).

**Tests:** `code_testing/pytest/fakes/fake_bizhawk.py`, `test_env_reward.py`, `test_action_macros.py`, `test_macro_wrapper.py`, `test_sf2_extractor.py`, `test_ppo_hyperparams.py`, `test_action_wrappers.py`, `test_elo.py`, `test_cross_module_contracts.py`, `test_dqn_config.py`, `test_dqn_qrdqn_warmup_parity.py`.

**Modified:** `lua/v2.0/training_env_client.lua` (payload 13→24), `src/envs/base_env.py` (heaviest — `step()` and `_parse_payload` touched by Tasks 2, 3, 5, 6 and both fix passes), `src/core/env_tools.py`, `src/scripts/train.py`, `src/agents/ppo/agent.py`, `src/agents/dqn/{agent,config,optuna_study}.py`, `src/agents/sac/{agent,optuna_study}.py`, `src/agents/pbt/pbt_orchestrator.py`, `src/scripts/{train_league,train_exploiter}.py`, `requirements.txt`, `.gitignore`.

**Deliberately untouched** (would break saved models or was out of scope): `src/envs/sf2_v1.py`, `sf2_v2.py`, `sf2_v3.py`, `src/core/config.py`, `src/envs/league_env.py`, `README.md`.

### 11.3 Commands

```bash
.venv/Scripts/python.exe -m pytest code_testing/pytest -q
```

Stage 1 validation run (do this before anything else):

```bash
.venv/Scripts/python.exe src/scripts/train.py --algo ppo --env v3 --auto_curriculum --steps 1000000 --device cpu
```

Flagship configuration (only after Stage 1 succeeds):

```bash
.venv/Scripts/python.exe src/scripts/train.py --algo ppo --env v4 --macros --auto_curriculum --steps 6000000 --device cpu
```

### 11.4 References

1. Schulman et al. (2017). *Proximal Policy Optimization Algorithms*. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
2. Schulman et al. (2015). *High-Dimensional Continuous Control Using GAE*. [arXiv:1506.02438](https://arxiv.org/abs/1506.02438)
3. Huang, Dossa, Raffin, Kanervisto & Wang (2022). *The 37 Implementation Details of PPO*. ICLR Blog Track.
4. Li et al. (2024). *FightLadder: A Benchmark for Competitive Multi-Agent RL*. [arXiv:2406.02081](https://arxiv.org/abs/2406.02081) · [code](https://github.com/wenzhe-li/FightLadder)
5. Nguyen, Driessens & Soemers (2026). *For How Long Should We Be Punching?* [arXiv:2605.20911](https://arxiv.org/abs/2605.20911)
6. **EGGROLL** — *Evolution Strategies at the Hyperscale* (2025). [arXiv:2511.16652](https://arxiv.org/abs/2511.16652) · [project page](https://eshyperscale.github.io/)
7. Sutton, Precup & Singh (1999). *Between MDPs and semi-MDPs*. Artificial Intelligence 112(1-2):181-211.
8. Ng, Harada & Russell (1999). *Policy invariance under reward transformations*. ICML.
9. Huang & Ontañón (2020). *A Closer Look at Invalid Action Masking*. [arXiv:2006.14171](https://arxiv.org/abs/2006.14171)
10. *Overcoming Valid Action Suppression in Unmasked Policy Gradient Algorithms* (2026). [arXiv:2603.09090](https://arxiv.org/abs/2603.09090)
11. Vinyals et al. (2019). *Grandmaster level in StarCraft II*. **Nature 575, 350-354**.
12. Christodoulou (2019). *Soft Actor-Critic for Discrete Action Settings*. [arXiv:1910.07207](https://arxiv.org/abs/1910.07207)
13. Dabney et al. (2018). *Distributional RL with Quantile Regression* (QR-DQN). [arXiv:1710.10044](https://arxiv.org/abs/1710.10044)
14. Hessel et al. (2017). *Rainbow: Combining Improvements in Deep RL*. [arXiv:1710.02298](https://arxiv.org/abs/1710.02298)
15. Kapturowski et al. (2019). *Recurrent Experience Replay in Distributed RL* (R2D2). ICLR.
16. Hafner et al. (2023). *Mastering Diverse Domains through World Models* (DreamerV3). [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)
17. Palanisamy et al. (2022). *DIAMBRA Arena*. [arXiv:2210.10595](https://arxiv.org/abs/2210.10595)
18. *Group Policy Gradient* (2025). [arXiv:2510.03679](https://arxiv.org/abs/2510.03679)
19. *Evolutionary Policy Optimization* (2025). [arXiv:2503.19037](https://arxiv.org/abs/2503.19037)
20. Lee et al. (2025). *Hyperspherical Normalization for Scalable Deep RL* (SimbaV2). [arXiv:2502.15280](https://arxiv.org/abs/2502.15280)
21. Ye et al. (2019). *Mastering Complex Control in MOBA Games with Deep RL*. [arXiv:1912.09729](https://arxiv.org/abs/1912.09729)
22. Farama **stable-retro** — [code](https://github.com/Farama-Foundation/stable-retro)
23. Raffin et al. *Stable-Baselines3 Contrib*. [docs](https://sb3-contrib.readthedocs.io/)
24. **SheepRL** — PyTorch DreamerV3 with Gymnasium support. [code](https://github.com/Eclectic-Sheep/sheeprl)

---

## 12. If you are the next model, start here

1. Read §5.1. Write the TensorBoard callback. **Nothing else is measurable until you do.**
2. Re-measure throughput on the 13900K (§6). Record the aggregate 16-env number.
3. Run Stage 1 (§7.2). Watch mean `rel_dist`. **This validates or refutes the project's core diagnosis.**
4. Fix the QR-DQN replay ratio (§5.4) — one line, before evaluating any DQN-family algorithm.
5. Then, and only then, pick from §7.3 (throughput) or §8 (algorithms).

Do not start EGGROLL. Read §7.6 first and check the gates honestly.
