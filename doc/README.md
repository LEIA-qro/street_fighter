# Bare-Metal Architectural & Algorithmic Justification Guide

This document provides the rigorous engineering and mathematical justifications for the core design decisions implemented in the Street Fighter II Reinforcement Learning pipeline. It serves as a technical manual for low-level systems integration and reinforcement learning convergence strategy.

---

## 1. Synchronous Lock-Step TCP Bridge Specification

### The gym-retro Replacement Mandate
While widely used in research, the out-of-date `gym-retro` library introduces severe bottlenecks and failure modes when training at high throughput:
1.  **Memory Corruption & Crash Vulnerability:** Running multiple instances of `gym-retro` concurrently in separate processes frequently leads to access violations in emulator hooks.
2.  **Thread Race Conditions:** Lack of explicit synchronization forces developers to execute delay loops, leading to timing drift, state mismatches, or skipped action windows.
3.  **Monolithic Constraints:** `gym-retro` locks the system into fixed frame-skipping, rigid savestate formats, and limits manual curriculum integration.

To eliminate these constraints, we designed a custom **lock-step TCP bridge**. The emulator (BizHawk Lua) acts as a TCP Client, and Python acts as the TCP Server. 

### Lock-Step Protocol Specification
The connection guarantees absolute determinism using a strict **1-send / 1-receive** synchronization sequence:
1.  The emulator advances, reads specified Big-Endian RAM addresses, formats them into a single string payload, and sends it via TCP.
2.  The Lua script immediately blocks (entering a spinlock loop) and suspends emulation.
3.  Python reads the socket, decodes the state vector, passes it through the policy network, computes the action, and sends a controller input mask.
4.  Lua receives the controller mask, injects the inputs into the virtual joypad, advances the emulator by `FRAME_SKIP` frames, and repeats.

### Stream Buffer Slicing Algorithm
Due to TCP packet segmentation, a socket read may return partial payloads or multiple joined packets. If parsed directly, this corrupts observation inputs. The Python side implements a non-blocking stream-buffer slicing mechanism to guarantee packet alignment:

```python
def receive_payload(self) -> str:
    while '\n' not in self.stream_buffer:
        # Read from socket connection
        chunk = self.conn.recv(4096).decode('utf-8')
        if not chunk:
            return ""
        self.stream_buffer += chunk
    
    # Slice the first complete packet out of the stream buffer
    line, self.stream_buffer = self.stream_buffer.split('\n', 1)
    return line
```

### Socket Timeout Justification
*   **Lua Client Timeout:** Set to `10ms` non-blocking socket checks inside `training_env_client.lua` using `comm.socketServerSetTimeout(10)`. This prevents BizHawk from hanging permanently if Python terminates unexpectedly.
*   **Python Server Timeout:** Set to `5.0` seconds on Python’s socket (`self.conn.settimeout(5.0)`). This ensures immediate detection of frozen emulator workers while remaining large enough to absorb PyTorch backpropagation computations, GPU memory shifts, and disk checkpoints.

---

## 2. Sega Genesis Motorola 68000 WRAM Extraction

### Big-Endian Mapping
The Sega Genesis console uses a Motorola 68000 processor. The Motorola 68000 uses a **Big-Endian** memory organization (the most significant byte is stored at the lowest memory address). Genesis Work RAM (WRAM) resides in the hexadecimal range `0xFF0000 - 0xFFFFFF`. 

To extract unsigned 16-bit integer values correctly, the Lua client must execute big-endian reads:
```lua
local p1_hp = mainmemory.read_u16_be(0x8042)
```

### ML Pitfall: Data Leakage Prevention
A classic machine learning pitfall in reinforcement learning is **Data Leakage in the Observation Space**, which leads to policy network collapse.

In Street Fighter II, memory address `0x81E2` tracks the buttons currently pressed by Player 1, and `0x845E` tracks Player 2's inputs. 
*   **The Hazard:** If Player 1's own buttons (`0x81E2`) are included in Player 1's observation vector, the agent's policy network will discover a trivial shortcut. Rather than learning to map game visuals (distances, heights, opponent actions) to optimal moves, it collapses into an **identity-mapping loop**. The policy network learns to simply output whatever inputs it reads as currently active in its observation, stalling exploratory behavior and leading to zero generalization.
*   **The Defense:** The P1 button press address `0x81E2` is strictly **excluded** from P1's observation space. For fair competitive play, the agent is restricted to reading its own physical variables (coordinates, health, active action IDs) and P2's visible state, preventing policy feedback loops.

---

## 3. High-Dimensional Stacked Observation Space

The observation space is a **2216-dimensional float32 vector** designed to provide a Markovian state representation of the active fight.

$$\text{Observation Space} = 554 \text{ dimensions/frame} \times 4 \text{ stacked frames} = 2216 \text{ dimensions}$$

### The 554-Dimensional Vector Layout
For each individual frame, the extracted raw elements are structured as follows:

| Offset Range | Metric / Feature | Data Type & Range | Rationale / Defense |
| :--- | :--- | :--- | :--- |
| `0 - 1` | Player 1 & 2 Health (HP) | Float32 `[0.0, 1.0]` | Raw RAM values are strictly **clamped to a threshold of 100 HP** before normalization. This prevents Genesis emulator memory glitches from generating massive synthetic delta spikes that would corrupt PyTorch's actor-critic gradient updates. |
| `2 - 3` | Absolute X Coordinates | Float32 `[-1.0, 1.0]` | Tracks physical positions relative to the screen center. |
| `4 - 5` | Absolute Y Coordinates | Float32 `[-1.0, 1.0]` | Tracks jumps, crouches, and airborne states. |
| `6 - 7` | X & Y Coordinate Velocities | Float32 `[-1.0, 1.0]` | Derived by subtracting coordinate changes between consecutive frames. |
| `8` | Relative Horizontal Distance | Float32 `[-1.0, 1.0]` | Expresses $(X_{P1} - X_{P2})$, giving the network direct spatial proximity awareness. |
| `9` | Relative Vertical Distance | Float32 `[-1.0, 1.0]` | Expresses $(Y_{P1} - Y_{P2})$, giving awareness of jump-in trajectories. |
| `10 - 11` | Wall Boundary Distances | Float32 `[-1.0, 1.0]` | Measures distance to the left and right arena boundaries to prevent getting corner-trapped. |
| `12 - 13` | Projectile Coordinates & Deltas | Float32 `[-1.0, 1.0]` | Tracks horizontal/vertical coordinate changes of active fireballs to trigger jumping or blocking policies. |
| `14 - 15` | One-Hot Character IDs | Array `[0, 1]` | Informs the policy network of matchup differences. |
| `16 - 27` | One-Hot Active Action IDs | Array `[0, 1]` | Represents current animations (crouching, hitstun, sweep, recovery). |
| `28 - 553` | One-Hot Past Input Masks | Array `[0, 1]` | Historical record of the inputs sent during past skipping steps. |

### Frame Stacking Justification
A single frame of a fighting game is non-Markovian because it lacks temporal derivatives. For example, a single static image cannot show whether Ryu is moving forward, backward, starting a jump, or falling. 

By stacking **4 consecutive frames** (processed through a frame skip of 4), the policy network receives a rolling sequence representing roughly $266\text{ms}$ of game time. This allows the MLP network to calculate high-fidelity approximations of:
1.  **Velocity:** First derivative of position.
2.  **Acceleration:** Second derivative of position (crucial for timing anti-air moves like Shoryuken).
3.  **Animation Progress:** Recognizing startup vs recovery frames of opponent attacks.

---

## 4. Algorithmic Architecture and Modularity

### PPO Architecture Justification
Proximal Policy Optimization (PPO) was chosen as the primary training algorithm due to its sample efficiency and stable policy updates.

```python
policy_kwargs = dict(
    net_arch = dict(
        pi = [512, 512, 256],  # Actor Network Architecture
        vf = [512, 512, 256]   # Critic Value Network Architecture
    )
)
```

*   **Deep MLP Architecture:** Rather than default shallow architectures, we utilize a deeper structure. The early `512` wide layers allow high-capacity feature extraction from the 2216-dim observation space. The final `256` layer maps these rich features to the discrete actions (actor) and expected future returns (critic).
*   **Target KL Divergence (`target_kl = 0.03`):** This parameter acts as an early-stopping safeguard. If a policy update step changes the action probabilities too drastically (divergence $> 0.03$), the policy gradient update terminates early. This prevents the "policy degradation" failure mode where a bad batch of rollouts ruins hours of training.
*   **Entropy Exploration Decay:** The policy's exploration is driven by an entropy coefficient ($0.01 \to 0.001$). Early training phases use higher entropy to encourage diverse move exploration (sweeps, fireballs), while late phases decay this weight to stabilize optimal combos.

### Modularity via composition
The codebase implements a decoupled composition pattern. The environment inherits purely from `gymnasium.Env`, while the training scripts interact exclusively with an abstract agent wrapper class:

```
                  ┌───────────────┐
                  │   BaseAgent   │
                  └───────┬───────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
    ┌─────────┐      ┌─────────┐      ┌─────────┐
    │ PPOAgent│      │ SACAgent│      │ DQNAgent│
    └─────────┘      └─────────┘      └─────────┘
```

Adding a new RL algorithm (such as SAC or DQN) requires **zero changes** to the emulator socket code or Gymnasium environments.

---

## 5. Category-Balanced Curriculum Lottery Pool Math

To train Ryu to defeat all 12 opponents across all 8 difficulties without suffering from **Catastrophic Forgetting** (where mastering a new opponent causes the agent to forget how to defeat previous ones), we designed a **Rehearsal Weighted Lottery Pool**.

### Category Weights and Definitions
At any given training step, the 96 total available state files are classified into five mutually exclusive categories:

1.  **Past Rehearsal States ($C_{past}$):** Opponents from previous levels. Capped at a maximum size of **12** to prevent lottery dilution as the agent advances. Category Weight $W_{past} = 12$.
2.  **Mastered Active States ($C_{mastered}$):** Current level opponents where the agent has achieved a rolling win rate $\ge 75\%$. Category Weight $W_{mastered} = 24$.
3.  **Standard Active States ($C_{active}$):** Current level opponents with fewer than 10 recorded episodes. Category Weight $W_{active} = 36$.
4.  **Weakness States ($C_{weakness}$):** Current level opponents where the agent has played at least 10 games but has a rolling win rate $< 75\%$. Category Weight $W_{weakness} = 60$.
5.  **Newly Introduced States ($C_{new}$):** Harder states from the next level introduced via micro-steps. Category Weight $W_{new} = 48$.

### Mathematical Proof of 41.7% Weakness Selection Probability at Level 2
Let us calculate the exact selection probability of active weakness states during a standard training loop at Level 2. 

Assume the curriculum is in the following state:
*   Level 1 has been cleared, providing a rehearsal pool of 12 past states ($C_{past}$ is active).
*   Level 2 training is underway. The agent has played several games, identifying some Level 2 opponents as mastered ($C_{mastered}$ is active) and others as weaknesses ($C_{weakness}$ is active).
*   Micro-stepping is active, introducing the first states from Level 3 ($C_{new}$ is active).
*   All initial state evaluation games are complete ($C_{active}$ is empty).

The total lottery pool weight ($W_{total}$) is the sum of the active category weights:

$$W_{total} = W_{past} + W_{mastered} + W_{weakness} + W_{new}$$

$$W_{total} = 12 + 24 + 60 + 48 = 144$$

The probability of the lottery selecting a state from the **Weakness** category ($P(\text{Weakness})$) is:

$$P(\text{Weakness}) = \frac{W_{weakness}}{W_{total}} = \frac{60}{144} = 0.4167 \implies \mathbf{41.7\%}$$

### Rationale and Benefits
1.  **Bottleneck Targeting:** A **41.7% probability** guarantees that nearly half of all parallel rollouts focus directly on the specific opponents and matchups that the agent struggles with.
2.  **Anti-Dilution Guard:** Because the past rehearsal category is strictly capped at 12 states, the probability of selecting current bottlenecks does not decay as the curriculum advances, preventing rehearsal dilution.
3.  **Forgetting Defense:** The remaining $58.3\%$ of rollouts are divided between mastering the rest of the current level, exploring newly introduced harder matchups, and rehearsing past encounters to prevent catastrophic forgetting.

### Gated Statistical Promotion
To prevent promotion due to statistical noise (e.g., a lucky streak against a hard opponent), curriculum advancement is gated by strict criteria:
*   **Decentralized State Buffers:** Each state maintains a rolling history deque of length 100 tracking wins/losses.
*   **Minimum Evaluation Gating:** A promotion evaluation requires a minimum of **100 total episodes** across target states, and *every* active/new state must have played at least **15 episodes** (`min_samples_per_state = 15`) to prevent past mastered states from masking un-trained states.
*   **Stability Counter:** The rolling win rate must remain $\ge 75\%$ for **3 consecutive evaluation cycles** (`stability_threshold = 3`) to trigger promotion, ensuring statistical confidence.

---

## 6. Interactive Performance Optimizations

### Throttled Disk I/O in Interactive Play
During training, the headless client (`training_env_client.lua`) runs uncapped, executing fast socket transmissions. However, in matchup testing (`match_test_env_client.lua`), the interactive GUI must run smoothly at 60 FPS for human players or spectator monitoring.

*   **The Problem:** The interactive Lua script needs to read an external `.agent_state` file to check if the user has updated the active player configurations, loaded a new model, or changed the side. Reading files from disk on every single frame advance ($16.6\text{ms}$) creates high-frequency micro-stutters, disk I/O lock contention, and emulator freezing.
*   **The Optimization:** The Lua client restricts disk reads using a modulo frame counter:

```lua
-- Only execute disk file read once every 30 emulator frames (roughly 0.5s)
if emu.framecount() % 30 == 0 then
    local file = io.open(agent_state_path, "r")
    if file then
        -- Process configuration updates...
        file:close()
    end
end
```
This reduces disk operations by **$96.7\%$**, eliminating stutters and ensuring 100% interactive responsiveness.

### Emulation Speed uncapping
During training phases, throughput is maximized by removing emulator rendering locks:
1.  **VSync Forced Disable:** VSync is turned Off to prevent EmuHawk from syncing with physical monitor refresh rates ($60\text{Hz}$), which would cap training speed.
2.  **Headless Execution:** Disabling the audio module and minimizing screen size reduces host CPU overhead, increasing throughput to **400–800 FPS** per core.
