# Design Document: Observation & Neural Network Telemetry Visualization

## 🎯 Goal Description
This document specifies the technical design for a real-time, high-fidelity observation space and neural network activation visualizer integrated into the Gradio Web Control Center of the Street Fighter II RL pipeline. 

The goal is to allow developers to inspect exactly what the AI agent is "seeing" in its stacked-frame observation vector across the three active environment versions (`v1`, `v2`, `v3`) during interactive match testing, as well as visualize how the policy networks are processing the state (predicting action probabilities and state values).

---

## 🏛️ System Architecture & Data Flow

The system uses a lock-step, file-based inter-process communication (IPC) protocol between the environment execution process (`test_agent_v2.py` / `test_ai_vs_ai_v2.py`) and the Gradio Dashboard server (`web_dashboard.py`).

```
+-----------------------------------+             +----------------------------------+
|      Game Environment Loop        |             |         Gradio Dashboard         |
|  (test_agent_v2.py / EmuHawk)     |             |        (web_dashboard.py)        |
+-----------------------------------+             +----------------------------------+
|                                   |             |                                  |
| 1. Step Environment & Infere      |             |                                  |
| 2. Extract Value/Action Probs    |             |                                  |
| 3. Serialize Telemetry State      |             |                                  |
|                                   |             |                                  |
|        Write JSON Payload         |             |        gr.Timer Tick (100ms)     |
|  +-----------------------------+  |             |  +----------------------------+  |
|  |     .telemetry.json         |==================>| Read & Parse Telemetry     |  |
|  +-----------------------------+  |             |  | Render Arena SVG/HTML      |  |
|                                   |             |  | Update HTML component      |  |
|                                   |             |  +----------------------------+  |
+-----------------------------------+             +----------------------------------+
```

### 1. Telemetry Capture (Match Test side)
During execution of `test_agent_v2.py` or `test_ai_vs_ai_v2.py`, the active loop performs a single-frame step. On each step, the script:
1.  Queries the loaded Stable-Baselines3 model for the **Value Function Estimate** of the current stacked observation vector:
    ```python
    value_estimate = float(model.policy.predict_values(obs).squeeze().item())
    ```
2.  Extracts the raw **probability distributions** from the Actor policy network's output head:
    *   **v3 (MultiDiscrete)**: Retrieves the probability tensor for the 9-dimensional directional head and the 7-dimensional button action head.
    *   **v2 (MultiBinary)**: Retrieves the individual Sigmoid activation probabilities for the 10 button bits.
    *   **v1 (Continuous Arena Only)**: Skips policy distributions (continuous-only tracking).
3.  Serializes the active frame stack (containing the last 4 stacked frames from `self.frames`) along with their decoded variables (continuous positions, projectiles, velocities, and one-hot decoded active Action/Character names) and the neural net telemetry into `.telemetry.json` at the project root.

### 2. Live Rendering (Dashboard side)
Under `src/scripts/web_dashboard.py`, a new tab **"🔮 Observation Telemetry"** is added:
*   A background `gr.Timer(value=0.1)` triggers a rendering hook every 100ms.
*   The hook reads `.telemetry.json`. If found and fresh, it builds a rich CSS/HTML layout representing the 2x2 stacked frames.
*   **The 2x2 Grid Layout**:
    *   Renders a vector-drawn stage for each frame with **AI** (P1) and **OPP** (P2) bounding boxes.
    *   Draws screen boundaries (Left/Right corners) and projectile positions.
    *   Displays 100% of the raw observations (HP, coordinates, velocities, decoded action IDs, character IDs) in a tabular overview.
*   **The Activation Panel**:
    *   Renders horizontal bar charts representing active policy output probabilities.
    *   Renders a linear **Value Function Estimate Gauge** ranging from `-50.0 (Danger)` to `+60.0 (Advantage)` representing the agent's internal value optimism.
*   **State Clean Up**: When a test process stops, the telemetry file is deleted or set to `stale`, causing the dashboard to transition back to a clean standby card.

---

## 💾 Serialization Schema (`.telemetry.json`)

The serialized telemetry state is written as a compact JSON object:

```json
{
  "model_name": "string",
  "env_version": "v1 | v2 | v3",
  "status": "PLAYING | PAUSED | STOPPED",
  "value_estimate": 12.45,
  "policy_distributions": {
    "directions": [0.08, 0.0, 0.14, 0.72, 0.01, 0.0, 0.0, 0.05, 0.0],
    "buttons": [0.88, 0.03, 0.01, 0.0, 0.07, 0.01, 0.0]
  },
  "frames": [
    {
      "frame_index": 3,
      "p1_hp": 140,
      "p2_hp": 126,
      "rel_x": 65,
      "rel_y": 0,
      "p1_corner_dist": 120,
      "p1_proj": -1,
      "p2_proj": 136,
      "p1_vel_x": 0,
      "p2_vel_x": -5,
      "rel_dist": 65,
      "p1_action_name": "Idle (0)",
      "p2_action_name": "Fireball (48)",
      "p1_char_name": "Ryu",
      "p2_char_name": "Ken"
    },
    "... [frames index 2, 1, 0]"
  ]
}
```

---

## 🎨 UI Component Mapping

1.  **Gradio HTML Component**: Employs a custom CSS grid in `gr.HTML` utilizing HSL tailored colors (`#080b13` primary, `#101626` secondary) for a premium dark mode look.
2.  **Vector Arena**: The stage lines are vector lines drawn dynamically inside `gr.HTML`. Fighter bounding boxes (`AI` and `OPP`) are absolute positioned divs based on relative X values, scaled to fit the 100% width display.
3.  **Advantage Gauge**: A linear bar styled with a color gradient representing `-50.0` (danger, red) through `0.0` (neutral, slate) to `+60.0` (advantage, green). A white pointer absolute-positioned at the calculated percentage represents the value head prediction.

---

## 🧪 Verification Plan

### 1. Automated Telemetry Unit Tests
*   Verify that `test_agent_v2.py` outputs a valid JSON file under `.telemetry.json` with correct keys upon starting.
*   Verify that values in `.telemetry.json` perfectly match the environment's `obs` vector indexes.
*   Verify that the model value prediction does not output `NaN` or crash on multiple formats.

### 2. Manual UI Verification
*   Launch `python src/scripts/web_dashboard.py` and inspect the "🔮 Observation Telemetry" tab. It should show the "Standby" message.
*   Start a match test from the UI. The tab should immediately change to show the 2x2 grid representing the running match test.
*   Verify that P1 is labeled "AI" and P2 is labeled "OPP".
*   Inspect the detailed stats table inside the 2x2 grid frames; ensure all 10 coordinates, actions, and character IDs are visible and dynamically updating.
*   Verify that the Advantage gauge moves dynamically corresponding to Ryu's relative life and positioning advantage.
