import os
import json
import numpy as np
from core import config

CHAR_NAMES = {
    0: "Ryu", 1: "E.Honda", 2: "Blanka", 3: "Guile", 4: "Ken", 5: "Chun-Li",
    6: "Zangief", 7: "Dhalsim", 8: "M.Bison", 9: "Sagat", 10: "Balrog", 11: "Vega"
}

ACTION_NAMES = {
    0: "Idle / Neutral",
    1: "Crouch",
    2: "Walk",
    3: "Walk Backward / Block",
    4: "Jump Neutral",
    5: "Jump Forward",
    6: "Jump Backward",
    7: "Crouch Light Punch",
    8: "Standing Light Punch",
    9: "Standing Med Punch",
    10: "Standing Hard Punch",
    11: "Crouch Med Punch",
    12: "Crouch Hard Punch",
    14: "Jump Light Punch",
    15: "Standing Light Kick",
    16: "Standing Med Kick",
    17: "Standing Hard Kick",
    18: "Standing Close Hard Kick",
    19: "Standing Roundhouse Kick",
    24: "Crouch Hard Kick (Sweep)",
    25: "Jump Hard Kick",
    26: "Jump Diagonal Hard Kick",
    27: "Jump Light Kick",
    30: "Jump Forward Hard Kick",
    32: "Light Kick",
    36: "Standing Fierce Punch",
    40: "Crouch Kick",
    46: "Jump Fierce Punch",
    47: "Fireball (Startup)",
    48: "Fireball (Hadouken)",
    56: "Dragon Punch (Shoryuken)",
    64: "Hurricane Kick (Tatsumaki)",
    70: "Hitstun (Light)",
    72: "Hitstun (Heavy)",
    74: "Blockstun",
    80: "Knockdown / Fall",
    82: "Knocked Down (Grounded)",
    84: "Wakeup / Getup",
}

_write_step_counter = 0

def clean_telemetry() -> None:
    for filename in [".telemetry.json", ".telemetry.json.tmp"]:
        target_path = os.path.join(config.PROJECT_ROOT, filename)
        if os.path.exists(target_path):
            try:
                os.remove(target_path)
            except OSError:
                pass

def decode_one_hot(array) -> int:
    idx = int(np.argmax(array))
    if array[idx] > 0.5:
        return idx
    return 0

def write_telemetry(
    model_name: str,
    env_version: str,
    status: str,
    model,
    obs: np.ndarray,
    player: int = 1
) -> None:
    global _write_step_counter
    _write_step_counter += 1
    # Throttle disk writes to every 2 steps (~30 FPS) to eliminate I/O lock contention
    if _write_step_counter % 2 != 0:
        return

    target_path = os.path.join(config.PROJECT_ROOT, ".telemetry.json")
    temp_path = os.path.join(config.PROJECT_ROOT, ".telemetry.json.tmp")
    
    obs_flat = np.array(obs).flatten()
    
    if env_version == "v1":
        frame_dim = 10
    else:
        frame_dim = 554

    num_frames = len(obs_flat) // frame_dim
    frames_list = []

    for f_idx in range(num_frames):
        frame_slice = obs_flat[f_idx * frame_dim : (f_idx + 1) * frame_dim]
        
        p1_hp = int(frame_slice[0])
        p2_hp = int(frame_slice[1])
        rel_x = int(frame_slice[2])
        rel_y = int(frame_slice[3])
        p1_corner_dist = int(frame_slice[4])
        p1_proj = int(frame_slice[5])
        p2_proj = int(frame_slice[6])
        p1_vel_x = int(frame_slice[7])
        p2_vel_x = int(frame_slice[8])
        rel_dist = int(frame_slice[9])
        
        p1_act_name = "Idle (0)"
        p2_act_name = "Idle (0)"
        p1_char_name = "Ryu"
        p2_char_name = "Ken"

        if env_version in ["v2", "v3"] and len(frame_slice) == 554:
            p1_act_oh = frame_slice[10 : 10 + 256]
            p2_act_oh = frame_slice[10 + 256 : 10 + 512]
            p1_char_oh = frame_slice[10 + 512 : 10 + 512 + 16]
            p2_char_oh = frame_slice[10 + 512 + 16 : 10 + 512 + 32]
            
            p1_act_id = decode_one_hot(p1_act_oh)
            p2_act_id = decode_one_hot(p2_act_oh)
            p1_char_id = decode_one_hot(p1_char_oh)
            p2_char_id = decode_one_hot(p2_char_oh)
            
            p1_act_name = f"{ACTION_NAMES.get(p1_act_id, 'Other')} ({p1_act_id})"
            p2_act_name = f"{ACTION_NAMES.get(p2_act_id, 'Other')} ({p2_act_id})"
            p1_char_name = CHAR_NAMES.get(p1_char_id, f"Char {p1_char_id}")
            p2_char_name = CHAR_NAMES.get(p2_char_id, f"Char {p2_char_id}")
            
        p1_hp = 0 if p1_hp > 200 else p1_hp
        p2_hp = 0 if p2_hp > 200 else p2_hp

        frames_list.append({
            "frame_index": f_idx,
            "p1_hp": p1_hp,
            "p2_hp": p2_hp,
            "rel_x": rel_x,
            "rel_y": rel_y,
            "p1_corner_dist": p1_corner_dist,
            "p1_proj": p1_proj,
            "p2_proj": p2_proj,
            "p1_vel_x": p1_vel_x,
            "p2_vel_x": p2_vel_x,
            "rel_dist": rel_dist,
            "p1_action_name": p1_act_name,
            "p2_action_name": p2_act_name,
            "p1_char_name": p1_char_name,
            "p2_char_name": p2_char_name
        })
        
    frames_list.reverse()

    value_estimate = 0.0
    policy_distributions = {}

    if model is not None:
        try:
            import torch
            obs_tensor = torch.tensor(obs_flat[np.newaxis, :], device=model.device, dtype=torch.float32)
            
            with torch.no_grad():
                if hasattr(model, "policy") and hasattr(model.policy, "predict_values"):
                    value_tensor = model.policy.predict_values(obs_tensor)
                    value_estimate = float(value_tensor.squeeze().item())
            
            with torch.no_grad():
                if hasattr(model, "policy") and hasattr(model.policy, "get_distribution"):
                    dist = model.policy.get_distribution(obs_tensor)
                    if env_version == "v3":
                        dir_probs = dist.distribution[0].probs.detach().cpu().numpy()[0]
                        btn_probs = dist.distribution[1].probs.detach().cpu().numpy()[0]
                        policy_distributions = {
                            "directions": [float(p) for p in dir_probs],
                            "buttons": [float(p) for p in btn_probs]
                        }
                    elif env_version == "v2":
                        bin_probs = dist.distribution.probs.detach().cpu().numpy()[0]
                        policy_distributions = {
                            "buttons": [float(p) for p in bin_probs]
                        }
        except Exception as e:
            pass

    payload = {
        "model_name": model_name,
        "env_version": env_version,
        "status": status,
        "value_estimate": value_estimate,
        "policy_distributions": policy_distributions,
        "frames": frames_list,
        "player": player
    }

    try:
        with open(temp_path, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(temp_path, target_path)
    except (OSError, PermissionError):
        # File currently locked by Gradio dashboard reader on Windows
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
