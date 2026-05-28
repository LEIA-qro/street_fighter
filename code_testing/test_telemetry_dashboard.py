import os
import json
import numpy as np
import pytest
import sys
from pathlib import Path

# Set up paths
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from core import config
from core.telemetry import write_telemetry, clean_telemetry

def test_clean_telemetry():
    target_path = os.path.join(config.PROJECT_ROOT, ".telemetry.json")
    with open(target_path, "w") as f:
        f.write("{}")
    assert os.path.exists(target_path)
    clean_telemetry()
    assert not os.path.exists(target_path)

def test_write_telemetry_v3_structure():
    # Create a mock 2216-dim observation (554 dims per frame * 4 stacked frames)
    # 10 continuous + 512 actions + 32 characters
    single_frame = np.zeros(554, dtype=np.float32)
    # Fill continuous: P1_HP=140, P2_HP=126, RelX=65, RelY=0, Wall=120, P1Proj=-1, P2Proj=136, P1Vel=0, P2Vel=-5, RelDist=65
    single_frame[:10] = [140.0, 126.0, 65.0, 0.0, 120.0, -1.0, 136.0, 0.0, -5.0, 65.0]
    # P1 action one-hot: index 12 (Crouch) = 1.0
    single_frame[10 + 12] = 1.0
    # P2 action one-hot: index 48 (Fireball) = 1.0
    single_frame[10 + 256 + 48] = 1.0
    # P1 char one-hot: index 0 (Ryu) = 1.0
    single_frame[10 + 512 + 0] = 1.0
    # P2 char one-hot: index 4 (Ken) = 1.0
    single_frame[10 + 512 + 16 + 4] = 1.0

    obs_mock = np.concatenate([single_frame] * 4) # Stack 4 times

    # Run telemetry writer with None model
    write_telemetry(
        model_name="test_model_v3",
        env_version="v3",
        status="PLAYING",
        model=None,
        obs=obs_mock,
        player=1
    )

    target_path = os.path.join(config.PROJECT_ROOT, ".telemetry.json")
    assert os.path.exists(target_path)

    with open(target_path, "r") as f:
        data = json.load(f)

    assert data["model_name"] == "test_model_v3"
    assert data["env_version"] == "v3"
    assert data["status"] == "PLAYING"
    assert data["value_estimate"] == 0.0
    assert len(data["frames"]) == 4

    # Validate parsed features of active frame (frame 0 in serialization is the latest)
    latest = data["frames"][0]
    assert latest["p1_hp"] == 140
    assert latest["p2_hp"] == 126
    assert latest["rel_x"] == 65
    assert latest["rel_y"] == 0
    assert latest["p1_corner_dist"] == 120
    assert latest["p1_proj"] == -1
    assert latest["p2_proj"] == 136
    assert latest["p1_vel_x"] == 0
    assert latest["p2_vel_x"] == -5
    assert latest["rel_dist"] == 65
    assert "Crouch" in latest["p1_action_name"]
    assert "Fireball" in latest["p2_action_name"]
    assert latest["p1_char_name"] == "Ryu"
    assert latest["p2_char_name"] == "Ken"

    # Cleanup
    clean_telemetry()
