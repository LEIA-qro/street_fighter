import os
import sys
from pathlib import Path

# Add src to path
SRC_DIR = str(Path(__file__).resolve().parents[2] / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from envs.sf2_v1 import StreetFighterEnv
from envs.sf2_v2 import StreetFighterEnvV2
from envs.base_env import StreetFighterBaseEnv

print("Imports successful!")

v1_env = StreetFighterEnv(trainable=False)
print("V1 Observation Space:", v1_env.observation_space)
print("V1 Action Space:", v1_env.action_space)

try:
    v1_env.close()
except Exception:
    pass

print("Tests passed.")
