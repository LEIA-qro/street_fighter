import os
import sys

# Add src to path
sys.path.append(os.path.abspath('src'))

from envs.sf2_v1 import StreetFighterEnv
from envs.sf2_v2 import StreetFighterEnvV2
from envs.base_env import StreetFighterBaseEnv

print("Imports successful!")

v1_env = StreetFighterEnv(trainable=False)
print("V1 Observation Space:", v1_env.observation_space)
print("V1 Action Space:", v1_env.action_space)

try:
    v1_env.close()
except Exception as e:
    pass

print("Tests passed.")
