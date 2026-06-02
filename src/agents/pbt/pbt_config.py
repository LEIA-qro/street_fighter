import os
from core.config import PORT

POPULATION_SIZE = 10          # Number of parallel agents
STEPS_PER_EXPLOIT = 500_000   # How often PBT pauses to rank and exploit
PERTURBATION_FACTORS = [0.8, 1.2]  # Multiply hyperparams by one of these
BOTTOM_FRACTION = 0.2         # Bottom 20% get overwritten
TOP_FRACTION = 0.2            # Top 20% are copied from

# PB2 Hyperparameter Search Space (continuous bounds for GP surrogate)
# Format: { "param_name": [min_value, max_value] }
PB2_HYPERPARAM_SPACE = {
    "lr":        [1e-6, 5e-4],
    "ent_coef":  [1e-8, 0.05],
    "clip_range": [0.1, 0.4],
}

# These are FIXED — PBT does not touch structural params
FIXED_PARAMS = {
    "n_steps":    2048,    # From ppo/config.py — DO NOT PERTURB
    "batch_size": 1024,    # From ppo/config.py — DO NOT PERTURB
}

# Port range safety check
# Agents use ports: config.PORT + 0 through config.PORT + (POPULATION_SIZE - 1)
# Ensure no port conflicts with other running processes before launch
# Current default PORT is 9999. Population of 10 uses 9999-10008.
