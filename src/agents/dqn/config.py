# ===========================================================
#                   Optuna Config
# ===========================================================   

N_HYPERPARAMETER_TRIALS = 50 

# --- HYPERPARAMETERS FROM OPTUNA TRIAL ---
LR = 1e-4
# With N_ENVS=16, 16 transitions land per env step, so 100k held only ~6,250
# iterations -- minutes of wall clock against a 96-savestate curriculum.
#
# SB3's ReplayBuffer.__init__ allocates (buffer_size // n_envs, n_envs,
# *obs_shape) for BOTH observations and next_observations
# (optimize_memory_usage defaults to False), so n_envs cancels out of the
# total footprint entirely. The v2/v3 observation is 554 floats x 4 stacked
# frames = 2216 float32:
#   1_000_000 x 2216 x 4 bytes x 2 = 17.7 GB  <- the old value; cannot start
#                                                alongside 16 live emulators.
#     250_000 x 2216 x 4 bytes x 2 =  4.4 GB  <- current value, still 2.5x
#                                                the old 100k history depth.
# The constraint is observation WIDTH, not an arbitrary cap on this number:
# the v4 compact observation is 92 floats (23 x 4 frames), where even
# 1_000_000 x 92 x 4 bytes x 2 = 0.74 GB. Raising this again for v2/v3 means
# accepting the corresponding memory cost above, or moving DQN to v4.
BUFFER_SIZE = 250_000
BATCH_SIZE = 256
GAMMA = 0.99
EXPLORATION_FRACTION = 0.1
EXPLORATION_INITIAL_EPS = 1.0
EXPLORATION_FINAL_EPS = 0.05

NET_ARCH = [512, 512, 256]

WIN_RATE_THRESHOLD = 0.75   
WIN_RATE_WINDOW    = 250    

OPTUNA_PHASE1_LR = LR
OPTUNA_PHASE1_EXPL_FRAC = EXPLORATION_FRACTION

TRANSFER_LR = 5e-5
TRANSFER_EXPL_FRAC = 0.05

PHASE_HYPERPARAMS = {
    0: {"lr": OPTUNA_PHASE1_LR,          "exploration_fraction": OPTUNA_PHASE1_EXPL_FRAC},
    1: {"lr": OPTUNA_PHASE1_LR * 0.85,   "exploration_fraction": OPTUNA_PHASE1_EXPL_FRAC * 0.8},
    2: {"lr": TRANSFER_LR,               "exploration_fraction": TRANSFER_EXPL_FRAC},
    3: {"lr": TRANSFER_LR * 0.75,        "exploration_fraction": TRANSFER_EXPL_FRAC * 0.75},
}
