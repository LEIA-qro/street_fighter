# ===========================================================
#                   Optuna Config
# ===========================================================   

N_HYPERPARAMETER_TRIALS = 50 # Number of Optuna Trials to run during hyperparameter optimization

'''
Change this every optuna study
'''

# --- HYPERPARAMETERS FROM OPTUNA TRIAL ---
LR = 2.1083173532291324e-05
ENT_COEF = 0.015356816943688252
CLIP_RANGE = 0.26030337888734206
N_STEPS = 2048 # Once set DO NOT CHANGE
BATCH_SIZE = 1024 # Once set DO NOT CHANGE


NET_ARCH   = dict(pi=[512, 512, 256], vf=[512, 512, 256])

# Curriculum advancement gate
WIN_RATE_THRESHOLD = 0.75   # Must win 75% of episodes to advance
WIN_RATE_WINDOW    = 250    # Rolling window of episodes to measure

# Phase hyperparameter decay — applied relative to Optuna results
# Set these after your first Optuna run finishes
OPTUNA_PHASE1_LR         = LR          # Placeholder — update after first Optuna
OPTUNA_PHASE1_ENT_COEF   = ENT_COEF
OPTUNA_PHASE1_CLIP_RANGE = CLIP_RANGE

# Transfer Optuna results (Phase 3->4) — update after second Optuna run
TRANSFER_LR         = 2e-5    # Placeholder
TRANSFER_ENT_COEF   = 0.015
TRANSFER_CLIP_RANGE = 0.15



# Update PHASE_HYPERPARAMS to re-index:
PHASE_HYPERPARAMS = {
    # NOTE: n_steps and batch_size are FIXED after model creation.
    # SB3's rollout buffer is sized at init. Changing them mid-training
    # requires rebuilding the model. Set them once in train_production_v2.py.
    0: {"lr": OPTUNA_PHASE1_LR,          "ent_coef": OPTUNA_PHASE1_ENT_COEF,          "clip": OPTUNA_PHASE1_CLIP_RANGE},
    1: {"lr": OPTUNA_PHASE1_LR * 0.85,   "ent_coef": OPTUNA_PHASE1_ENT_COEF * 0.80,  "clip": OPTUNA_PHASE1_CLIP_RANGE},
    2: {"lr": TRANSFER_LR,               "ent_coef": TRANSFER_ENT_COEF,               "clip": TRANSFER_CLIP_RANGE},
    3: {"lr": TRANSFER_LR * 0.75,        "ent_coef": TRANSFER_ENT_COEF * 0.70,        "clip": TRANSFER_CLIP_RANGE},
}
