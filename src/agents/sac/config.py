# ===========================================================
#                   Optuna Config
# ===========================================================   

N_HYPERPARAMETER_TRIALS = 50 # Number of Optuna Trials to run during hyperparameter optimization

'''
Change this every optuna study
'''

# --- HYPERPARAMETERS FROM OPTUNA TRIAL ---
LR = 3e-4
BUFFER_SIZE = 100000 
BATCH_SIZE = 256
TAU = 0.005
GAMMA = 0.99

NET_ARCH = dict(pi=[512, 512, 256], qf=[512, 512, 256])

# Curriculum advancement gate
WIN_RATE_THRESHOLD = 0.75   # Must win 75% of episodes to advance
WIN_RATE_WINDOW    = 250    # Rolling window of episodes to measure

OPTUNA_PHASE1_LR = LR
OPTUNA_PHASE1_TAU = TAU

TRANSFER_LR = 1e-4
TRANSFER_TAU = 0.005

# Update PHASE_HYPERPARAMS to re-index:
PHASE_HYPERPARAMS = {
    0: {"lr": OPTUNA_PHASE1_LR,          "tau": OPTUNA_PHASE1_TAU},
    1: {"lr": OPTUNA_PHASE1_LR * 0.85,   "tau": OPTUNA_PHASE1_TAU},
    2: {"lr": TRANSFER_LR,               "tau": TRANSFER_TAU},
    3: {"lr": TRANSFER_LR * 0.75,        "tau": TRANSFER_TAU},
}
