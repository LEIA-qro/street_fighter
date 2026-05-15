# ===========================================================
#                   Optuna Config
# ===========================================================   

N_HYPERPARAMETER_TRIALS = 50 

# --- HYPERPARAMETERS FROM OPTUNA TRIAL ---
LR = 1e-4
BUFFER_SIZE = 100000 
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
