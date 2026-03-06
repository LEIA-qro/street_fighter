import os

# Base Directories
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

ROMS_DIR = os.path.join(PROJECT_ROOT, "roms")
STATES_DIR = os.path.join(PROJECT_ROOT, "states")
LUA_DIR = os.path.join(PROJECT_ROOT, "lua")

# Executables & Files
BIZHAWK_PATH = r"C:\\Users\Diego Perea\Documents\\Apps\BizHawk-2.8-win-x64\\EmuHawk.exe"
ROM_PATH = os.path.join(ROMS_DIR, "Street Fighter II' - Special Champion Edition (USA).md")
TRAINING_ENV_CLIENT_LUA_PATH = os.path.join(LUA_DIR, "training_env_client.lua")
MATCH_TEST_ENV_CLIENT_LUA_PATH = os.path.join(LUA_DIR, "match_test_env_client.lua")

# Reset Config Lua Script Path (if needed in the future)
RESET_CONFIG_LUA_SCRIPT_PATH = os.path.join(LUA_DIR, "reset_config.lua")

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
MODEL_PRODUCTION_DIR = os.path.join(PROJECT_ROOT, "models", "production")
OPTUNA_DIR = os.path.join(PROJECT_ROOT, "models", "optuna_best")
MODEL_DIR_USING = os.path.join(PROJECT_ROOT, "models", "using_model")


def get_directory():
    directories = {
        "src": SRC_DIR,
        "project_root": PROJECT_ROOT,
        "roms": ROMS_DIR,
        "states": STATES_DIR,
        "lua": LUA_DIR,
        "logs": LOG_DIR,
        "production": MODEL_PRODUCTION_DIR,
        "optuna": OPTUNA_DIR,
        "using_model": MODEL_DIR_USING
    }
    for name, path in directories.items():
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created missing directory: {name} with path {path}")
    return directories

# Network
HOST = '127.0.0.1'
PORT = 9999

# Model & Training Config
MODEL_NAME = "PPO_sf2_ryu_specialist_7_2"
TRAINING_ZIP_FILE = "models/production/PPO_sf2_ryu_specialist_7_1_CRASH_SAVE.zip"
TRAINING_PKL_FILE = "models/production/PPO_sf2_ryu_specialist_7_1_vecnormalize_CRASH_SAVE.pkl"

ACTION_DIM = 10
NUM_FRAMES = 4
OBS_DIM = 10 

N_ENVS = 10 # Number of parallel BizHawk instances for Optuna trials
N_HYPERPARAMETER_TRIALS = 50 # Number of Optuna Trials to run during hyperparameter optimization
STARTING_TOTAL_TIMESTEPS = 3000000
RESUME_PRODUCTION_TIMESTEPS = 10_000_000 
SAVE_FREQ_STEPS = 100_000

# --- HYPERPARAMETERS FROM OPTUNA TRIAL ---
LR = 5.6948095644433695e-05
ENT_COEF = 0.03530287430683962
CLIP_RANGE = 0.19964140088107324
N_STEPS = 4096
BATCH_SIZE = 1024

# ---- TESTING CONFIG ----
TESTING_ZIP_FILE_P1 = "models/production/PPO_sf2_ryu_specialist_1_3_CRASH_SAVE.zip"
TESTING_PKL_FILE_P1 = "models/production/PPO_sf2_ryu_specialist_1_3_vecnormalize_CRASH_SAVE.pkl"

TESTING_ZIP_FILE_P2 = "models/production/PPO_sf2_ryu_specialist_3_2_CRASH_SAVE.zip"
TESTING_PKL_FILE_P2 = "models/production/PPO_sf2_ryu_specialist_3_2_vecnormalize_CRASH_SAVE.pkl"

# Available Savestates for Randomization
AVAILABLE_STATES = [
    "BALROG_GUILE_R1_HARD.State",
    "BLANKA_ZANGIEF_R1_HARD.State",
    "CHUNLI_ZANGIEF_R1_HARD.State",
    "DHALSIM_RYU_R1_HARD.State",
    "EHONDA_EHONDA_R1_HARD.State",
    "GUILE_GUILE_R1_HARD.State",
    "KEN_BLANKA_R1_HARD.State",
    "MBISON_KEN_R1_HARD.State",
    "RYU_BLANKA_R1_HARD.State",
    "RYU_CHUNLI_R1_HARD.State",
    "RYU_DHALSIM_R1_HARD.State",
    "RYU_EHONDA_R1_HARD.State",
    "RYU_GUILE_R1_HARD.State",
    "RYU_KEN_R1_HARD.State",
    "RYU_RYU_R1_HARD.State",
    "SAGAT_BLANKA_R1_HARD.State",
    "VEGA_KEN_R1_HARD.State",
    "ZANGIEF_RYU_R1_HARD.State"
]

RYU_ONLY_STATES = [
    "RYU_BLANKA_R1_HARD.State",
    "RYU_CHUNLI_R1_HARD.State",
    "RYU_DHALSIM_R1_HARD.State",
    "RYU_EHONDA_R1_HARD.State",
    "RYU_GUILE_R1_HARD.State",
    "RYU_KEN_R1_HARD.State",
    "RYU_RYU_R1_HARD.State",
    "RYU_RYU_R1_PEACEFUL.State"
]

RYU_ONLY_STATES_PHASE_1 = [
    "RYU_RYU_R1_PEACEFUL.State"
]

RYU_ONLY_STATES_PHASE_2 = [
    "RYU_CHUNLI_R1_lvl1.State",
    "RYU_DHALSIM_R1_lvl1.State",
    "RYU_BLANKA_R1_lvl2.State",
    "RYU_KEN_R1_lvl2.State",
    "RYU_BLANKA_R1_lvl3.State",
    "RYU_KEN_R1_lvl3.State"
]

RYU_ONLY_STATES_PHASE_3 = [
    "RYU_BLANKA_R1_lvl5.State",
    "RYU_CHUNLI_R1_lvl5.State",
    "RYU_DHALSIM_R1_lvl4.State",
    "RYU_KEN_R1_lvl4.State",
    "RYU_RYU_R1_lvl5.State",
    "RYU_ZANGIEF_R1_lvl4.State"
]

RYU_ONLY_STATES_PHASE_4 = [
    "RYU_DHALSIM_R1_lvl7.State",
    "RYU_EHONDA_R1_lvl6.State",
    "RYU_GUILE_R1_lvl6.State",
    "RYU_KEN_R1_lvl7.State",
    "RYU_RYU_R1_lvl6.State",
    "RYU_RYU_R1_lvl7.State"
]

RYU_ONLY_STATES_PHASE_5 = [
    "RYU_BLANKA_R1_HARD.State",
    "RYU_CHUNLI_R1_HARD.State",
    "RYU_DHALSIM_R1_HARD.State",
    "RYU_EHONDA_R1_HARD.State",
    "RYU_GUILE_R1_HARD.State",
    "RYU_KEN_R1_HARD.State",
    "RYU_RYU_R1_HARD.State"
]