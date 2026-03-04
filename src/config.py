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

# Network
HOST = '127.0.0.1'
PORT = 9999
ACTION_DIM = 10
N_ENVS = 16 # Number of parallel BizHawk instances for Optuna trials
N_HYPERPARAMETER_TRIALS = 20 # Number of Optuna Trials to run during hyperparameter optimization

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