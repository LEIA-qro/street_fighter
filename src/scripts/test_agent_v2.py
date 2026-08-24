import os, sys, argparse, random, time
from typing import Any
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from stable_baselines3 import PPO, SAC, DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from core import config
from envs.sf2_v2 import StreetFighterEnvV2, TOTAL_OBS_DIM
from core.selective_norm import SelectiveVecNormalize
from core.env_tools import failsafe_env
from core.telemetry import write_telemetry, clean_telemetry

import signal
def _sigbreak_handler(sig, frame):
    raise KeyboardInterrupt

if hasattr(signal, "SIGBREAK"):
    try:
        signal.signal(signal.SIGBREAK, _sigbreak_handler)
    except Exception:
        pass

def get_model_class(algo_name):
    if algo_name.lower() == "sac":
        return SAC
    elif algo_name.lower() == "dqn":
        return DQN
    return PPO

def test_agent():
    # GPU Verification
    import torch
    # Performance Optimization: Restrict PyTorch to 1 CPU core during testing
    # This prevents it from hijacking all logical cores for inference.
    torch.set_num_threads(1)

    if torch.cuda.is_available():
        print(f"[Hardware] Testing on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[Hardware] WARNING: No GPU detected for testing. Running on CPU.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="ppo")
    parser.add_argument("--env",  type=str, default="v2", choices=["v2", "v3"])
    parser.add_argument("--load_zip", type=str, required=True)
    parser.add_argument("--load_pkl", type=str, required=True)
    parser.add_argument("--player", type=int, default=1)
    parser.add_argument("--opponent_type", type=str, choices=["human", "cpu"], default="human")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--profile", action="store_true", help="Enable performance profiling via cProfile")
    parser.add_argument("--infinite_match", action="store_true", help="Automatically reset and start rematches on KO")
    parser.add_argument("--rematch_delay", type=float, default=2.0, help="Delay in seconds before triggering auto-rematch")
    parser.add_argument("--cpu_level_cap", type=int, default=5, choices=range(1, 9), help="Maximum CPU difficulty level cap (1-8) for infinite matchups")
    args = parser.parse_args()

    # Auto-detect algorithm override based on path to prevent mismatch errors
    if args.load_zip != "None":
        lower_path = args.load_zip.lower()
        if "dqn" in lower_path and args.algo.lower() != "dqn":
            print(f"\n[WARN] Auto-detected 'dqn' in path. Overriding algorithm '{args.algo.upper()}' to 'DQN'.")
            args.algo = "dqn"
        elif "sac" in lower_path and args.algo.lower() != "sac":
            print(f"\n[WARN] Auto-detected 'sac' in path. Overriding algorithm '{args.algo.upper()}' to 'SAC'.")
            args.algo = "sac"
        elif "ppo" in lower_path and args.algo.lower() != "ppo":
            print(f"\n[WARN] Auto-detected 'ppo' in path. Overriding algorithm '{args.algo.upper()}' to 'PPO'.")
            args.algo = "ppo"

    # Set Model Labels for Lua UI
    model_name = os.path.basename(args.load_zip).replace(".zip", "")
    opp_name = "Human" if args.opponent_type == "human" else "CPU"
    
    if args.player == 1:
        config.P1_MODEL_NAME = f"AI: {model_name}"
        config.P2_MODEL_NAME = opp_name
    else:
        config.P1_MODEL_NAME = opp_name
        config.P2_MODEL_NAME = f"AI: {model_name}"

    config.generate_lua_config()

    # Device Auto-Logic
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Suppress SB3 GPU Warnings for MlpPolicy
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

    print(f"Initializing Street Fighter Evaluation Mode for Player {args.player}...")
    print(f"Compute Device: {device}")
    
    model_load_path = os.path.join(config.PROJECT_ROOT, args.load_zip)
    vec_load_path = os.path.join(config.PROJECT_ROOT, args.load_pkl)

    # 1. Boot a single emulator window
    if args.env == "v3":
        from envs.sf2_v3 import StreetFighterEnvV3
        env = StreetFighterEnvV3(
            lua_path = config.MATCH_TEST_ENV_CLIENT_LUA_PATH, 
            trainable = False, 
            rank=-1, 
            player=args.player
        )
    else:
        env = StreetFighterEnvV2(
            lua_path = config.MATCH_TEST_ENV_CLIENT_LUA_PATH , 
            trainable = False, 
            rank=-1, 
            player=args.player
        ) 
    
    env = DummyVecEnv([lambda: env])
    
    # 2. Load the Normalization Math safely
    print(f"Loading normalization stats from {args.load_pkl}...")
    env = SelectiveVecNormalize.load(vec_load_path, env)
    
    # CRITICAL: Lock the normalization math. Do NOT let it update during testing!
    env.training = False
    env.norm_reward = False 
    
    # 3. Load the Grandmaster Brain
    print(f"Loading neural network weights from {args.load_zip}...")
    
    def load_model_safely(algo: str, path: str, device: str) -> Any:
        ModelClass = get_model_class(algo)
        custom_objs = {
            "learning_rate": 0.0,
            "clip_range": 0.0,
        }
        if algo.lower() in ["dqn", "sac"]:
            custom_objs["buffer_size"] = 1  # Memory optimization
            
        try:
            model = ModelClass.load(
                path,
                device=device,
                custom_objects=custom_objs
            )
            print(f"Model ({algo.upper()}) loaded successfully.")
            return model
        except (AttributeError, TypeError, ValueError) as e:
            err_msg = str(e)
            # Detect common SB3 mismatch indicators
            is_mismatch = any(keyword in err_msg for keyword in [
                "unexpected keyword argument", 
                "object has no attribute",
                "missing 1 required positional argument"
            ])
            
            if is_mismatch:
                print(f"\n[CRITICAL ERROR] Algorithm '{algo.upper()}' is incompatible with the provided file: {path}")
                print(f"Details: {err_msg}")
            else:
                print(f"\n[ERROR] Unexpected error loading model: {err_msg}")
            sys.exit(1)
        except Exception as e:
            print(f"\n[ERROR] Unexpected error loading model: {e}")
            sys.exit(1)

    model = load_model_safely(args.algo, model_load_path, device)
    
    match_count = 1
    ai_wins = 0
    opp_wins = 0
    round_started = False
    ko_time = None
    round_winner_msg = None

    print(f"\nFIGHT! (The {args.algo.upper()} AI engine is now running in the background)")
    if args.infinite_match:
        if args.opponent_type == "human":
            initial_state_file = "RYU_RYU_R1_PvP.State"
        else:
            cpu_candidates = config.get_cpu_states_up_to_level(args.cpu_level_cap)
            available_cpu = [s for s in cpu_candidates if os.path.exists(os.path.join(config.STATES_DIR, s))]
            initial_state_file = random.choice(available_cpu) if available_cpu else "RYU_BALROG_R1_HARD.State"

        initial_state_path = os.path.join(config.STATES_DIR, initial_state_file)
        print(f"[Infinite Match] Cold-starting in active state: {initial_state_file}", flush=True)
        base_env = env.envs[0]
        p1_score = ai_wins if args.player == 1 else opp_wins
        p2_score = opp_wins if args.player == 1 else ai_wins
        base_env.send_command(f"RESET {initial_state_path}|{p1_score}|{p2_score}\n")
        reset_payload = base_env.receive_payload()
        observation = base_env._parse_payload(reset_payload, is_reset=True)
        base_env.frames.clear()
        for _ in range(config.NUM_FRAMES):
            base_env.frames.append(observation)
        obs_raw = base_env._get_obs()[np.newaxis, :]
        obs = env.normalize_obs(obs_raw, update=False)
    else:
        obs = env.reset()
    
    # Optional Profiling setup
    profiler = None
    if args.profile:
        import cProfile, pstats
        profiler = cProfile.Profile()
        profiler.enable()
        print("[Profiling] Performance tracking ENABLED. Results will display after the session ends.", flush=True)

    try:
        while True:
            # Check for graceful stop signal from Dashboard
            if os.path.exists(os.path.join(config.PROJECT_ROOT, ".stop_training")):
                print("\n[Match] Stop signal received from dashboard. Exiting gracefully...", flush=True)
                break

            action, _states = model.predict(obs, deterministic=False) 
            
            # Process action depending on algo and environment version
            if args.algo.lower() == "sac":
                if args.env == "v3":
                    seg1 = action[0][:9]
                    seg2 = action[0][9:]
                    processed_action = np.array([[np.argmax(seg1), np.argmax(seg2)]], dtype=np.int64)
                else:
                    bin_act = (action[0] > 0.0).astype(np.int8)
                    processed_action = np.array([bin_act])
            elif args.algo.lower() == "dqn":
                val = action[0] if isinstance(action, np.ndarray) else action
                if args.env == "v3":
                    # Decode flat index -> MultiDiscrete([9, 7]) via divmod
                    nvec = [9, 7]
                    decoded = []
                    remaining = int(val)
                    for n in reversed(nvec):
                        decoded.append(remaining % n)
                        remaining //= n
                    processed_action = np.array([list(reversed(decoded))], dtype=np.int64)
                else:
                    binary_str = format(val, f'0{config.ACTION_DIM}b')
                    bin_act = np.array([int(b) for b in binary_str], dtype=np.int8)
                    processed_action = np.array([bin_act])
            else:
                processed_action = action

            obs, reward, done, info = env.step(processed_action)
            
            # Stream observations & activations to disk for the Dashboard
            unnorm_obs = env.unnormalize_obs(obs)
            write_telemetry(
                model_name=model_name,
                env_version=args.env,
                status="PLAYING",
                model=model,
                obs=unnorm_obs,
                player=args.player
            )

            # Check KO for infinite matchups (exact RAM readings via step info with latest frame fallback)
            latest_idx = (config.NUM_FRAMES - 1) * TOTAL_OBS_DIM
            if isinstance(info, (list, tuple)) and len(info) > 0 and isinstance(info[0], dict) and "my_hp" in info[0]:
                ai_hp = int(info[0]["my_hp"])
                opp_hp = int(info[0]["enemy_hp"])
            else:
                ai_hp = int(round(float(unnorm_obs[0, latest_idx + 0])))
                opp_hp = int(round(float(unnorm_obs[0, latest_idx + 1])))

            if args.infinite_match:
                # Mark round as active once both fighters have positive health
                if not round_started and ai_hp > 0 and opp_hp > 0:
                    round_started = True
                    ko_time = None
                    round_winner_msg = None

                # Detect KO once round is active
                if round_started and (ai_hp <= 0 or opp_hp <= 0):
                    if ko_time is None:
                        ko_time = time.time()
                        if ai_hp > 0 and opp_hp <= 0:
                            ai_wins += 1
                            round_winner_msg = f"[WINNER] AI ({model_name}) WINS!"
                        elif opp_hp > 0 and ai_hp <= 0:
                            opp_wins += 1
                            round_winner_msg = f"[WINNER] Opponent ({opp_name}) WINS!"
                        else:
                            round_winner_msg = "[DRAW] DOUBLE K.O. (Draw)!"

                    # Check if rematch delay has elapsed before sending RESET
                    if time.time() - ko_time >= args.rematch_delay:
                        print(f"\n{'='*50}", flush=True)
                        print(f"[ROUND {match_count} OVER] {round_winner_msg}", flush=True)
                        print(f"[SCOREBOARD] AI ({model_name}): {ai_wins} | Opponent ({opp_name}): {opp_wins}", flush=True)
                        print(f"[Rematch] Loading new round...", flush=True)
                        print(f"{'='*50}\n", flush=True)

                        # Select rematch state
                        if args.opponent_type == "human":
                            rematch_state_file = "RYU_RYU_R1_PvP.State"
                        else:
                            cpu_candidates = config.get_cpu_states_up_to_level(args.cpu_level_cap)
                            available_cpu = [s for s in cpu_candidates if os.path.exists(os.path.join(config.STATES_DIR, s))]
                            rematch_state_file = random.choice(available_cpu) if available_cpu else "RYU_BALROG_R1_HARD.State"

                        print(f"[Rematch] Loading state: {rematch_state_file}", flush=True)
                        full_state_path = os.path.join(config.STATES_DIR, rematch_state_file)
                        
                        # Send RESET command directly via underlying base env
                        base_env = env.envs[0]
                        p1_score = ai_wins if args.player == 1 else opp_wins
                        p2_score = opp_wins if args.player == 1 else ai_wins
                        base_env.send_command(f"RESET {full_state_path}|{p1_score}|{p2_score}\n")
                        reset_payload = base_env.receive_payload()
                        observation = base_env._parse_payload(reset_payload, is_reset=True)
                        base_env.frames.clear()
                        for _ in range(config.NUM_FRAMES):
                            base_env.frames.append(observation)
                        obs_raw = base_env._get_obs()[np.newaxis, :]
                        obs = env.normalize_obs(obs_raw, update=False)

                        match_count += 1
                        round_started = False
                        ko_time = None
                        round_winner_msg = None
                        continue
            
    except KeyboardInterrupt:
        print("\nInteractive session ended by user.", flush=True)
    finally:
        clean_telemetry()
        if profiler:
            profiler.disable()
            print("\n" + "="*60)
            print("        PERFORMANCE PROFILE (CUMULATIVE TIME)")
            print("="*60)
            stats = pstats.Stats(profiler).sort_stats('cumulative')
            stats.print_stats(20)
        env.close()

if __name__ == "__main__":
    test_agent()