# train_league.py
import os
import sys
import time
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from core import config
from core.selective_norm import SelectiveVecNormalize
from core.env_tools import failsafe_env
from core.rl_constants import AGENT_GAMMA
from agents.league.pool_manager import LeaguePoolManager

def make_league_env(rank, env_version="v2", matchup_mode="ryu_vs_ryu", custom_state=None):
    from stable_baselines3.common.monitor import Monitor
    from envs.league_env import StreetFighterLeagueEnv
    
    def _init():
        if rank > 0:
            delay = rank * 3.5
            print(f"[Rank {rank}] Staggering boot: waiting {delay:.1f}s...")
            time.sleep(delay)
            
        env = StreetFighterLeagueEnv(
            rank=rank,
            version=env_version,
            verbose=(rank == 0),
            trainable=True
        )
        
        # Determine training states based on matchup_mode
        if matchup_mode == "ryu_vs_all":
            valid_states = [s for s in config.RYU_PVP_STATES if os.path.exists(os.path.join(config.STATES_DIR, s))]
            if not valid_states:
                valid_states = list(config.RYU_ONLY_STATES) # fallback
            env.set_training_states(valid_states)
            if rank == 0:
                print(f"[League] Initializing with Ryu vs All characters PvP pool ({len(valid_states)} states).")
        elif matchup_mode == "custom" and custom_state and custom_state != "None":
            env.set_training_states([custom_state])
            if rank == 0:
                print(f"[League] Initializing with custom state: {custom_state}")
        else: # ryu_vs_ryu or fallback
            ryu_ryu_states = ["RYU_RYU_R1_PvP.State", "RYU_RYU_R1_HARD.State"]
            valid_states = [s for s in ryu_ryu_states if os.path.exists(os.path.join(config.STATES_DIR, s))]
            if not valid_states:
                valid_states = ["RYU_RYU_R1_PvP.State"]
            env.set_training_states(valid_states)
            if rank == 0:
                print(f"[League] Initializing with Ryu vs Ryu PvP pool ({len(valid_states)} states: {valid_states}).")
            
        log_dir = os.path.join(config.LOG_DIR, f"monitor_league_rank_{rank}")
        os.makedirs(log_dir, exist_ok=True)
        return Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))
    return _init

class LeagueMatchmakingCallback(CallbackList):
    """Callback to handle rolling win rate recording, adaptive matchmaking swaps, and automated checkpoint pool additions."""
    
    def __init__(self, pool_manager: LeaguePoolManager, current_model_path: str, model_name: str = "league",
                 env_version: str = "v2", save_freq: int = 250000, verbose: int = 1,
                 algo: str = "ppo", matchup_mode: str = "ryu_vs_ryu"):
        self.pool_manager = pool_manager
        self.current_model_path = current_model_path
        self.model_name = model_name
        self.env_version = env_version.lower()
        self.save_freq = save_freq
        self.verbose = verbose
        self.last_save_step = 0
        self.algo = algo.lower()
        self.matchup_mode = matchup_mode.lower()
        
        # Initialize internal callbacks list to satisfy parent class
        super().__init__([])
        
    def _on_step(self) -> bool:
        # 1. Process episode outcomes and record in the pool manager
        for info in self.locals.get("infos", []):
            if "win" in info and "opponent_id" in info:
                win = info["win"]
                opp_id = info["opponent_id"]
                self.pool_manager.record_outcome(opp_id, win)
                
                if self.verbose:
                    wr = self.pool_manager.get_win_rate(opp_id)
                    print(f"[League] Finished match against {opp_id} | Result: {'WIN' if win == 1 else 'LOSS'} | Rolling WR: {wr:.1%}")
 
        # 2. Check for episode completions (dones) and trigger process-safe opponent swaps
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                opp_id, zip_path, pkl_path = self.pool_manager.get_matchup_opponent(self.current_model_path)
                # Call set_opponent_paths locally in the process to avoid serialization lag
                self.training_env.env_method(
                    "set_opponent_paths",
                    opp_id,
                    zip_path,
                    pkl_path,
                    self.env_version,
                    indices=i
                )
                
        # 3. Handle periodic automated checkpoints
        if self.num_timesteps - self.last_save_step >= self.save_freq:
            self.last_save_step = self.num_timesteps
            self._save_milestone_checkpoint()
            
        return True
        
    def _save_milestone_checkpoint(self):
        """Saves a new Main Agent milestone model directly to the pool, automatically expanding the league opponent pool."""
        # Compute overall rolling win rate
        import numpy as np
        all_wins = []
        for buffer in self.pool_manager.win_buffers.values():
            all_wins.extend(buffer)
        
        winrate_pct = int(round(np.mean(all_wins) * 100)) if all_wins else 50
        
        # Build base name: {algo}_{env}_{customName}_{matchup_mode}_WR{winRate}pct_ckpt_{steps}
        base_name = f"{self.algo}_{self.env_version}_{self.model_name}_{self.matchup_mode}_WR{winrate_pct}pct_ckpt_{self.num_timesteps}steps"
        
        # Target paths in the league directory
        model_zip = os.path.join(self.pool_manager.league_dir, f"{base_name}.zip")
        vecnorm_pkl = os.path.join(self.pool_manager.league_dir, f"{base_name}_vecnorm.pkl")
        
        # Save active model weights
        self.model.save(model_zip)
        
        # Save active vec normalize wrapper
        if hasattr(self.training_env, "save"):
            self.training_env.save(vecnorm_pkl)
        else:
            # Unwrap if nested
            env_root = self.training_env
            while hasattr(env_root, "venv"):
                env_root = env_root.venv
            if hasattr(env_root, "save"):
                env_root.save(vecnorm_pkl)
                
        print(f"\n{'='*60}")
        print(f"[LeagueCheckpoint] Automated milestone checkpoint saved to League Pool!")
        print(f"  Model  : {os.path.basename(model_zip)}")
        print(f"  Steps  : {self.num_timesteps:,}")
        print(f"  Note   : Automatically added to matchmaking queue.")
        print(f"{'='*60}\n")

def train_league():
    parser = argparse.ArgumentParser(description="Street Fighter II RL Self-Play League Training")
    parser.add_argument("--steps", type=int, default=5000000)
    parser.add_argument("--env_version", type=str, default="v2", choices=["v2", "v3"])
    parser.add_argument("--matchup_mode", type=str, default="ryu_vs_ryu", choices=["ryu_vs_ryu", "ryu_vs_all", "custom"])
    parser.add_argument("--custom_state", type=str, default=None, help="Custom fight state filename (e.g. RYU_RYU_R1_HARD.State)")
    parser.add_argument("--model_name", type=str, default="league", help="Custom name for the active model")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing active model")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    
    # Enforce thread restriction to prevent core thrashing
    import torch
    torch.set_num_threads(1)
    
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    print(f"\n{('='*60)}")
    print(f"      Street Fighter II' — Auto-Learning League System")
    print(f"{('='*60)}")
    print(f"  Mode        : Self-Play League Training")
    print(f"  Version     : {args.env_version.upper()}")
    print(f"  Device      : {device.upper()}")
    print(f"  Matchup Mode: {args.matchup_mode.upper()}")
    print(f"  Model Name  : {args.model_name}")
    print(f"  Custom State: {args.custom_state if args.custom_state else 'None'}")
    print(f"{('='*60)}\n")

    # Define model naming paths
    directories = config.get_directory()
    # Support environment subfolders (v1, v2, v3) inside the production directory
    league_dir = os.path.join(directories["production"], args.env_version, "league")
    os.makedirs(league_dir, exist_ok=True)
    
    active_model_path = os.path.join(league_dir, f"PPO_{args.model_name}_active.zip")
    active_vecnorm_path = os.path.join(league_dir, f"PPO_{args.model_name}_active_vecnormalize.pkl")
    
    pool_manager = LeaguePoolManager(league_dir=league_dir)
    
    # ── 1. Spawn parallel League Environments ──
    n_envs = config.N_ENVS
    env_fns = [make_league_env(i, args.env_version, args.matchup_mode, args.custom_state) for i in range(n_envs)]
    
    # Using SubprocVecEnv because our path-based loader is 100% process-safe
    env = SubprocVecEnv(env_fns)
    
    try:
        # Wrap with SelectiveVecNormalize
        if args.resume and os.path.exists(active_vecnorm_path):
            env = SelectiveVecNormalize.load(active_vecnorm_path, env)
            env.training = True
            print(f"[League] Loaded active VecNormalize from: {active_vecnorm_path}")
        else:
            env = SelectiveVecNormalize(
                env, 
                n_continuous_dims=config.OBS_DIM, 
                n_frames=config.NUM_FRAMES
            )
            
        # ── 2. Pre-load first opponent for all env instances ──
        for i in range(n_envs):
            opp_id, zip_path, pkl_path = pool_manager.get_matchup_opponent(active_model_path)
            env.env_method(
                "set_opponent_paths", 
                opp_id, 
                zip_path, 
                pkl_path, 
                args.env_version, 
                indices=i
            )
            
        # ── 3. Initialize Neural Network ──
        if args.resume and os.path.exists(active_model_path):
            print(f"[League] Resuming active Main Agent training from: {active_model_path}")
            model = PPO.load(
                active_model_path,
                env=env,
                device=device,
                tensorboard_log=os.path.join(directories["logs"], "league")
            )
        else:
            print("[League] Creating fresh active Main Agent PPO network...")
            # PPO config from our tuned hyperparameters
            model = PPO(
                policy="MlpPolicy",
                env=env,
                learning_rate=2e-5,
                n_steps=2048,
                batch_size=1024,
                ent_coef=0.015,
                clip_range=0.20,
                n_epochs=10,
                gamma=AGENT_GAMMA,
                target_kl=0.03,
                policy_kwargs=dict(net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256])),
                verbose=1,
                tensorboard_log=os.path.join(directories["logs"], "league"),
                device=device
            )
            
        callback = LeagueMatchmakingCallback(
            pool_manager=pool_manager,
            current_model_path=active_model_path,
            model_name=args.model_name,
            env_version=args.env_version,
            save_freq=500000, # Checkpoint and expand opponent pool every 500k steps
            verbose=1,
            algo="ppo",
            matchup_mode=args.matchup_mode
        )
        
        print("\nStarting League Auto-Learning loop. Press Ctrl + C to stop and save.")
        model.learn(
            total_timesteps=args.steps,
            callback=callback,
            tb_log_name=f"PPO_{args.model_name}_active",
            reset_num_timesteps=False
        )
        
        # Save final models (fixed active models for resumption)
        model.save(active_model_path)
        env.save(active_vecnorm_path)
        
        # Save dynamic final checkpoint
        import numpy as np
        all_wins = []
        for buffer in pool_manager.win_buffers.values():
            all_wins.extend(buffer)
        winrate_pct = int(round(np.mean(all_wins) * 100)) if all_wins else 50
        final_base = f"ppo_{args.env_version.lower()}_{args.model_name}_{args.matchup_mode.lower()}_final_WR{winrate_pct}pct_{callback.num_timesteps}steps"
        
        model.save(os.path.join(league_dir, final_base))
        env.save(os.path.join(league_dir, f"{final_base}_vecnorm.pkl"))
        
        print(f"[League] Finished! Saved final active Main Agent to: {active_model_path}")
        print(f"[League] Saved dynamic final checkpoint as: {final_base}")
        
    except KeyboardInterrupt:
        print("\n[MANUAL OVERRIDE] League training forcefully interrupted. Saving active models...")
        if 'model' in locals():
            model.save(active_model_path)
        if 'env' in locals() and hasattr(env, "save"):
            env.save(active_vecnorm_path)
            
    finally:
        # Failsafe cleanup to prevent zombie processes and release CUDA memory
        failsafe_env(env=env if 'env' in locals() else None, model=model if 'model' in locals() else None)

if __name__ == "__main__":
    train_league()
