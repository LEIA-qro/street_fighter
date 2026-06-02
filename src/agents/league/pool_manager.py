# pool_manager.py
import os
import random
import glob
from collections import deque
from typing import Dict, List, Tuple, Optional
import numpy as np
from stable_baselines3 import PPO

from core import config

class LeaguePoolManager:
    """Manages the historical pool of checkpoints and specialized exploiters.
    
    Tracks rolling win rates against each opponent type and dynamically adjusts 
    matchmaking probabilities to focus training on the Main Agent's active weaknesses.
    """
    
    def __init__(self, league_dir: str = None, max_past_checkpoints: int = 10):
        if league_dir is None:
            self.league_dir = os.path.join(config.PROJECT_ROOT, "models", "production", "league")
        else:
            self.league_dir = league_dir
            
        self.max_past_checkpoints = max_past_checkpoints
        os.makedirs(self.league_dir, exist_ok=True)
        
        # Win Rate Tracking per opponent type/identifier
        # Keys: 'current_self', 'past_self_{filename}', 'exploiter_rusher', 'exploiter_spammer', 'exploiter_turtle'
        self.win_buffers: Dict[str, deque] = {}
        self.win_rate_window = 100
        
        self.win_rates_json = os.path.join(self.league_dir, "win_rates.json")
        self._load_win_rates()
        
        # Caching dictionary to keep policies pre-loaded in memory/VRAM to eliminate disk I/O lag
        # Structure: { opponent_id: {"model": PPO, "vecnorm_path": str} }
        self.model_cache: Dict[str, Dict] = {}
        
        # Default baseline matchmaking weights
        self.base_weights = {
            "current_self": 0.50,
            "past_self": 0.35,
            "exploiters": 0.15
        }

    def _load_win_rates(self):
        import json
        if os.path.exists(self.win_rates_json):
            try:
                with open(self.win_rates_json, "r") as f:
                    data = json.load(f)
                for opp_id, outcomes in data.items():
                    self.win_buffers[opp_id] = deque(outcomes, maxlen=self.win_rate_window)
            except Exception as e:
                print(f"[LeaguePool] Error loading win rates: {e}")

    def _save_win_rates(self):
        import json
        try:
            data = {opp_id: list(buffer) for opp_id, buffer in self.win_buffers.items()}
            with open(self.win_rates_json, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[LeaguePool] Error saving win rates: {e}")
        
    def register_opponent(self, opponent_id: str):
        """Registers a new opponent in the win rate tracker if not already present."""
        if opponent_id not in self.win_buffers:
            self.win_buffers[opponent_id] = deque(maxlen=self.win_rate_window)
            print(f"[LeaguePool] Registered opponent ID: {opponent_id}")
            
    def record_outcome(self, opponent_id: str, win: int):
        """Records the win/loss outcome of an episode against a specific opponent."""
        self.register_opponent(opponent_id)
        self.win_buffers[opponent_id].append(win)
        self._save_win_rates()
        
    def get_win_rate(self, opponent_id: str) -> float:
        """Returns the rolling win rate against a specific opponent, or 1.0 if untested."""
        buffer = self.win_buffers.get(opponent_id)
        if not buffer or len(buffer) == 0:
            return 1.0 # Treat as fully master/untested to avoid early bias
        return float(np.mean(buffer))
        
    def scan_pool(self) -> Tuple[List[str], List[str]]:
        """Scans the league directory and lists all past self checkpoints and active exploiters.
        
        Returns:
            Tuple containing (past_self_checkpoints_paths, exploiters_paths)
        """
        # Find all model zip files
        zips = glob.glob(os.path.join(self.league_dir, "*.zip"))
        
        past_checkpoints = []
        exploiters = []
        
        for p in zips:
            basename = os.path.basename(p)
            if "exploiter" in basename.lower():
                exploiters.append(p)
            elif "ckpt" in basename.lower() or "phase" in basename.lower():
                past_checkpoints.append(p)
                
        # Handle Pruning: If past checkpoints exceed limit, sort by step number or creation time and remove oldest
        if len(past_checkpoints) > self.max_past_checkpoints:
            # Sort by creation time (oldest first)
            past_checkpoints.sort(key=os.path.getmtime)
            to_remove = past_checkpoints[:-self.max_past_checkpoints]
            past_checkpoints = past_checkpoints[-self.max_past_checkpoints:]
            
            for p in to_remove:
                try:
                    os.remove(p)
                    # Also remove corresponding vecnormalize pkl if it exists
                    pkl_path = p.replace(".zip", "_vecnormalize.pkl")
                    if os.path.exists(pkl_path):
                        os.remove(pkl_path)
                    print(f"[LeaguePool][Pruner] Deleted older redundant checkpoint: {os.path.basename(p)}")
                except Exception as e:
                    print(f"[LeaguePool][Pruner] Error pruning old checkpoint: {e}")
                    
        return past_checkpoints, exploiters

    def get_matchup_opponent(self, current_model_path: Optional[str] = None) -> Tuple[str, str, str]:
        """Dynamically selects an opponent path and returns its ID, zip path, and vecnorm path.
        
        Matchmaking adapts dynamically based on rolling win rates to patch active weaknesses.
        """
        past_checkpoints, exploiters = self.scan_pool()
        
        # Build list of active candidates
        candidates = []
        
        # 1. Current self candidate
        if current_model_path and os.path.exists(current_model_path):
            candidates.append(("current_self", current_model_path, "self"))
            
        # 2. Historical self checkpoints
        for cp in past_checkpoints:
            cp_id = f"past_self_{os.path.basename(cp)}"
            candidates.append((cp_id, cp, "past_self"))
            
        # 3. Active specialized exploiters
        for exp in exploiters:
            exp_id = f"exploiter_{os.path.basename(exp)}"
            candidates.append((exp_id, exp, "exploiters"))
            
        if not candidates:
            # Fallback to standard config files if directory is empty
            fallback_zip = config.TRAINING_ZIP_FILE
            fallback_pkl = config.TRAINING_PKL_FILE
            return "fallback_opponent", fallback_zip, fallback_pkl

        # --- ADAPTIVE MATCHMAKING PROBABILITY CALCULATION ---
        # Group candidates by category to calculate base distributions
        grouped: Dict[str, List[Tuple[str, str, str]]] = {"self": [], "past_self": [], "exploiters": []}
        for entry in candidates:
            grouped[entry[2]].append(entry)
            
        # Distribute weights among available groups
        active_weights = {}
        total_weight = 0.0
        for cat, baseline in self.base_weights.items():
            # If the category has candidates, allocate baseline weight
            if grouped[cat]:
                active_weights[cat] = baseline
                total_weight += baseline
                
        # Normalize weights if some groups are missing
        if total_weight > 0:
            for cat in active_weights:
                active_weights[cat] /= total_weight
                
        # Adjust weight dynamically based on rolling win rates (weakness patching)
        # If the rolling win rate against a category's candidates falls below 60%,
        # we siphon probability from high-winrate categories and inject it into the struggling category.
        for cat, entries in grouped.items():
            if not entries or cat not in active_weights:
                continue
                
            # Average win rate across all candidates in this category
            win_rates = [self.get_win_rate(item[0]) for item in entries]
            avg_win_rate = np.mean(win_rates) if win_rates else 1.0
            
            # If win rate is low (< 60%), increase category probability up to double its baseline
            if avg_win_rate < 0.60:
                boost = (0.60 - avg_win_rate) * 0.50 # Max siphoned boost
                active_weights[cat] = min(active_weights[cat] + boost, active_weights[cat] * 2.0)
                
        # Normalize weights again to sum to 1.0
        total_normalized = sum(active_weights.values())
        for cat in active_weights:
            active_weights[cat] /= total_normalized
            
        # Select category
        chosen_cat = random.choices(
            list(active_weights.keys()), 
            weights=list(active_weights.values()), 
            k=1
        )[0]
        
        # Within the chosen category, select the specific model
        # To focus on specific weak matchups, we use softmax over (1.0 - win_rate)
        chosen_entries = grouped[chosen_cat]
        if len(chosen_entries) == 1:
            chosen = chosen_entries[0]
        else:
            # Softmax selection to favor models with lower win rates
            scores = [max(0.01, 1.0 - self.get_win_rate(item[0])) for item in chosen_entries]
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores)
            chosen = chosen_entries[random.choices(range(len(chosen_entries)), weights=probs, k=1)[0]]
            
        opponent_id, zip_path, _ = chosen
        
        # VectorNormalize path is tightly coupled to model path (standard naming)
        vecnorm_path = zip_path.replace(".zip", "_vecnormalize.pkl")
        if not os.path.exists(vecnorm_path):
            # Try parsing with alternative standard format
            vecnorm_path = zip_path.replace(".zip", "_vecnorm.pkl")
            if not os.path.exists(vecnorm_path):
                # Fallback to global production PKL
                vecnorm_path = config.TRAINING_PKL_FILE
                
        return opponent_id, zip_path, vecnorm_path
        
    def load_cached_opponent_policy(self, opponent_id: str, zip_path: str, device: str = "cpu") -> PPO:
        """Loads and caches the opponent policy to avoid high disk loading latency during environment resets.
        
        Memory Optimization: Opponents are loaded on 'cpu' to save VRAM, leaving P1 (CUDA) unconstrained.
        """
        if opponent_id in self.model_cache:
            return self.model_cache[opponent_id]["model"]
            
        print(f"[LeaguePool] Loading new opponent model into memory cache: {opponent_id}")
        
        # Load SB3 policy safely without loading replay buffer parameters (minimal footprint)
        model = PPO.load(
            zip_path,
            device=device,
            custom_objects={"buffer_size": 1}
        )
        
        # Freeze policy network parameters to disable autograd tracking and maximize performance
        model.policy.eval()
        for param in model.policy.parameters():
            param.requires_grad = False
            
        self.model_cache[opponent_id] = {
            "model": model,
            "zip_path": zip_path
        }
        
        # Keep cache capped at max_past_checkpoints + 3 (exploiters) to prevent RAM bloating
        if len(self.model_cache) > self.max_past_checkpoints + 4:
            # Remove oldest cached model that is not 'current_self'
            oldest_key = None
            for key in self.model_cache:
                if key != "current_self":
                    oldest_key = key
                    break
            if oldest_key:
                del self.model_cache[oldest_key]
                print(f"[LeaguePool] Evicted oldest cached model from RAM: {oldest_key}")
                
        return model
