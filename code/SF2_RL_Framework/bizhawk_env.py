# libraries needed pip install gymnasium numpy

# bizhawk_env.py
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Dict, MultiBinary
import numpy as np
import socket
import time

class BizHawkSF2Env(gym.Env):
    def __init__(self):
        super(BizHawkSF2Env, self).__init__()
        
        # 12 Buttons for Genesis
        self.action_space = MultiBinary(12)
        
        # Bifurcated Observation Space for Entity Embeddings
        self.observation_space = Dict({
            'continuous': Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32), # HP, X, Y
            'p1_state': Discrete(256), # Categorical Action IDs
            'p2_state': Discrete(256)
        })
        
        # IPC Socket Setup
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(("127.0.0.1", 1337))
        self.server.listen(1)
        print("Python Server Listening on port 1337. Start the Lua script in BizHawk now.")
        
        self.conn, self.addr = self.server.accept()
        print(f"BizHawk connected from {self.addr}")
        
        self.prev_p1_hp = 176
        self.prev_p2_hp = 176

    def _normalize_and_format(self, raw_data):
        # raw_data: [p1_hp, p2_hp, p1_x, p2_x, p1_y, p2_y, p1_action_id, p2_action_id]
        cont_vars = np.array([
            raw_data[0] / 176.0,           # P1 HP
            raw_data[1] / 176.0,           # P2 HP
            (raw_data[2] - 55.0) / 402.0,  # P1 X
            (raw_data[3] - 55.0) / 402.0,  # P2 X
            raw_data[4] / 192.0,           # P1 Y
            raw_data[5] / 192.0            # P2 Y
        ], dtype=np.float32)
        
        # Clip continuous variables strictly to [0.0, 1.0] to prevent embedding drift
        cont_vars = np.clip(cont_vars, 0.0, 1.0)

        return {
            'continuous': cont_vars,
            'p1_state': int(raw_data[6]),
            'p2_state': int(raw_data[7])
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Trigger Lua Savestate Load
        self.conn.sendall(b"RESET\n")
        
        # 2. Add a tiny sleep to allow BizHawk to load the state and send the first frame
        time.sleep(0.05) 
        
        # 3. Read the first frame of the new state
        data = self.conn.recv(1024).decode('utf-8').strip()
        raw_state = [int(x) for x in data.split(',')]
        
        # Reset reward trackers
        self.prev_p1_hp = raw_state[0]
        self.prev_p2_hp = raw_state[1]
        
        return self._normalize_and_format(raw_state), {}

    def step(self, action):
        # 1. Convert MultiBinary array [1, 0, 0, 1...] to string "1001..."
        action_str = "".join(str(int(b)) for b in action) + "\n"
        self.conn.sendall(action_str.encode('utf-8'))
        
        # 2. Wait for next frame from Lua
        data = self.conn.recv(1024).decode('utf-8').strip()
        raw_state = [int(x) for x in data.split(',')]
        
        # 3. Parse state
        obs = self._normalize_and_format(raw_state)
        p1_hp, p2_hp = raw_state[0], raw_state[1]
        
        # 4. Calculate Asymmetric Reward
        delta_p2 = self.prev_p2_hp - p2_hp # Damage dealt
        delta_p1 = self.prev_p1_hp - p1_hp # Damage taken
        
        # Reward shaping: Highly reward damage dealt, lightly penalize damage taken
        reward = (delta_p2 * 2.0) - (delta_p1 * 1.0)
        
        self.prev_p1_hp = p1_hp
        self.prev_p2_hp = p2_hp
        
        # 5. Check Termination (Round over when someone hits 0 HP)
        terminated = bool(p1_hp <= 0 or p2_hp <= 0)
        truncated = False # You can add a 99-second frame limit here later
        
        return obs, reward, terminated, truncated, {}