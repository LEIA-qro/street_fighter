import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

# pip install tensorboard 
# pip install "setuptools<70.0.0"

# Import your custom environment and configs
import config
from env_sf2 import StreetFighterEnv

# Define directories for weights and logs
LOG_DIR = os.path.join(config.PROJECT_ROOT, "logs")
MODEL_DIR = os.path.join(config.PROJECT_ROOT, "models")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def train_baseline():
    print("Initializing Street Fighter Environment...")
    
    # 1. Instantiate and Wrap the Environment
    # The Monitor wrapper records episode statistics (rewards/lengths) for TensorBoard
    raw_env = StreetFighterEnv()
    env = Monitor(raw_env, LOG_DIR)
    
    # 2. Setup Checkpoint Callback
    # Saves a .zip file of the neural network weights every 10,000 Agent Steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=MODEL_DIR,
        name_prefix="ppo_sf2_baseline"
    )
    
    # 3. Initialize the Neural Network (PPO)
    print("Initializing PPO Algorithm (MlpPolicy) on GPU...")
    model = PPO(
        policy="MlpPolicy",   # Multi-Layer Perceptron: Perfect for our 10D integer vector
        env=env,
        learning_rate=3e-4,   # Standard baseline learning rate
        n_steps=2048,         # Number of steps to collect before updating the network
        batch_size=64,        # Minibatch size for gradient descent
        n_epochs=10,          # How many times to pass over the collected data
        gamma=0.99,           # Discount factor for future rewards
        tensorboard_log=LOG_DIR,
        verbose=1,
        device="cuda"         # Explicitly targets your RTX 5070 Ti
    )
    
    # 4. Execute the Training Loop
    # 100,000 steps with k=4 frame skip is ~1.8 hours of in-game time (assuming 60fps).
    # This is a good baseline to verify if the agent learns basic blocking and striking.
    TOTAL_TIMESTEPS = 100000 
    print(f"Starting training loop for {TOTAL_TIMESTEPS} timesteps...")
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=checkpoint_callback,
            tb_log_name="PPO_Baseline_Run_1"
        )
        
        # 5. Save the final model state
        final_model_path = os.path.join(MODEL_DIR, "ppo_sf2_final_baseline")
        model.save(final_model_path)
        print(f"\nTraining complete! Final model saved to {final_model_path}.zip")
        
    except KeyboardInterrupt:
        # Graceful interruption: Save the model if you manually kill the script
        print("\nTraining forcefully interrupted by user. Executing emergency save...")
        emergency_path = os.path.join(MODEL_DIR, "ppo_sf2_EMERGENCY_SAVE")
        model.save(emergency_path)
        print(f"Emergency weights saved to {emergency_path}.zip")
        
    finally:
        # 6. Execute the Poison Pill
        env.close()

if __name__ == "__main__":
    train_baseline()