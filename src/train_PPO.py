import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

# pip install tensorboard 
# pip install "setuptools<70.0.0"
# run tensorboard --logdir=logs\ in another terminal to visualize training metrics in real-time

# Import your custom environment and configs
import config
from env_sf2 import StreetFighterEnv

# Define directories for weights and logs
directories = config.get_directory()

def train_baseline():
    print("Initializing Street Fighter Environment...")
    
    # 1. Instantiate and Wrap the Environment
    # The Monitor wrapper records episode statistics (rewards/lengths) for TensorBoard
    raw_env = StreetFighterEnv()
    monitored_env = Monitor(raw_env, directories["logs"])

    # 2. Vectorize the Environment (Required for VecNormalize)
    # Even though we only have 1 environment right now, we wrap it in a DummyVecEnv
    vec_env = DummyVecEnv([lambda: monitored_env])

    # 3. Apply the Normalization Layer
    # This dynamically scales all observations to a mean of 0 and variance of 1.
    # It also normalizes the reward signal, which stabilizes the Critic network.
    env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,   # Normalizing rewards accelerates PPO convergence
        clip_obs=10.0       # Clamps extreme outliers
    )
    
    # 2. Setup Checkpoint Callback
    # Saves a .zip file of the neural network weights every 10,000 Agent Steps
    checkpoint_callback = CheckpointCallback(
        save_freq=config.SAVE_FREQ_STEPS,
        save_path=directories["MODEL_DIR"],
        name_prefix=config.MODEL_NAME
    )
    
    # 3. Initialize the Neural Network (PPO)
    print("Initializing PPO Algorithm (MlpPolicy) on GPU...")
    model = PPO(
        policy="MlpPolicy",   # Multi-Layer Perceptron: Perfect for our 10D integer vector
        env=env,
        learning_rate=1e-4,     # SEATBELT: Lowered from 3e-4 to prevent violent unlearning
        n_steps=4096,           # SEATBELT: Collect twice as much data before updating the brain
        batch_size=256,         # SEATBELT: Smoothes out the gradients over larger batches
        n_epochs=10,          
        gamma=0.99,           
        ent_coef=0.05,          # EXPLORATION TAX: Forces the agent to keep pressing different buttons
        clip_range=0.1,         # SEATBELT: Forcibly prevents the KL Divergence from explodingtensorboard_log=LOG_DIR,
        tensorboard_log=directories["logs"],
        verbose=1,
        device="cuda"         # Explicitly targets your RTX 5070 Ti
    )
    
    # 4. Execute the Training Loop
    # 100,000 steps with k=4 frame skip is ~1.8 hours of in-game time (assuming 60fps).
    # This is a good baseline to verify if the agent learns basic blocking and striking.
    TOTAL_TIMESTEPS = config.STARTING_TOTAL_TIMESTEPS 
    print(f"Starting training loop for {TOTAL_TIMESTEPS} timesteps...")
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=checkpoint_callback,
            tb_log_name=config.MODEL_NAME
        )
        
        # 5. Save the final model state
        # Save the final model AND the normalization statistics
        final_model_path = os.path.join(directories["MODEL_DIR"], f"{config.MODEL_NAME}_final")
        model.save(final_model_path)

        stats_path = os.path.join(directories["MODEL_DIR"], f"{config.MODEL_NAME}_vec.pkl")
        env.save(stats_path)

        print(f"\nTraining complete! Final model saved to {final_model_path}.zip")
        
    except KeyboardInterrupt:
        # Graceful interruption: Save the model if you manually kill the script
        print("\nTraining forcefully interrupted by user. Executing emergency save...")
        emergency_path = os.path.join(directories["MODEL_DIR"], f"{config.MODEL_NAME}_EMERGENCY_SAVE")
        model.save(emergency_path)
        env.save(os.path.join(directories["MODEL_DIR"], f"{config.MODEL_NAME}_vec_EMERGENCY.pkl"))
        print(f"Emergency weights and stats saved to {emergency_path}.zip")
        
    finally:
        # 6. Execute the Poison Pill
        env.close()

if __name__ == "__main__":
    train_baseline()