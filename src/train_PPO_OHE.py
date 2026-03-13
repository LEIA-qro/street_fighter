# train_PPO_OHE.py

import os
import torch
import gc
import multiprocessing
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# pip install tensorboard 
# pip install "setuptools<70.0.0"
# run tensorboard --logdir=logs\ in another terminal to visualize training metrics in real-time

# Import your custom environment and configs
import config
from env_sf2_v2 import StreetFighterEnvV2
from selective_norm import SelectiveVecNormalize



# Define directories for weights and logs
directories = config.get_directory()

class SaveOnStepCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f"{config.MODEL_NAME}_{self.num_timesteps}_steps")
            vec_path   = os.path.join(self.save_path, f"{config.MODEL_NAME}_vecnormalize_{self.num_timesteps}_steps.pkl")
            self.model.save(model_path)
            self.training_env.save(vec_path)  # Calls SelectiveVecNormalize.save()
            print(f"[Checkpoint] Saved at {self.num_timesteps} steps.")
        return True


def train_baseline():

    checkpoint_callback = SaveOnStepCallback(save_freq=config.SAVE_FREQ_STEPS, save_path=directories["production"])
    print("Initializing Street Fighter Environment...")
    
    # 1. Instantiate and Wrap the Environment
    # The Monitor wrapper records episode statistics (rewards/lengths) for TensorBoard
    raw_env = StreetFighterEnvV2()
    monitored_env = Monitor(raw_env)

    # 2. Vectorize the Environment (Required for VecNormalize)
    # Even though we only have 1 environment right now, we wrap it in a DummyVecEnv
    vec_env = DummyVecEnv([lambda: monitored_env])

    # 3. Apply the Normalization Layer
    env = SelectiveVecNormalize(vec_env, n_continuous_dims=10, n_frames=4, clip=10.0)
    
    # 4. Initialize the Neural Network (PPO)
    print("Initializing PPO Algorithm (MlpPolicy) on GPU...")
    model = PPO(
        policy="MlpPolicy",   # Multi-Layer Perceptron
        env=env,
        learning_rate=1e-4,     # SEATBELT: Lowered from 3e-4 to prevent violent unlearning
        n_steps=4096,           # SEATBELT: Collect twice as much data before updating the brain
        batch_size=256,         # SEATBELT: Smoothes out the gradients over larger batches
        n_epochs=10,          
        gamma=0.99,           
        ent_coef=0.05,          # EXPLORATION TAX: Forces the agent to keep pressing different buttons
        clip_range=0.1,         # SEATBELT: Forcibly prevents the KL Divergence from exploding
        # --- THE FIX: ADD THE MASSIVE BRAIN ---
        policy_kwargs=dict(net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256])),
        tensorboard_log=directories["logs"],
        verbose=1,
        device="cuda"         # Explicitly targets your RTX 5070 Ti
    )
    
    # 4. Execute the Training Loop
    # 100,000 steps with k=4 frame skip is ~1.8 hours of in-game time (assuming 60fps).
    # This is a good baseline to verify if the agent learns basic blocking and striking.
    print(f"Starting training loop for {config.STARTING_TOTAL_TIMESTEPS} steps...")
    
    try:
        model.learn(
            total_timesteps=config.STARTING_TOTAL_TIMESTEPS, 
            callback=checkpoint_callback,
            tb_log_name=config.MODEL_NAME
        )
        
        # 5. Save the final model state
        # Save the final model AND the normalization statistics
        final_model_path = os.path.join(directories["production"], f"{config.MODEL_NAME}_final")
        model.save(final_model_path)

        stats_path = os.path.join(directories["production"], f"{config.MODEL_NAME}_vec.pkl")
        env.save(stats_path)

        print(f"\nTraining complete! Final model saved to {final_model_path}.zip")
        
    except KeyboardInterrupt:
        # Graceful interruption: Save the model if you manually kill the script
        print("\nTraining forcefully interrupted by user. Executing emergency save...")
        emergency_path = os.path.join(directories["production"], f"{config.MODEL_NAME}_EMERGENCY_SAVE")
        model.save(emergency_path)
        env.save(os.path.join(directories["production"], f"{config.MODEL_NAME}_vec_EMERGENCY.pkl"))
        print(f"Emergency weights and stats saved to {emergency_path}.zip")
        
    finally:
        # THE FIX: Full Nuclear Cleanup
        print("Executing Nuclear Cleanup...")
        env.close()
        os.system("taskkill /F /IM EmuHawk.exe >nul 2>&1")
        active_children = multiprocessing.active_children()
        for child in active_children:
            try: child.kill()
            except: pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    train_baseline()