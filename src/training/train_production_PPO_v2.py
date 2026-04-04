# edited train_productio_v2.py
import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.callbacks import BaseCallback

import config
# from env_sf2_v2 import StreetFighterEnvV2
from selective_norm import SelectiveVecNormalize
from env_tools import SFv2_make_env, failsafe_env
from manual_curriculum_callback import ManualCurriculumCallback

directories = config.get_directory()


def train_production_PPO():
    print("[Training] Initializing Curriculum Production Training...")

    # Phase 0 starts here — config.TRAINING_STATES must equal CURRICULUM_PHASES[0]
    config.TRAINING_STATES = config.CURRICULUM_PHASES[0]

    n_envs = config.N_ENVS
    phase = config.PHASE_HYPERPARAMS[0]
    env = None
    model = None
    

    try:
        env = SubprocVecEnv([SFv2_make_env(i) for i in range(n_envs)])
        # USE THE NEW SELECTIVE NORMALIZER
        env = SelectiveVecNormalize(env,
                                     n_continuous_dims=config.OBS_DIM, 
                                     n_frames=config.NUM_FRAMES)

        model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=phase["lr"],
        n_steps=config.N_STEPS,
        batch_size=config.BATCH_SIZE,
        ent_coef=phase["ent_coef"],
        clip_range=phase["clip"],
        n_epochs=10,
        gamma=0.99,
        target_kl=0.03,
        policy_kwargs=dict(net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256])),
        verbose=1,
        tensorboard_log=directories["logs"],
        device="cuda"
        )

        callback = ManualCurriculumCallback(
        save_path=directories["production"],
        start_phase=0,
        eval_interval=500,
        save_interval=config.SAVE_FREQ_STEPS
        )
        
        print("[Training] Press Ctrl + C to stop the training. ")

        model.learn(
            total_timesteps=config.STARTING_TOTAL_TIMESTEPS, 
            callback=callback,
            tb_log_name=config.MODEL_NAME
        )
        
        # Save Final Production Model
        model.save(os.path.join(directories["production"], config.MODEL_NAME + "_FINAL"))
        env.save(os.path.join(directories["production"], config.MODEL_NAME + "_vecnormalize_FINAL.pkl"))
        print("\nProduction Training Complete!")
        
    except KeyboardInterrupt:
        print("\n[MANUAL OVERRIDE] Training forcefully interrupted by user.")
        if model is not None: model.save(os.path.join(directories["production"], config.MODEL_NAME + "_EMERGENCY"))
        if env is not None: env.save(os.path.join(directories["production"], config.MODEL_NAME + "_vecnormalize_EMERGENCY.pkl"))

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Training crashed: {e}")
        if model is not None: model.save(os.path.join(directories["production"], config.MODEL_NAME + "_CRASH_SAVE"))
        if env is not None: env.save(os.path.join(directories["production"], config.MODEL_NAME + "_vecnormalize_CRASH_SAVE.pkl"))

    finally:
        failsafe_env(
            env=env if 'env' in dir() else None,
            model=model if 'model' in dir() else None
        )

if __name__ == "__main__":
    train_production_PPO()