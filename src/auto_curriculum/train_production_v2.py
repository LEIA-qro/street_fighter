# edited train_productio_v2.py
import os, multiprocessing, gc, torch, time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

import config
from env_sf2_v2 import StreetFighterEnvV2
from selective_norm import SelectiveVecNormalize
from curriculum_callback import CurriculumCallback

directories = config.get_directory()
    
def make_env(rank):
    def _init():
        env = StreetFighterEnvV2(rank=rank)
        return Monitor(env)
    return _init

def train_production():
    print("[Training] Initializing Curriculum Production Training...")

    # Phase 0 starts here — config.TRAINING_STATES must equal CURRICULUM_PHASES[0]
    config.TRAINING_STATES = config.CURRICULUM_PHASES[0]

    n_envs = config.N_ENVS
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    # USE THE NEW SELECTIVE NORMALIZER
    env = SelectiveVecNormalize(env, n_continuous_dims=10, n_frames=4)

    phase = config.PHASE_HYPERPARAMS[0]
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

    callback = CurriculumCallback(save_path=directories["production"], verbose=1)
    
    print("[Training] Press Ctrl + C to stop the training. ")

    try:
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
        print("\nTraining forcefully interrupted. Executing emergency save...")
        model.save(os.path.join(directories["production"], config.MODEL_NAME + "_EMERGENCY"))
        env.save(os.path.join(directories["production"], config.MODEL_NAME + "_vecnormalize_EMERGENCY.pkl"))
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Training crashed: {e}")
        model.save(os.path.join(directories["production"], config.MODEL_NAME + "_CRASH_SAVE"))
        env.save(os.path.join(directories["production"], config.MODEL_NAME + "_vecnormalize_CRASH_SAVE.pkl"))

    finally:
        print("Executing Failsafe: Purging zombie instances and VRAM...")
        # 1. Kill Emulators
        os.system("taskkill /F /IM EmuHawk.exe >nul 2>&1")
        time.sleep(2)
        
        # 2. The Thread Sniper
        active_children = multiprocessing.active_children()
        if active_children:
            for child in active_children:
                try:
                    child.kill()
                except Exception:
                    pass
        
        # 3. The VRAM Purge
        try:
            del model
            del env
        except UnboundLocalError:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        time.sleep(3)

if __name__ == "__main__":
    train_production()