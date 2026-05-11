import os

content = """import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from core import config
from core.selective_norm import SelectiveVecNormalize
from core.env_tools import failsafe_env
from src.manual_curriculum_callback import ManualCurriculumCallback
from agents.base_agent import BaseAgent
from agents.ppo.config import PHASE_HYPERPARAMS, N_STEPS, BATCH_SIZE

class PPOAgent(BaseAgent):
    def train(self, env_fn, save_dir, steps):
        print(f"[Training] Initializing Curriculum Production Training in {save_dir}...")

        config.TRAINING_STATES = config.CURRICULUM_PHASES[0]
        n_envs = config.N_ENVS
        phase = PHASE_HYPERPARAMS[0]
        env = None
        model = None
        directories = config.get_directory()
        
        try:
            env = SubprocVecEnv([env_fn(i) for i in range(n_envs)])
            env = SelectiveVecNormalize(env,
                                         n_continuous_dims=config.OBS_DIM, 
                                         n_frames=config.NUM_FRAMES)

            model = PPO(
                policy="MlpPolicy",
                env=env,
                learning_rate=phase["lr"],
                n_steps=N_STEPS,
                batch_size=BATCH_SIZE,
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

            # NOTE: manual_curriculum_callback is in src/, so we import it from there
            # Assuming it's in the python path
            callback = ManualCurriculumCallback(
                save_path=save_dir,
                start_phase=0,
                eval_interval=500,
                save_interval=config.SAVE_FREQ_STEPS
            )
            
            print("[Training] Press Ctrl + C to stop the training. ")

            model.learn(
                total_timesteps=steps, 
                callback=callback,
                tb_log_name=config.MODEL_NAME
            )
            
            model.save(os.path.join(save_dir, config.MODEL_NAME + "_FINAL"))
            env.save(os.path.join(save_dir, config.MODEL_NAME + "_vecnormalize_FINAL.pkl"))
            print("\\nProduction Training Complete!")
            
        except KeyboardInterrupt:
            print("\\n[MANUAL OVERRIDE] Training forcefully interrupted by user.")
            if model is not None: model.save(os.path.join(save_dir, config.MODEL_NAME + "_EMERGENCY"))
            if env is not None: env.save(os.path.join(save_dir, config.MODEL_NAME + "_vecnormalize_EMERGENCY.pkl"))

        except Exception as e:
            print(f"\\n[CRITICAL ERROR] Training crashed: {e}")
            if model is not None: model.save(os.path.join(save_dir, config.MODEL_NAME + "_CRASH_SAVE"))
            if env is not None: env.save(os.path.join(save_dir, config.MODEL_NAME + "_vecnormalize_CRASH_SAVE.pkl"))

        finally:
            failsafe_env(
                env=env if 'env' in dir() else None,
                model=model if 'model' in dir() else None
            )

    def resume(self):
        pass

    def tune(self):
        pass

    def test(self):
        pass
"""

with open("src/agents/ppo/agent.py", "w") as f:
    f.write(content)
print("agent.py created successfully!")
