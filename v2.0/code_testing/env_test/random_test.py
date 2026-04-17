import os, sys, traceback, time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)
sys.path.append(os.path.join(project_root, "src"))

"""print(f"[TEST] Current Directory: {current_dir}")
print(f"[TEST] Parent Directory: {parent_dir}")
print(f"[TEST] Project Root: {project_root}")"""


from env_sf2_v2 import StreetFighterEnvV2
from env_tools import failsafe_env
import config


def random_test_telemetry():
    
    print("[TEST] Starting random telemetry test...")
    print("[TEST] Press Ctrl + C to stop the test. ")
    time.sleep(5)  # Give the user a moment to read the message before the test starts
    
    env = None   

    try:
        env = StreetFighterEnvV2() 
        env.active_training_states = config.RYU_ONLY_STATES_PHASE_0
        obs, _ = env.reset()
        print(f"[TEST] Environment reset. Observation shape: {obs.shape}") 
        print(f"[TEST] Active Training States: {env.active_training_states}")
        step = 0
        print("[TEST] Press Ctrl + C to stop the test. ")

        while True:
            random_action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(random_action)
            

            if step % 240 == 0:  # Print every 240 steps
                print(f"[TEST] Step {step} - Sampled Action: {random_action}")
                print(f"[TEST] Step {step} - Observation: {obs}")
                print(f"[TEST] Step {step} - Reward: {reward}, Terminated: {terminated}, Info: {info}")
            if terminated:
                print(f"[TEST] Episode ended at step {step+1}. Resetting environment.")
                obs, _ = env.reset()  # Capture return value; resets frame-stack correctly

            step += 1
    except KeyboardInterrupt:
        print("[TEST] Random telemetry test interrupted by user. Exiting gracefully.")
    except Exception:
        print(f"[TEST] An error occurred during the random telemetry test.")
        traceback.print_exc()
    finally:
        print("[TEST] Cleaning up resources...")
        failsafe_env(env=env)
    
if __name__ == "__main__":
    random_test_telemetry()