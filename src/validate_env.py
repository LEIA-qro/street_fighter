from stable_baselines3.common.env_checker import check_env # pip install stable-baselines3
from env_sf2 import StreetFighterEnv

def test_environment():
    print("Initializing Street Fighter Environment...")
    env = StreetFighterEnv()
    
    try:
        # 1. Strict API Validation
        print("Running Stable-Baselines3 Environment Checker...")
        # check_env will throw aggressive assertion errors if ANYTHING is wrong
        check_env(env, warn=True)
        print("\nSUCCESS: Environment passed all strict API checks!\n")
        
        # 2. Emulate a random agent for 100 steps
        print("Running Random Agent Test...")
        obs, info = env.reset()
        
        total_reward = 0.0
        
        for step in range(1, 1001):
            # Sample a valid random action from the MultiBinary(10) space
            random_action = env.action_space.sample()
            
            # Execute step
            obs, reward, terminated, truncated, info = env.step(random_action)
            total_reward += reward
            
            if step % 20 == 0:
                print(f"Step {step}/1000 | Action: {random_action} | Reward: {reward} | Terminated: {terminated}")
                
            # Handle episode end
            if terminated or truncated:
                print(f"--- Match Terminated at step {step} ---")
                obs, info = env.reset()
                
        print(f"\nRandom Agent Test Complete. Total Reward Accumulated: {total_reward}")
        
    except Exception as e:
        print(f"\nCRITICAL ERROR VALIDATING ENVIRONMENT: {e}")
        
    finally:
        # 3. Clean teardown
        env.close()

if __name__ == "__main__":
    test_environment()