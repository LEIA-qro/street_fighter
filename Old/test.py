import gym, retro
import os
import sys
import time

print("gym:", gym.__version__)    # 0.21.0
print("retro:", retro.__version__) # 0.8.0
print(f"Python version: {sys.version}")

print(retro.data.list_games())


# Start game environment
env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')


# Closes game environment since we can only run one at a time
# env.close()

# Test to see everything working

# Reset game to starting state
obs = env.reset()

# Flag to false
done = False

# We only play one game
for game in range(1):

    # If game is not over.
    while not done:
        if done:
            # We reset the game
            obs = env.reset()

        # Render environment
        env.render()

        # We take random actions inside the environment
        obs, reward, done, info = env.step(env.action_space.sample())

        # We slow down the renders so they are watchable
        time.sleep(0)

        # We print the reward
        print(reward)
