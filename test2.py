# This test works in colab + conda env with gym-retro installed in python 3.8.10
# It only calculates the environment and does not render it.

# Import libraries
import gym, retro # For the environment
import time  # For slowing down fights
import os
import sys

# Import the libraries necessary for data preprocessing.

from gym import Env  # Base environment class for a wrapper
from gym.spaces import MultiBinary, Box  # Ensure we pick the correct action space type. (Space shapes for the environment)
import numpy as np  # To calculate frame delta
import cv2  # For grayscaling
from matplotlib import pyplot as plt  # For plotting observation images


# import os
os.environ['DISPLAY'] = ':1'

# import torch


# Create custom environment
class StreetFighter(Env):
    def __init__(self):

        # Inherit from our base environment
        super().__init__()

        # Specify action and observation spaces
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)  # We create our observation space based on the new size and colors
        self.action_space = MultiBinary(12)  # We replicate the base action environment

        # Startup and instance the game
        # The second parameter will limit actions to only valid ones.
        try:
            self.game = retro.make(
                game='StreetFighterIISpecialChampionEdition-Genesis', 
                use_restricted_actions=retro.Actions.FILTERED
            )
        except Exception as e:
            print(f"Error creating retro environment: {e}")
            sys.exit(1)


    def reset(self):
        # Return first frame, preprocess the frame, and define score back to 0.

        self.previous_frame = np.zeros(self.game.observation_space.shape)

        obs = self.game.reset()  # Will return our observation
        obs = self.preprocess(obs)  # We preprocess the observation

        self.health = 176  # Initial health
        self.enemy_health = 176

        self.matches_won = 0
        self.enemy_matches_won = 0
        
        
        # Game delta = Current_frame - Previous_frame
        # Preprocess
        self.previous_frame = obs

        # Attribute to hold delta score.
        self.score = 0

        return obs
    
    def preprocess(self, observation):
        # Grayscale, and resize frame
        
        # Grayscaling
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)

        # Resizing
        resize = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)
        
        channel = np.reshape(resize, (84, 84, 1))  # We add the grayscale layer since its what gym expects

        return channel

    def step(self, action):
        # We take a step, preprocess the observation, calculate frame delta and reshape the reward function

        # Take a step
        obs, reward, done, info = self.game.step(action)  # New step based on an action

        obs = self.preprocess(obs)  # We preprocess the observation

        # Frame delta

        # We subtract the current one from the previous one and then we set the current as the last one.
        frame_delta = obs  # - self.previous_frame
        # self.previous_frame = obs

        # Reshape the reward function based on relative score
        # reward = info['score'] - self.score  # Current reward minus the previous score
        # self.score = info['score']  # We set our score to the current score.
        
        delta_enemy = (self.enemy_health - info['enemy_health']) / 176
        delta_self = (info['health'] - self.health) / 176
        reward = delta_enemy * 10 - delta_self * 5

        if abs(delta_enemy) < 1e-6 and abs(delta_self) < 1e-6:
            reward -= 0.001 # small penalty for idling

        if done:
            reward += (info['matches_won'] - info['enemy_matches_won']) * 20  # match win/loss bonus
            if info["enemy_health"] <= 0:
                reward += 50  # big win bonus
            elif info["health"] <= 0:
                reward -= 60  # big loss penalty
            


                
        delta_score = info.get('score', 0) - getattr(self, 'score', 0)
        self.score = info.get('score', self.score)
        reward += delta_score * 0.001


       # Update values
        self.health = info['health']
        self.enemy_health = info['enemy_health'] 

        return frame_delta, reward, done, info


    def render(self, *args, **kwargs):
        # We render the game
        self.game.render()

    def close(self):
        # We close the game
        self.game.close()




# Test to see everything working


for game in range(1):
    # Reset game to starting state
    try:
        env = StreetFighter()
        obs = env.reset()
        done = False
        reward_sum = 0
        number_of_rewards = 0
        
        steps = 0
        # max_steps = 1000  # Límite para pruebas
        
        while not done:  # and steps < max_steps:
            if done:
                # We reset the game
                obs = env.reset()

            # COMENTA ESTA LÍNEA - No renderizar en entorno headless
            # env.render()

            obs, reward, done, info = env.step(env.action_space.sample())
            
            if reward != 0:
                print(f"Step {steps}: Reward: {reward:.3f}")
            
            reward_sum += reward
            number_of_rewards += 1
            steps += 1
            
            if steps % 100 == 0:
                print(f"Step {steps}, Total reward: {reward_sum:.2f}")
        
        print(f"Game {game+1} finished")
        done = False  # Reset for next game if you have multiple
        obs = env.reset()

        print(f"Finished after {steps} steps")
        print(f"Mean reward: {reward_sum / number_of_rewards if number_of_rewards > 0 else 0}")
        
        env.close()
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()  
