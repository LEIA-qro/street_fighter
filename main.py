# 

# Set up libraries
import gym, retro # For the environment
import time  # For slowing down fights
import os
import sys

# Preprocessing the environment
# Import the libraries necessary for data preprocessing.
from gym import Env  # Base environment class for a wrapper
from gym.spaces import MultiBinary, Box  # Ensure we pick the correct action space type. (Space shapes for the environment)
import numpy as np  # To calculate frame delta
import cv2  # For grayscaling
from matplotlib import pyplot as plt  # For plotting observation images

# Hyperparameter tunning
import optuna  # Importing the optimization framework that allows to both train and tune at the same time
import torch
import os  # For exporting the model
from stable_baselines3 import PPO  # PPO algorithm for RL
from stable_baselines3.common.evaluation import evaluate_policy  # Metric calculation of agent performance
from stable_baselines3.common.monitor import Monitor  # SB3 Monitor for logging
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack  # Vec wrappers to vectorize and frame stack
import tensorboard as tb


# import os
os.environ['DISPLAY'] = ':1'

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

# Directories where saved optimization models are going to be saved

LOG_DIR = './logs/'  # SB3 has the ability to log out to a support log
OPT_DIR = './opt/'  # Location to save every single model after every try

# Hyperparameter function to return test hyperparameters - define the objective function

def optimize_ppo(trial):  # i.e. objective
    return {
        # Ranges of possible values that will be optimized
        'n_steps': trial.suggest_int('n_steps', 2048, 8192, step=64),  # SB3 requires  the range to be a multiple of 64
        'gamma': trial.suggest_loguniform('gamma', 0.8, 0.999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-4),
        'clip_range': trial.suggest_uniform('clip_range', 0.1, 0.4),
        'gae_lambda': trial.suggest_uniform('gae_lambda', 0.8, 0.99),
    }


# Hyperparameter function to run a training loop and return mean 
# device = torch.device("cpu")  # Fuerza CPU

def optimize_agent(trial):
    # A try - except section can prevent the model from breaking mid-training

    model_params = optimize_ppo(trial)  # Variable where we store the parameters from the previous function

    # Create environment
    env = StreetFighter()
    env = Monitor(env, LOG_DIR)  # We specify the location where monitor values will be exported to
    env = DummyVecEnv([lambda: env])  # We wrap the environment on a DummyVec
    env = VecFrameStack(env, 4, channels_order='last')  # We will stack 4 different frames

    # Create training algorithm
    # model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=0, **model_params)  # We unpack the model parameters obtained from the tuner and pass them to the PPO model
    model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=0, **model_params)
    model.learn(total_timesteps=30000) #300000  # We train the model. Longer timesteps means a better model, but also a longer training time. 100k is good, 30k is quick but inaccurate
    
    # Evaluate model
    mean_reward = evaluate_policy(model, env, n_eval_episodes=5) #10  # We unpack the results obtained from evaluate policy. We will evaluate the model on 5 different games (more == better)
    env.close()

    SAVE_PATH = os.path.join(OPT_DIR, 'trial_{}_best_model'.format(trial.number))
    model.save(SAVE_PATH)  # We save all models to get the best one

    # We have to give optuna a value it expects, so if its a tuple we return only an int
    if isinstance(mean_reward, (tuple, list)):
        mean_reward = mean_reward[0]

    return mean_reward 
    try:
        pass

    except Exception as e:
        return -1000  # Model did not work, we resume training

# When we train we will get a set of best parameters

# Tuning

study = optuna.create_study(direction='maximize')  # We create the experiment / study that seeks to maximize the mean reward
study.optimize(optimize_agent, n_trials=1, n_jobs=1)  # We optimize the study based on the agent created, and how many sets we will set. 10 is good for testing, 100+ is recommended for a good model
# n_trials=20


# NOTE: Using 100k timesteps on the model and 100 trials can take a long time to train (depending on the strength of the gpu from a few hours to a couple of days)

# If we wanted to speed things up whilst keeping accuracy, we could raise n_jobs, however retro does not support more than one environment at once. We can fix
# this by using retrowrapper: https://github.com/MaxStrange/retrowrapper. This allows for multiple instances at once which exponentially speeds trainig up.


print("Study best params", study.best_params)  # We print the best parameters found
print( "Study best value", study.best_value)  # We print the best mean reward obtained


# Test to see everything working

