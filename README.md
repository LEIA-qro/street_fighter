# Reinforcement Learning – Street Fighter

[![Python](https://img.shields.io/badge/python-3.13.12-blue.svg)](https://www.python.org/)


_Windows Setup guide (English)_

---

## Project overview

This repository contains code and experiments for Reinforcement Learning agents applied to a Street Fighter II' - Special Champion Edition (USA) environment. The project is designed to run with **Python 3.13.12** & **Lua 5.4.6** inside an isolated virtual environment using **BizHawk 2.8** to guarantee compatibility.

The project aims to make an easy and straightforward implementation of Reinforcement Learning via Python, there already exists an already implemented alternative to this project, you can find it here: [Build a Street Fighter AI Model with Python | Gaming Reinforcement Learning Full Course](https://www.youtube.com/watch?v=rzbFhu6So5U)

This is great for understanding the basics of the project, but it relies heavily on **gym retro**, a very old library that has multiple difficulties with different configurations and in-game changes. Therefore this project creates a custom **RL pipeline** with robust **lock-step TCP bridge** between **Python** and **Bizhawk**, enabling a production-grade approach to reinforcement learning, with the sole purpose of achieving a manual curriculum based architecture.

This allows to have an absolute control over every variable of the training.

### How Does it works?

(1) Python launches BizHawk → (2) Lua reads RAM → (3) TCP sends game state → (4) Python computes action → (5) Lua injects inputs.

---

## Video Tutorial

---

## Prerequisites

### Install Python 3.13.12

> Note. It could work in any version, but this project was built using [Python 3.13.12](https://www.python.org/downloads/release/python-31312/) to ensure full compatibility and avoid unwanted problems.

Install [Python 3.13.12](https://www.python.org/downloads/release/python-31312/).

### Install Lua 5.4.6

Install [Lua](https://www.lua.org/download.html)

### Install Bizhawk 2.8

Install [Bizhawk 2.8 from web](https://tasvideos.org/Bizhawk/PreviousReleaseHistory)

Install [Bizhawk 2.8 from Github](https://github.com/TASEmulators/BizHawk/releases/tag/2.8)

---

## Virtual environment (venv)

> Using a virtual environment is heavily recommended for this project.

### Using Visual Studio Code

<details>
  <ol>
    <li>Install <strong>Python Environments</strong> extension in VS Code</li>
    <li>In Environment Manager >> Create Environment >> Use Python 3.13.12</li>
  </ol>
</details>

### Other Alternatives

<details>
  From your project folder run:
  
  <details>
    <summary>
      If python is not in PATH:
    </summary>
  
    
    bash
    py -3.13.12 -m venv venv
    
  
  </details>
  
  <details>
    <summary>
      If python is available in PATH:
    </summary>
    
    bash
    python3.13.12 -m venv venv
    
  
  </details>
  
  > This should create a folder named `venv` containing the isolated environment.
  
  <br>
</details>



### Activating the virtual environment

<details>
  <strong>PowerShell</strong>:

  ```powershell
  .\venv\Scripts\Activate
  ```
  
  **Warning**
  > PowerShell may block script execution by default.  
  > If you see an error about `ExecutionPolicy`, either switch to **CMD** (below)  
  > or run PowerShell as administrator (this might be more flexible, but for simplicity use CMD):
</details>




---

## Dependencies

<ul>
  <li>stable_baselines3</li>
  <li>gymnasium</li>
  <li>torch</li>
  <li>optuna</li>
  <li>numpy</li>
  <li>pandas</li>
  <li>tensorboard</li>
</ul>

To install the dependencies, in a new terminal, go to the project folder _(it is recommended to have a venv activated)_, and run the following:
```
pip install stable_baselines3 gymnasium torch optuna numpy pandas tensorboard
```
Alternatively you can run:
```
pip install  requirements.txt
```

---

## Getting Started

### Testing

> Note. You can Skip this part and go directly to __Training__, if you suspect something is wrong, or want to debug then proceed.

#### Checking  if __Bixhawk__ and __Python__ are connected.

Run `test_telemetry2.py`. You can find this script in the `testing` folder. 

When running, _Bizhawk_ and a _Lua Console_ should pop up, the Python script is configured to start the Lua Script automatically, because it is set to making random actions, it is advised to untoggle or pause the Lua script to facilitate navigation inside the ROM, you can do this by double clicking on it or clicking the Toggle Script button.
After this, load or start a match, it can be any character, and right before the match start, activate or toggle the Lua script.
You should be able to see the player 1, doing random actions.

If this is the case, the TCP bridge between Bizhawk and Python is working. You can close Bizkawk.

If this is not the case, ensure you have loaded the correct BIZHAWK_PATH, ROM_PATH, and LUA_SCRIPT_PATH variables to the Python script, match your local setup before running this test.
Alternatively, check if all of the versions required for the project are sound. `Python 3.13.12`, `Lua 5.4.6` and `BizHawk 2.8`. Specially `Bizhawk 2.8`, Python and Lua have not been tested in other versions, but because the project was made with this specific versions, using others might cause trouble.

#### Checking  if `stable_baselines3` & `gymnasium` are working.

Run `random_test.py`. You can find this script in the `testing` folder. 

When running, _Bizhawk_ and a _Lua Console_ should pop up, creating one instance of a "training env", you should be able to see how the agent is making random actions, the ROM is unthrotled, meaning is running at the highest performance, and the match should autostart every time either the agent wins or loses. This is how training will happen, but with more instances.

If this is the case then, you have all set to start.

If this is not the case, check if you have correctly installed the dependencies. 

### Training

Training a model is the sole purpose of this project. The way the code is built is to train a model based on the character RYU, this can be changed, check the documentation if you wish to train the model with another character or another configuration.

There are only four training scripts:

<ul>
  <li><code>train_production_PPO_v2.py</code></li>
  <li><code>resume_production_v2.py</code></li>
  <li><code>train_optuna.py</code></li>
  <li><code>transfer_optuna.py</code></li>  
</ul>

> Note. <code>transfer_optuna.py</code> is currently under development

#### `train_production_PPO_v2.py`

You can find this script in the `training` folder. 

This script initializes a model, creates it from scratch. Uses the hyperparameters set in `config.py`.

Check the documentation (`doc` folder) for further explanation on how the code works and how to configure it according to your needs.

#### `resume_production_v2.py`

You can find this script in the `training` folder. 

This script, allows you to continue the training of an already existing model, loads the normalization stats and the neural network from `config.py`.

Check the documentation (`doc` folder) for further explanation on how the code works and how to configure it according to your needs.

#### `train_optuna.py`

You can find this script in the `training` folder. 

One of the most important scripts, this script allows _optuna_ to find the best hyperparameters of the model, without this the model could be capable of training, but would not be training in the most optimized and efficient way, slowing down the convergence , and in some cases, making it impossible to converge if the hyperparameter are not well tuned.

Check the documentation (`doc` folder) for further explanation on how the code works and how to configure it according to your needs.

#### `transfer_optuna.py`

> Currently under development

You will be able to find this script in the `training` folder. 

This script is intentioned to be used for a curriculum training, allows to load an already existing model into an optuna study, works for hyperparameter tunning, not changing the already existing architecture of the model _(n_steps and batch_size)_, just changing _the search space_ being the _learning rate_, _ent coef_ and the _clip range_.

Check the documentation (`doc` folder) for further explanation on how the code works and how to configure it according to your needs.

---
## Checking How good is the model

To check how good is the model we handle different metrics.

You can check all of them running in the Terminal of the project:

```Terminal
tensorboard --logdir=logs\
```

### `ep_len_mean` and `ep_rew_mean`

`ep_len_mean`: The episode leangth mean indicates how long in average the episodes are lasting every value represents a frame, for example if the episode length mean is of 1500, this means that in average the matches are lasting 100 seconds, since a second is 60 frames and we are using a FRAME SKIPING of 4 (Check the documentation), it means that 1500 * 4 / 60 = 100

`ep_rew_mean`: Episode reward mean, this indicates what the reward average of the episodes is, it has a complete correlation with the REWARD function, it tells us how good the model is performing in relation with the REWARD function.

If the `ep_len_mean` is low and the `ep_rew_mean` is high, it means that the model is succesfully beating every oponent. But if the `ep_len_mean` is low and the `ep_rew_mean` is also very low, this means the model is getting his ass kicked.

### Other Metrics

`train/policy_gradient_loss`: measures how much the policy is being pushed to change each update. You want this to trend gradually downward and stay small. If it's spiking erratically, the agent is receiving inconsistent gradient signals, which usually means the reward function has too much variance or your learning rate is too high.

`train/value_loss`: how wrong the critic (value function) is when predicting expected return. Early training: high and dropping. If it plateaus at a high value, the critic can't accurately predict reward from the 554-dim obs, which starves the policy of good advantage estimates. Watch this alongside `ep_rew_mean`.

`train/entropy_loss`: measures action diversity. High entropy means the agent is still exploring broadly; low entropy means it's committing to specific moves. If this collapses to near zero early in training, the agent has latched onto a narrow strategy (like spamming one button) and stopped exploring. The `ent_coef` in config directly controls this.

`train/approx_kl`: the KL divergence between the old and new policy per update. The config sets `target_kl=0.03`. If this consistently exceeds that threshold, SB3 will cut the update short, meaning the `n_epochs=10` is never fully used. A persistently high KL suggests the learning rate is too aggressive for the current phase.

`train/clip_fraction`: the fraction of gradient steps where the PPO clipping mechanism activated. Healthy range is roughly 0.05–0.20. Values above 0.30 mean the policy is trying to change too fast and PPO is constantly clamping it, wasting compute. Values near zero mean the policy is barely updating.

`train/explained_variance`: how much of the return variance the value function actually explains. Ranges from -∞ to 1.0; values below 0 mean the critic is worse than a constant baseline. You want this above 0.8 during stable training. Starting with a low explained variance is normal; if it's still low after a long time, the network architecture may need attention.

### Callback Metricks

Here we have the metric of `win_rate`, which obviously means how many games out of a episode window, set to 250 episodes in `config.py`, is wining. This checks the last 250 episodes and sees how many of them has won, therefore making a percentage called win rate. The higher the win rate the better.

### Testing AI Models

Once you have a trained Model, you can test it with the following:

<ul>
  <li>
    <code>test_agent_v2.py</code>
  </li>
  <li>
    <code>test_ai_vs_ai_v2.py</code>
  </li>
</ul>

#### `test_agent_v2.py`

You can find this script in the `testing` folder. 

Allows you to play against the model, uses the model set in `config.py`, you can either select in the Python script if you want the model to be player 1 or 2.
Alternatively you can also put the model to play against the other cpu oponents and see how far in the chalengers campaign can it go.

Check the documentation (`doc` folder) for further explanation on how the code works and how to configure it according to your needs.

#### `test_ai_vs_ai_v2.py`

You can find this script in the `testing` folder. 

This Python script allows you to load two different or same models to battle against each other. Uses the models set in `config.py`.
Load a Player vs Player battle, select the characters and toggle or activate the Lua script.

Check the documentation (`doc` folder) for further explanation on how the code works and how to configure it according to your needs.

---
## Creating a custom Model

There are different ways to create a custom model:

<ul>
  <li>
    Change the REWARD function inside env_sf2_v2.py. This affects the overall behavior of the model. 
  </li>

  <li>
    Change the Observation Space or the data passed to the model. Example: Passing extra information to the model, Lua gathers the data from the ram values of the ROM and passes it to Python bia the TCP bridge. [Note] Be carefull when editing the Lua script.
  </li>

  <li>
    Changing the <strong>trained character</strong>, specializing the model with another Character. This is the most fun customization, since you can fully select which character you want your model to specialize, it is far better to make an specialist agent than a globaly good agent, since the model is better and faster trained when specializing it. To do this, open Bizhawk without any script, load the ROM, and create a savestate for every new batle with that character. Check The documentation for full guide.
  </li>
</ul>


---
## Future Implementations

<ul>
  <li>
    Enhancing the Obs space, the actual model struggles heavily with projectiles. Even with projectile tracking it still has a big trouble understanding why avoiding or protecting from a projectile is a good idea.
  </li>

  <li>
    Enhancing the REWARD function, it is very hard to get the best REWARD function while avoiding the coward's local optimum, or REWARD hacking, the current REWARD function works, but there can be a posible better REWARD function that could accelerate convergence of the model.
  </li>

  <li>
    Including a UI and a compiled version of the project, currently everything runs with scripts and can be messy to explore and understand.
  </li>
</ul>

---
## Extra

For this project it was used a CPU Intel(R) Core(TM) Ultra 9 275HX. With a NVIDIA GeForce RTX 5070 Ti Laptop GPU



	







