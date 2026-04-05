# Reinforcement Learning – Street Fighter


_Windows Setup guide (English)_

---

## Project overview

This repository contains code and experiments for Reinforcement Learning agents applied to a Street Fighter II' - Special Champion Edition (USA) environment. The project is designed to run with **Python 3.13.12** & **Lua 5.4.6** inside an isolated virtual environment using **BizHawk 2.8** to guarantee compatibility.

The project aims to make an easy and straightforward implementation of Reinforcement Learning via Python, there already exists an already implemented alternative to this project, you can find it here: [Build a Street Fighter AI Model with Python | Gaming Reinforcement Learning Full Course](https://www.youtube.com/watch?v=rzbFhu6So5U)

This is great for understanding the basics of the project, but it relys heavily on **gym retro**, a very old library that has multiple difficulties with different configuratioons and in game changes. Therefore this project creates a custom **RL pipeline** with robust **lock-step TCP bridge** between **Python** and **Bizhawk**, that allows a production-grade approach to reinforcement learning. With the sole purpose of achieving a manual curriculum based architecture.

This allows to have an absolute control over every variable of the training, 

---

## Video Tutorial

---

## Prerequisites

### Install Python

### Install Lua

### Install Bizhawk

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

> Note. You can Skip this part and go directly to __Training__, if you suspect something is wrong, or want to debbug then proceed.

#### Checking  if __Bixhawk__ and __Python__ are connected.

Run `test_telemetry2.py`

When running, _Bizhawk_ and a _Lua Console_ should pop up, the Python script is configured to start the Lua Script automatically, because it is set to making random actions, it is adviced to untoggle or pause the Lua script to facilitate navigation inside the ROM, you can do this by double clicking on it or clicking the Toggle Script button.
After this, load or start a match, it can be any character, and right before the match start, activate or toggle the Lua script.
You should be able to see the player 1, doing random actions.

If this is the case, the TCP bridge between Bizhawk and Python is working. You can close Bizkawk.

If this is not the case, ensure you have loaded the correct BIZHAWK_PATH, ROM_PATH, and LUA_SCRIPT_PATH variables to the Python script, match your local setup before running this test.
Alternatively, check if all of the versions required for the project are sound. `Python 3.13.12`, `Lua 5.4.6` and `BizHawk 2.8`. Specially `Bizhawk 2.8`, Python and Lua have not been tested in other versions, but because the project was made with this specific versions, using others might cause trouble.

#### Checking  if `stable_baselines3` & `gymnasium` are working.

Run `random_test.py`

When running, _Bizhawk_ and a _Lua Console_ should pop up, creating one instance of a "training env", you should be able to see how the agent is making random actions, the ROM is unthrotled, meaning is running at the highest performance, and the match should autostart every time either the agent wins or loses. This is how the training will be happening, but with more instances.

If this i the case then, you have all set to start.

If this is not the case, check if you have correctly installed the dependencies. 

### Training

### Testing AI Models

---
## Creating a Model



