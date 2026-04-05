# Reinforcement Learning – Street Fighter


_Windows Setup guide (English)_

---

## Project overview

This repository contains code and experiments for Reinforcement Learning agents applied to a Street Fighter environment. The project is designed to run with **Python 3.13.12** & **Lua 5.4.6** inside an isolated virtual environment using **BizHawk 2.8** to guarantee compatibility.

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

---

## Getting Started

### Testing

### Training

### Testing AI Models


