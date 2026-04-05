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

---

## Virtual environment (venv)

> Using a virtual environment is heavily recommended for this project.

