# Reinforcement Learning – Street Fighter

<details>
  <summary>Cosas por cambiar del readme</summary>
    - instalar jupyther
    - en \roms ejecutar:   python -m retro.import
    - ejecutar test.py en el venv:  python test.py
    <br>
    Press <kbd>Ctrl</kbd> + <kbd>C</kbd> to copy
    H<sub>2</sub>O and x<sup>2</sup>
    <details>
      <summary>Click to expand</summary>
        This content is hidden by default!
        ```python
        print("Hidden code example")
        ```
    </details>
</details>



[![Python](https://img.shields.io/badge/python-3.8.10-blue.svg)](https://www.python.org/)

_Windows Setup guide (English)_

---

## Project overview

This repository contains code and experiments for Reinforcement Learning agents applied to a Street Fighter environment. The project is designed to run with **Python 3.8.10** inside an isolated virtual environment to guarantee compatibility.



<br>


---

## Video Tutorial

[![Ver en YouTube](https://img.youtube.com/vi/ubqhopAanQM/maxresdefault.jpg)](https://youtu.be/ubqhopAanQM)

---

## Prerequisites
- Install [**Python 3.8.10**](https://www.python.org/downloads/release/python-3810/)

> **IMPORTANT**: Select custom installation and add python to the **PATH**.
  
_Verify installation in the terminal:_

```bash
python --version
```
<br>

If it was installed correctly, it will return:
```
Python 3.8.10
```


<br>


---

## Virtual environment (venv)

> Using a virtual environment is heavily recommended for this project. 


### 1. Create the venv

From your project folder run:

<details>
  <summary>
    If python is not in PATH:
  </summary>

  <br>
  
  ```bash
  py -3.8 -m venv venv
  ```

</details>

<details>
  <summary>
    If python is available in PATH:
  </summary>
  <br>
  
  ```bash
  python3.8 -m venv venv
  ```

</details>

> This should create a folder named `venv` containing the isolated environment.

<br>

### 2. Activate the virtual environment

**PowerShell**:

```powershell
.\venv\Scripts\Activate
```

**Warning**
> PowerShell may block script execution by default.  
> If you see an error about `ExecutionPolicy`, either switch to **CMD** (below)  
> or run PowerShell as administrator (this might be more flexible, but for simplicity use CMD) and set:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

<br>

**Command Prompt (CMD)**:

```cmd
venv\Scripts\activate
```

When active, one should see the prompt prefixed with `(venv)`:

```
(venv) C:\Users\User\project>
```

<br>

To **deactivate** the venv simply type:

```bash
deactivate
```

---

### 3. Verify interpreter

1. **Python: Select Interpreter**.  
2. Select the interpreter that points to:

```
...\venv\Scripts\python.exe
```

<br>

Quick script to confirm the running interpreter:

```python
import sys
print("Python running from:", sys.executable)
```

Expected when correct:

```
C:\Users\User\Desktop\Coding\street_fighter\venv\Scripts\python.exe
```

If it shows a path under `AppData\Local\Programs\Python\...`  
you are using the global Python and not the venv.


<br>

### Common `pip` / compatibility notes (Windows + Python 3.8)
- Newer releases of `pip` (25.x) may use typing features that are only compatible with Python >= 3.9. If `pip` inside the venv throws errors like `TypeError: 'ABCMeta' object is not subscriptable`, then the venv's pip is broken.

<br>
<details>
  <summary>Regenerating VENV</summary>

  
  <br>
  
  1. Remove the broken venv:

  ```cmd
  rmdir /s /q venv
  ```
  <br>
  
  2. Recreate it with Python 3.8:
  
  ```cmd
  py -3.8 -m venv venv
  ```
  <br>
  
  3. Activate and verify `pip`:
  
  ```cmd
  venv\Scripts\activate
  pip --version
  ```
  
  You should see a `pip` version in the `23.x` series.
  
  ---
</details>


<br>


---
## Dependencies
<br>

With the virtual environment active, install the required packages:

```bash
pip install gym==0.21.0 gym-retro==0.8.0
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install stable-baselines3==1.7.0
pip install optuna
pip install "shimmy>=2.0"
```


<br>

(Optional) create a `requirements.txt` once your dependencies are installed:

```bash
pip freeze > requirements.txt
```
<br>

To install from `requirements.txt` on another machine:

```bash
pip install -r requirements.txt
```


<br>


---
## Testing
Load `test.py` from the repository.

Run:

```bash
python test.py
```
