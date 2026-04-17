# Documentation

---
# High-Level Summary

Transitioning from mathematical theory to bare-metal implementation requires strict modularity. To successfully bridge BizHawk and Python via Inter-Process Communication (IPC), we will establish a rigid synchronous TCP Client-Server architecture. Python will act as the TCP Server, waiting for a connection. BizHawk (via Lua) will act as the TCP Client, connecting to Python, sending the RAM state, and waiting for the neural network's action before advancing the emulator frame.

## Data collection

The Street Fighter II' - Special Champion Edition (USA) ROM stores memory using the Motorola 68000 processor's big-endian addressing convention, mapped to the Genesis WRAM at the 0xFF0000 range, this is an **Hexadecimal (Base-16)** address format. To extract this data we have to figure out, where are the changes or the data adresses that we want. 

Thankfully Bizhawk already counts with integrated tools that can help us map the desired addresses. Inside BizHawk/EmuHawk, go to Tools -> There you will be able to find RAM Watch & RAM Search. Open both. In [`doc`](https://github.com/LEIA-qro/street_fighter/tree/main/doc) there is a file called **Street Fighter II' - Special Champion Edition (U) [!].wch**, you can load this inside RAM Watch to continue searching for other desired RAM addresses. There are different techniques to finding new RAM locations, most of the player-related RAM locations are in the 0xFF8000 - 0xFF9000 range.

> Potential Data Leakage in addresses 81E2 & 845E, this correspond to the "Buttons Pressed by Player 1" (81E2) and "Buttons Pressed by Player 2" (845E). Including "Buttons Pressed by Player 1" (81E2) in Player 1's observation space is a classic Machine Learning pitfall known as Data Leakage. The agent's policy network is responsible for generating the button presses. If you feed the current button press as an input state, the network can collapse into an identity-mapping loop.

> However, you could use 845E (Player 2's inputs), or its counterpart 81E2 (when training as player 2), if you want your AI to have superhuman reaction times (reading inputs before animations start), but for fair "human-like" AI, rely only on P2's physical state/position.

## BizHawk (Lua) - Python connection

### Lua

Once the desired RAM locations are found, in the Lua script the RAM values will be read with `mainmemory.read_u16_be(RAM_LOCATION)`. It is used `read_u16_be` specifically for big-endian unsigned 16-bit.

Example:

```Lua
local p1_hp = mainmemory.read_u16_be(0x8042)
local p2_hp = mainmemory.read_u16_be(0x82C2)
local p1_x  = mainmemory.read_u16_be(0x8006)
local p2_x  = mainmemory.read_u16_be(0x8358)
local p1_y  = mainmemory.read_u16_be(0x800A)
local p2_y  = mainmemory.read_u16_be(0x828A)
```

With this values stored, Lua has to create a formatted reply with  `string.format("0 %d\n")`.

Example:

```Lua
local payload = string.format("0 %d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n", 
    p1_hp, p2_hp, p1_x, p2_x, p1_y, p2_y, 
    p1_action_id, p2_action_id, 
    active_p1_proj_x, active_p2_proj_x,
    p1_char_id, p2_char_id)
```

Finally, once we have a formatted reply we use `comm.socketServerSend(payload)` to send it to Python.

```Lua
comm.socketServerSend(payload) -- This will send the string to Python
```

### Python

After the payload has been sent via Lua script, Python receives an encoded formated reply. Lets take for example the code inside `bizhawk_base.py`, found in [src](https://github.com/LEIA-qro/street_fighter/tree/main/src). In the `receive_payload()` function, inside the `BizHawkBaseEnv` class, we have the following:

```Python
def receive_payload(self) -> str:
    while '\n' not in self.stream_buffer:
        chunk = self.conn.recv(4096).decode('utf-8')
        if not chunk:
            return ""
        self.stream_buffer += chunk # We add the new elements for each iteration
    
    line, self.stream_buffer = self.stream_buffer.split('\n', 1) # We reset self.stream_buffer
    
    return line # We return line, where the RAM data is stored
```

The method blocks until a full newline-terminated payload is received, decodes it from UTF-8, and returns the CSV portion for parsing

## Input Injection

Once the model decides what its action is action according to the data, Python sends a formatted reply. 

Lua expects a specific format in the replies being:

```Python
formatted_reply = f"{len(command)} {command}" # Here we send first the length of the command and then the command itself
```

Then Python encodes the formatted reply with.

```Python
conn.sendall(formatted_reply.encode('utf-8'))
```

After this, Lua receives and reads the formatted reply.

```Lua
local response = ""  
while response == "" or response == nil do
    response = comm.socketServerResponse()
end

-- Remove the newline character for clean processing
response = string.gsub(response, "\n", "")
```

Then Lua parses the response and injects the inputs via Bizhawk commands.

```Lua
local input = {}

if string.sub(response, 1, 1) == "1" then input["P1 Up"] = true end
if string.sub(response, 2, 2) == "1" then input["P1 Down"] = true end
if string.sub(response, 3, 3) == "1" then input["P1 Left"] = true end
if string.sub(response, 4, 4) == "1" then input["P1 Right"] = true end
if string.sub(response, 5, 5) == "1" then input["P1 A"] = true end
if string.sub(response, 6, 6) == "1" then input["P1 B"] = true end
if string.sub(response, 7, 7) == "1" then input["P1 C"] = true end
if string.sub(response, 8, 8) == "1" then input["P1 X"] = true end
if string.sub(response, 9, 9) == "1" then input["P1 Y"] = true end
if string.sub(response, 10, 10) == "1" then input["P1 Z"] = true end
-- We sleep the Start and Mode buttons to avoid the agent accidentally pausing or opening the menu
-- if string.sub(response, 11, 11) == "1" then input["P1 Start"] = true end
-- if string.sub(response, 12, 12) == "1" then input["P1 Mode"] = true end

joypad.set(input)
emu.frameadvance()
```

Alternatively, inside the scripts of the project, it is applied a FRAME skipping technique to emulate real human input.

``` Lua
for i = 1, FRAME_SKIP do
    joypad.set(input)
    emu.frameadvance()
end
```

## Optimization: Maximizing Throughput

To achieve mathematical convergence in a reasonable timeframe, the environment must operate purely as a mathematical state-machine, completely divorced from human perceivable time.

We must apply the following three optimizations to BizHawk:

1. Clock Unthrottling (Mandatory)
We must uncap the framerate so the CPU processes emu.frameadvance() as fast as its clock speed allows (often pushing the emulator to 400–800 FPS depending on your processor).
How to apply: In BizHawk's top menu, go to Config -> Speed/Skip -> Unthrottle (or press the Tab key). Alternatively, we will force this via Lua.

However, if you wish **not to unthrottle** the Bizhawks instances, you can also set the game speed to a higher more stable speed. Inside the Lua script, check for `emu.limitframerate(false)`
 and replace it with:
 ```Lua
client.speedmode(200) -- current speed is %200, change the argument for other configurations
```

3. Disable Audio Emulation (Highly Recommended)
Generating audio waveforms and syncing them to the soundcard is highly CPU-intensive. Because the agent is learning purely from RAM vectors (X/Y coordinates, HP, Action IDs), audio provides zero mathematical value to the Markov Decision Process.
How to apply: In BizHawk, go to Config -> Sound -> uncheck Enable Sound. Alternatively, this will be forced via Lua.

4. Minimize Video Rendering Overhead (Highly Recommended)
While we cannot easily run BizHawk purely "headless" (without a GUI) in this specific configuration, we can minimize the resources it spends drawing pixels.
How to apply: * Go to Config -> Display -> Ensure VSync is strictly Off. VSync forces the emulator to wait for your monitor's physical refresh rate, destroying training speed.
Shrink the BizHawk window to its minimum possible size on your desktop. Fewer pixels to scale and draw means fewer CPU/GPU cycles wasted on UI rendering.

These settings are applied automatically by the Lua training script; manual configuration is only needed when running BizHawk outside of training mode.

---
# Training a Model

Training a model is the sole purpose of this project. The way the code is built is to train a model based on the character RYU, this can be changed following these steps [Changing the trained character](#changing-the-trained-character).

## Training Configurations



### Changing the trained character

If you wish to change your focused character (the character played by the AI), do this:

1. Open Bizhawk without Python
2. Load the ROM, File -> Open ROM.. or `Ctrl` + `O`.
3. Go to Options and set your in-game configurations -> Set your configurations -> Press `Enter` or the **Start** button when ready.

> It is highly recommended to set the In-Game configurations as they were used in the project, specifically: no time limit, so the match always ends with a winner rather than a timeout. For the manual curriculum we used a scaling difficulty, you can select the difficulty you want to train  your model, just remember that for every increasing difficulty the model might have more problems to converge. Every other configuration was left as default.

4. Start a match. Go to Champion -> Game Start -> **Select your DESIRED CHARACTER** -> Pause the game **Very Important**, see the next step.

> There is no easy way to manually select your oponent, the game handles an initial random phase, where you fight the first 8 opponents in a random order, being Ryu, Ken, E. Honda, Chun-Li, Blanka, Zangief, Guile and Dhalsim. The remaining oponents Balrog, Vega, Sagat and M Bison, appear in that order after you have defeated the first 8 opponents.

5. Before the match begins, there is a screen title where the characters cannot yet move. With the game paused use the advance frame key, set to `F` in the default config and in the exact frame the fight title disappears. Go to File -> Save State -> Save Named State... -> Go to the project directory -> Change the name (A name easy to understand example: RYU_BLANKA_R1_lvl3, where RYU is player 1, BLANKA is player 2, R1 is round 1, and lvl3 is the difficulty level) -> Then place the file inside the **states** folder.

6. Repeat for the amount of Battles your Model will have. In most cases, more states is better. At least have more than one available state.

7. Give Python the instructions to find the states, in `config.py` inside [src](https://github.com/LEIA-qro/street_fighter/tree/main/src) folder, go to the bottom of the file in **States configuration** and create or edit a list, for example:

```Python
NEW_VEGA_STATES = [
    "VEGA_ZANGIEF_R1_lvl2.State", # Important: Be sure that this match the file names
    "VEGA_RYU_R1_lvl2.State",
    "VEGA_CHUNLI_R1_lvl2.State",
    "VEGA_KEN_R1_lvl2.State",
]
```

8. Just to make sure, change the `TRAINING_STATES` variable to your new variable.

``` Python
TRAINING_STATES = NEW_VEGA_STATES
```

9. With this set, check the training tutorials to see how to train your model!

---
## PPO

> Currently this is the only method to train the model, in the future there will be added more ways or other algorithms.

>  Note. <code>transfer_optuna.py</code> is currently under development

For PPO there are only four training scripts:

<ul>
  <li><code>train_production_PPO_v2.py</code></li>
  <li><code>resume_production_v2.py</code></li>
  <li><code>train_optuna.py</code></li>
  <li><code>transfer_optuna.py</code></li>  
</ul>

### Training Scripts

#### `train_production_PPO_v2.py`

You can find this script inside [`src/training`](src/training) folder. 

This script initializes a model, creates it from the imported code `env_tools`, which initializes a gymnasium class containing the SF (Street Fighter) env. The  code also uses the hyperparameters set in `config.py`.

It is worth understanding that the code has the states hardcoded for the manual curriculum, but you can change it at any time:

```Python
# In this part
config.TRAINING_STATES = config.CURRICULUM_PHASES[0]

# You can change it to other states, or leave the ones already set in config.py
config.TRAINING_STATES = config.NEW_VEGA_STATES # Alternatively you can delete this line
                                                # and automatically will select the states from config.TRAINING_STATES
```

This script creates N parallel RL instances declared with `config.N_ENVS`.

An important thing to understand about PPO are its **Hyperparameters** and other model configurations. When the model is initialized:

``` Python
model = PPO(
        policy="MlpPolicy", # Standard fully-connected policy network
        env=env,    # Here it uses the already instantiated SF envs
        learning_rate=phase["lr"],    # Hyperparameter
        n_steps=config.N_STEPS,    # Hyperparameter
        batch_size=config.BATCH_SIZE,    # Hyperparameter
        ent_coef=phase["ent_coef"],    # Hyperparameter
        clip_range=phase["clip"],    # Hyperparameter
        n_epochs=10,    # Hyperparameter
        gamma=0.99,    # Hyperparameter
        target_kl=0.03,    # Hyperparameter
        policy_kwargs=dict(net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256])),    # Hyperparameter
        verbose=1,    # Sets the verbose to 1
        tensorboard_log=directories["logs"],    # Allows graphic visualization
        device="cuda"    # Allows the use of the GPU for the calculations
        )
```

After this, the models learning process is initialized:

```Python
model.learn(
            total_timesteps=config.STARTING_TOTAL_TIMESTEPS, # The amount of steps the model will learn
            callback=callback, # This allows further model monitoring
            tb_log_name=config.MODEL_NAME # The name the model will have
        )
```


#### `resume_production_v2.py`

You can find this script inside [`src/training`](src/training) folder. 

This script, allows you to continue the training of an already existing model, loads the normalization stats and the neural network from `config.py`.

Similar to the `train_production_PPO_v2.py` file. 


#### `train_optuna.py`

You can find this script inside [`src/training`](src/training) folder. 

One of the most important scripts, this script allows _optuna_ to find the best hyperparameters of the model, without this the model could be capable of training, but would not be training in the most optimized and efficient way, slowing down the convergence , and in some cases, making it impossible to converge if the hyperparameter are not well tuned.


#### `transfer_optuna.py`

> Currently under development

You will be able to find this script inside [`src/training`](src/training) folder.  

This script is intentioned to be used for a curriculum training, allows to load an already existing model into an optuna study, works for hyperparameter tunning, not changing the already existing architecture of the model _(n_steps and batch_size)_, just changing _the search space_ being the _learning rate_, _ent coef_ and the _clip range_.

### Metrics

#### Checking How good is the model

To check how good is the model we handle different metrics.

You can check all of them running in the Terminal of the project:

```Terminal
tensorboard --logdir=logs\
```

#### `ep_len_mean` and `ep_rew_mean`

`ep_len_mean`: The episode leangth mean indicates how long in average the episodes are lasting every value represents a frame, for example if the episode length mean is of 1500, this means that in average the matches are lasting 150 seconds, since a second is 40 frames and we are using a FRAME SKIPING of 4, it means that 1500 * 4 / 40 = 150

`ep_rew_mean`: Episode reward mean, this indicates what the reward average of the episodes is, it has a complete correlation with the REWARD function, it tells us how good the model is performing in relation with the REWARD function.

If the `ep_len_mean` is low and the `ep_rew_mean` is high, it means that the model is succesfully beating every oponent. But if the `ep_len_mean` is low and the `ep_rew_mean` is also very low, this means the model is getting his ass kicked.

#### Other Metrics

`train/policy_gradient_loss`: measures how much the policy is being pushed to change each update. You want this to trend gradually downward and stay small. If it's spiking erratically, the agent is receiving inconsistent gradient signals, which usually means the reward function has too much variance or your learning rate is too high.

`train/value_loss`: how wrong the critic (value function) is when predicting expected return. Early training: high and dropping. If it plateaus at a high value, the critic can't accurately predict reward from the 554-dim obs, which starves the policy of good advantage estimates. Watch this alongside `ep_rew_mean`.

`train/entropy_loss`: measures action diversity. High entropy means the agent is still exploring broadly; low entropy means it's committing to specific moves. If this collapses to near zero early in training, the agent has latched onto a narrow strategy (like spamming one button) and stopped exploring. The `ent_coef` in config directly controls this.

`train/approx_kl`: the KL divergence between the old and new policy per update. The config sets `target_kl=0.03`. If this consistently exceeds that threshold, SB3 will cut the update short, meaning the `n_epochs=10` is never fully used. A persistently high KL suggests the learning rate is too aggressive for the current phase.

`train/clip_fraction`: the fraction of gradient steps where the PPO clipping mechanism activated. Healthy range is roughly 0.05–0.20. Values above 0.30 mean the policy is trying to change too fast and PPO is constantly clamping it, wasting compute. Values near zero mean the policy is barely updating.

`train/explained_variance`: how much of the return variance the value function actually explains. Ranges from -∞ to 1.0; values below 0 mean the critic is worse than a constant baseline. You want this above 0.8 during stable training. Starting with a low explained variance is normal; if it's still low after a long time, the network architecture may need attention.

#### Callback Metricks

Here we have the metric of `win_rate`, which as the name suggests, means how many games out of a episode window, set to 250 episodes in `config.py`, is wining. This checks the last 250 episodes and sees how many of them has won, therefore making a percentage called win rate. The higher the win rate the better.

---
# Testing a Model

Once you have a trained Model, you can test it with the following:

<ul>
  <li>
    <code>test_agent_v2.py</code>
  </li>
  <li>
    <code>test_ai_vs_ai_v2.py</code>
  </li>
</ul>

## `test_agent_v2.py`

You can find this script in the [`testing`](src/testing) folder. 

Allows you to play against the model, uses the model set in `config.py`, you can either select in the Python script if you want the model to be player 1 or 2.
Alternatively you can also put the model to play against the other cpu oponents and see how far in the chalengers campaign can it go.


## `test_ai_vs_ai_v2.py`

You can find this script in the [`testing`](src/testing) folder. 

This Python script allows you to load two different or same models to battle against each other. Uses the models set in `config.py`.
Load a Player vs Player battle, select the characters and toggle or activate the Lua script.



