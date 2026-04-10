# Documentation

---
# High-Level Summary

Transitioning from mathematical theory to bare-metal implementation requires strict modularity. To successfully bridge BizHawk and Python via Inter-Process Communication (IPC), we will establish a rigid synchronous TCP Client-Server architecture. Python will act as the TCP Server, waiting for a connection. BizHawk (via Lua) will act as the TCP Client, connecting to Python, sending the RAM state, and waiting for the neural network's action before advancing the emulator frame.

## Data recolection

The Street Fighter II' - Special Champion Edition (USA) rom uses a base-conversion artifact of the Motorola 68000 processor architecture used in the Genesis. The Genesis WRAM is mapped to the 0xFF0000 memory range this is an **Hexadecimal (Base-16)** address format. To extract this data we have to figure out, where are the changes or the data adresses that we want. 

Thankfully Bizhawk already counts with integrated tools that can help us map the desired addresses. Inside BizHawk/EmuHawk, go to Tools -> There you will be able to fin RAM Watch & RAM Search. Open both. In [`doc`](https://github.com/LEIA-qro/street_fighter/blob/main/doc/) there is a file called **Street Fighter II' - Special Champion Edition (U) [!].wch**, you can load this inside RAM Watch to continue searching for other desired RAM addresses. There are different techniques to finding new RAM locations, most of the player-related RAM locations are in the 0xFF8000 - 0xFF9000 range.

> Potential Data Leakage in addresses 81E2 & 845E, this correspond to the "Buttons Pressed by Player 1" (81E2) and "Buttons Pressed by Player 2" (845E). Including "Buttons Pressed by Player 1" (81E2) in Player 1's observation space is a classic Machine Learning pitfall known as Data Leakage. The agent's policy network is responsible for generating the button presses. If you feed the current button press as an input state, the network can collapse into an identity-mapping loop.

> However, you could use 845E (Player 2's inputs), or its counterpart 81E2 (when training as player 2), if you want your AI to have superhuman reaction times (reading inputs before animations start), but for fair "human-like" AI, rely only on P2's physical state/position.

## BizHawk (Lua) - Python connection

### Lua

Once the desired RAM location are found, in the Lua script the RAM values will be read with `mainmemory.read_u16_be(RAM_LOCATION)`. 

Example:

```Lua
local p1_hp = mainmemory.read_u16_be(0x8042)
local p2_hp = mainmemory.read_u16_be(0x82C2)
local p1_x  = mainmemory.read_u16_be(0x8006)
local p2_x  = mainmemory.read_u16_be(0x8358)
local p1_y  = mainmemory.read_u16_be(0x800A)
local p2_y  = mainmemory.read_u16_be(0x828A)
```

With this values stored, Lua has to create a formated repply with  `string.format("0 %d\n")`.

Example:

```Lua
local payload = string.format("0 %d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n", 
    p1_hp, p2_hp, p1_x, p2_x, p1_y, p2_y, 
    p1_action_id, p2_action_id, 
    active_p1_proj_x, active_p2_proj_x,
    p1_char_id, p2_char_id)
```

Finally, once we have a formatted repply we use `comm.socketServerSend(payload)` to send it to Python.

```Lua
comm.socketServerSend(payload) -- This will send the string to Python
```

### Python

After the payload has been sent via Lua script, Python receives an encoded formated repply. Python has to decode the repply and separate the data. Lets take for example the code inside `bizhawk_base.py`, found in [src](https://github.com/LEIA-qro/street_fighter/tree/main/src). In the `receive_payload()` function, inside the `BizHawkBaseEnv` class, we habe the following:

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

`self.conn.recv(4096)`, this calls the `recv()` method on a socket connection object (stored in `self.conn`). The **4096** parameter specifies the maximum number of bytes to receive from the network buffer. The method returns a **bytes** object containing the raw data received from the network. This is a blocking call, meaning your program pauses here until data arrives. If no data is available, it blocks until data arrives or the connection closes. 

`.decode('utf-8')`, this method is called immediately on the bytes object returned from `recv()`. It converts the raw bytes into a human-readable string using UTF-8 encoding, which is the most common text encoding for modern applications. The method returns a `str` object.

Together, this line receives up to 4096 bytes of data from the network and converts them from bytes to UTF-8 text in one chained operation.

## Input Injection

Once the model decides what is his action according to the data, Python sends a formatted repply. Lua accepts a specific format in the replies, 


## Optimization: Maximizing Throughput

To achieve mathematical convergence in a reasonable timeframe, the environment must operate purely as a mathematical state-machine, completely divorced from human perceivable time.

We must apply the following three optimizations to BizHawk:

1. Clock Unthrottling (Mandatory)
We must uncap the framerate so the CPU processes emu.frameadvance() as fast as its clock speed allows (often pushing the emulator to 400–800 FPS depending on your processor).
How to apply: In BizHawk's top menu, go to Config -> Speed/Skip -> Unthrottle (or press the Tab key). Alternatively, we will force this via Lua.

2. Disable Audio Emulation (Highly Recommended)
Generating audio waveforms and syncing them to the soundcard is highly CPU-intensive. Because the agent is learning purely from RAM vectors (X/Y coordinates, HP, Action IDs), audio provides zero mathematical value to the Markov Decision Process.
How to apply: In BizHawk, go to Config -> Sound -> uncheck Enable Sound. Alternatively, this will be forced via Lua.

4. Minimize Video Rendering Overhead (Highly Recommended)
While we cannot easily run BizHawk purely "headless" (without a GUI) in this specific configuration, we can minimize the resources it spends drawing pixels.
How to apply: * Go to Config -> Display -> Ensure VSync is strictly Off. VSync forces the emulator to wait for your monitor's physical refresh rate, destroying training speed.
Shrink the BizHawk window to its minimum possible size on your desktop. Fewer pixels to scale and draw means fewer CPU/GPU cycles wasted on UI rendering.


