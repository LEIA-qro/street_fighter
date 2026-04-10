# Documentation

---
# High-Level Summary

Transitioning from mathematical theory to bare-metal implementation requires strict modularity. To successfully bridge BizHawk and Python via Inter-Process Communication (IPC), we will establish a rigid synchronous TCP Client-Server architecture. Python will act as the TCP Server, waiting for a connection. BizHawk (via Lua) will act as the TCP Client, connecting to Python, sending the RAM state, and waiting for the neural network's action before advancing the emulator frame.

## Data recolection and BizHawk (Lua) - Python connection

The Street Fighter II' - Special Champion Edition (USA) rom uses a base-conversion artifact of the Motorola 68000 processor architecture used in the Genesis. The Genesis WRAM is mapped to the 0xFF0000 memory range this is an **Hexadecimal (Base-16)** address format. To extract this data we have to figure out, where are the changes or the data adresses that we want. 

Thankfully Bizhawk already counts with integrated tools that can help us map the desired addresses. Inside BizHawk/EmuHawk, go to Tools -> There you will be able to fin RAM Watch & RAM Search. Open both. In [`doc`](doc) there is a file called **Street Fighter II' - Special Champion Edition (U) [!].wch**, you can load this inside RAM Watch to continue searching for other desired RAM addresses. There are different techniques to finding new RAM locations, most of the player-related RAM locations are in the 0xFF8000 - 0xFF9000 range.

> Potential Data Leakage in addresses 81E2 & 845E, this correspond to the "Buttons Pressed by Player 1" (81E2) and "Buttons Pressed by Player 1" (845E). Including "Buttons Pressed by Player 2" (81E2) in Player 1's observation space is a classic Machine Learning pitfall known as Data Leakage. The agent's policy network is responsible for generating the button presses. If you feed the current button press as an input state, the network can collapse into an identity-mapping loop.

> However, you could use 845E (Player 2's inputs), or its counterpart 81E2 (when training as player 2), if you want your AI to have superhuman reaction times (reading inputs before animations start), but for fair "human-like" AI, rely only on P2's physical state/position.

With this set, in the Lua script the RAM values will be read with `mainmemory.read_u16_be(RAM_LOCATION)`. 

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


