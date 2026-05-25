import gymnasium as gym
import socket
import subprocess
import time
import sys

import core.config as config

class BizHawkBaseEnv(gym.Env):
    """Universal Base Environment for BizHawk socket communication."""
    
    def __init__(self, bizhawk_path, rom_path, lua_path, host, port, trainable=True, debug_mode=False, verbose=True):
        super().__init__()
        # Initialization parameters
        self.bizhawk_path = bizhawk_path
        self.rom_path = rom_path
        self.lua_path = lua_path
        self.host = host
        self.port = port
        self.trainable = trainable
        self.verbose = verbose
        
        # Internal state
        self.server_socket = None
        self.conn = None
        self.emulator_process = None # Track the subprocess

        # NEW: The TCP Holding Tank
        self.stream_buffer = ""
        
        # Debugging
        self.debug_mode = debug_mode
        self.step_count = 0
        self.step_debug_interval = 10000  # Print debug info every N steps

        # When initialized, start the emulator and establish the socket connection
        self._start_emulator_bridge()

    # For debugging: A simple method to print the current step count and received payload
    def debug_print(self, payload):
        if self.debug_mode:
            if self.step_count % self.step_debug_interval == 0:  # Print every N steps to avoid flooding the console
                print(f"[Step {self.step_count}] {payload}")
            self.step_count += 1

    def _start_emulator_bridge(self):
        """Binds the socket and launches the emulator."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        if self.verbose: print(f"Python ML Server actively listening on {self.host}:{self.port}...")
        
        if self.verbose: print("Launching BizHawk as a subprocess...")

        # Base arguments
        launch_args = [
            self.bizhawk_path, 
            self.rom_path, 
            f"--socket_ip={self.host}", 
            f"--socket_port={self.port}"
        ]

        # Always auto-inject the Lua script if provided (it will wait for dashboard trigger if in test mode)
        if self.lua_path:
            if self.verbose: print(f"Auto-loading Lua script: {self.lua_path}")
            launch_args.append(f"--lua={self.lua_path}")
            
        self.emulator_process = subprocess.Popen(launch_args)
        
        if not self.trainable:
            print("\n[INTERACTIVE MODE] BizHawk launched.")
            print("1. Navigate the game menus manually.")
            print("2. Use the 'Toggle Agent' button in the Dashboard to start/pause the AI.")
            print(f"\n[Connection] Waiting for your Lua connection...")
            
        self.conn, addr = self.server_socket.accept()
        
        # CONDITIONAL TIMEOUT: Strict failsafe for training, Infinite patience for testing
        if self.trainable:
            self.conn.settimeout(5.0)  # Safe but aggressive failsafe to prevent lock-step training hangs
        else:
            self.conn.settimeout(None) # Wait forever while human navigates menus
            
        if self.verbose: print(f"[Connection] Connection established with BizHawk at {addr}")
            

    def send_command(self, command: str):
        """Standardized protocol for sending a command to Lua."""
        try:
            formatted_reply = f"{len(command)} {command}"
            self.conn.sendall(formatted_reply.encode('utf-8'))

    
        except (ConnectionResetError, BrokenPipeError) as e:
            if self.trainable:
                raise RuntimeError(f"Socket broken during training: {e}")
            
            # Non-trainable (interactive) mode:
            # 1. Check if the emulator was closed by the user
            if self.emulator_process and self.emulator_process.poll() is not None:
                if self.verbose: print(f"\n[INFO] BizHawk process terminated. Exiting Python...")
                sys.exit(0)
            
            # 2. If it's still open, wait for a new Lua connection (e.g. script restart)
            if self.verbose: 
                print(f"[WARN] send_command failed: {e}")
                print(f"\n[Connection] Waiting for your Lua connection...")
            
            self.conn, addr = self.server_socket.accept()
            self.conn.settimeout(None) # Always wait forever in interactive mode
            if self.verbose: print(f"[Connection] Connection RE-ESTABLISHED at {addr}")


    def receive_payload(self) -> str:
        """Blocks and waits for a complete, mathematically perfect payload."""
        try:
            # Keep receiving bytes until we see a newline
            while '\n' not in self.stream_buffer:
                chunk = self.conn.recv(4096).decode('utf-8')
                if not chunk:
                    return ""
                self.stream_buffer += chunk
            
            # Slice the buffer precisely at the first newline.
            line, self.stream_buffer = self.stream_buffer.split('\n', 1)
            
            return line
            
        except socket.timeout:
            if self.verbose: print("\n[FAILSAFE] Python timed out waiting for BizHawk. Forcing crash...")
            raise RuntimeError("BizHawk Socket Timeout")
            
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as e:
            if self.trainable:
                raise RuntimeError(f"Socket broken during training: {e}")
            
            # Non-trainable (interactive) mode:
            # 1. Check if the emulator was closed by the user
            if self.emulator_process and self.emulator_process.poll() is not None:
                if self.verbose: print(f"\n[INFO] BizHawk process terminated. Exiting Python...")
                sys.exit(0)
            
            # 2. If it's still open, wait for a new Lua connection (e.g. script restart)
            if self.verbose: 
                print(f"[WARN] receive_payload failed: {e}")
                print(f"\n[Connection] Waiting for your Lua connection...")
            
            self.conn, addr = self.server_socket.accept()
            self.conn.settimeout(None) # Always wait forever in interactive mode
            if self.verbose: print(f"[Connection] Connection RE-ESTABLISHED at {addr}")
            return "" # Return empty string once to satisfy the call, but loop will resume correctly now that conn is valid again

    def close(self):
        """Clean teardown of network and subprocess."""
        if self.verbose: print("Closing Environment: Initiating graceful teardown...")
        
        # 1. Send the Poison Pill to Lua
        if self.conn:
            try:
                self.send_command("EXIT\n")
                time.sleep(0.5) # Give Lua a fraction of a second to process the command
            except (ConnectionResetError, BrokenPipeError):
                pass
            finally:
                self.conn.close()
            
        if self.server_socket:
            self.server_socket.close()
            
        # 2. Ensure the BizHawk process is actually dead
        if self.emulator_process:
            try:
                # Wait up to 3 seconds for BizHawk to close itself via client.exit()
                self.emulator_process.wait(timeout=3)
                if self.verbose: print("BizHawk closed successfully.")
            except subprocess.TimeoutExpired:
                # If it froze, execute a ruthless OS-level termination
                if self.verbose: print("BizHawk did not close in time. Terminating process...")
                self.emulator_process.terminate()
                try:
                    self.emulator_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    if self.verbose: print("Process still alive after terminate(). Forcing kill...")
                    self.emulator_process.kill()
                    self.emulator_process.wait()
        