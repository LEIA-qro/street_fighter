import gymnasium as gym
import socket
import subprocess
import time

class BizHawkBaseEnv(gym.Env):
    """Universal Base Environment for BizHawk socket communication."""
    
    def __init__(self, bizhawk_path, rom_path, lua_path, host, port, trainable=True):
        super().__init__()
        self.bizhawk_path = bizhawk_path
        self.rom_path = rom_path
        self.lua_path = lua_path
        self.host = host
        self.port = port
        self.trainable = trainable
        
        self.server_socket = None
        self.conn = None
        self.emulator_process = None # Track the subprocess

        # NEW: The TCP Holding Tank
        self.stream_buffer = ""
        
        self._start_emulator_bridge()

    def _start_emulator_bridge(self):
        """Binds the socket and launches the emulator."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"Python ML Server actively listening on {self.host}:{self.port}...")
        
        print("Launching BizHawk as a subprocess...")

        # Base arguments
        launch_args = [
            self.bizhawk_path, 
            self.rom_path, 
            f"--socket_ip={self.host}", 
            f"--socket_port={self.port}"
        ]

        # Only auto-inject the Lua script if we are training
        if self.trainable and self.lua_path:
            print(f"Auto-loading training Lua script: {self.lua_path}")
            launch_args.append(f"--lua={self.lua_path}")
            
        self.emulator_process = subprocess.Popen(launch_args)
        
        if not self.trainable:
            print("\n[INTERACTIVE MODE] BizHawk launched.")
            print("1. Navigate the game menus manually.")
            print("2. When the match is ready, open the BizHawk Lua Console. Tools → Lua Console")
            print(f"3. Run the script: {self.lua_path}. Script → Open Script → Select {self.lua_path}")
            print(f"\n[Connection] Waiting for your Lua connection...")
            
        self.conn, addr = self.server_socket.accept()
        
        # CONDITIONAL TIMEOUT: Strict failsafe for training, Infinite patience for testing
        if self.trainable:
            self.conn.settimeout(180.0) 
        else:
            self.conn.settimeout(None) # Wait forever while human navigates menus
            
        print(f"[Connection] Connection established with BizHawk at {addr}")
            

    def send_command(self, command: str):
        """Standardized protocol for sending a command to Lua."""
        try:
            formatted_reply = f"{len(command)} {command}"
            self.conn.sendall(formatted_reply.encode('utf-8'))
        except (ConnectionResetError, BrokenPipeError) as e:
            print(f"[WARN] send_command failed in interactive mode: {e}")
            if self.trainable:
                raise RuntimeError(f"Socket broken during training: {e}")
            # Non-trainable (interactive) mode: log and continue
            else:
                print(f"\n[Connection] Waiting for your Lua connection...")


    def receive_payload(self) -> str:
        """Blocks and waits for a complete, mathematically perfect payload."""
        try:
            # 1. Keep receiving bytes until we see a newline
            while '\n' not in self.stream_buffer:
                chunk = self.conn.recv(4096).decode('utf-8')
                if not chunk:
                    print("\n[Connection] No response from BizHawk.")
                    return ""
                self.stream_buffer += chunk
            
            # 2. Slice the buffer precisely at the first newline.
            line, self.stream_buffer = self.stream_buffer.split('\n', 1)
            
            return line
            
        except socket.timeout:
            print("\n[FAILSAFE] Python timed out waiting for BizHawk. Forcing crash...")
            raise RuntimeError("BizHawk Socket Timeout")
            
        except (ConnectionResetError, ConnectionAbortedError):
            print("\n[FAILSAFE] No Connection from BizHawk.")
            return ""  

    def close(self):
        """Clean teardown of network and subprocess."""
        print("Closing Environment: Initiating graceful teardown...")
        
        # 1. Send the Poison Pill to Lua
        if self.conn:
            self.send_command("EXIT\n")
            time.sleep(0.5) # Give Lua a fraction of a second to process the command
            self.conn.close()
            
        if self.server_socket:
            self.server_socket.close()
            
        # 2. Ensure the BizHawk process is actually dead
        if self.emulator_process:
            try:
                # Wait up to 3 seconds for BizHawk to close itself via client.exit()
                self.emulator_process.wait(timeout=3)
                print("BizHawk closed successfully.")
            except subprocess.TimeoutExpired:
                # If it froze, execute a ruthless OS-level termination
                print("BizHawk did not close in time. Terminating process...")
                self.emulator_process.terminate()
        