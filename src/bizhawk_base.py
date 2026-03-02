import gymnasium as gym
import socket
import subprocess
import time

class BizHawkBaseEnv(gym.Env):
    """Universal Base Environment for BizHawk socket communication."""
    
    def __init__(self, bizhawk_path, rom_path, lua_path, host, port, reset_lua_path=None):
        super().__init__()
        self.bizhawk_path = bizhawk_path
        self.rom_path = rom_path
        self.lua_path = lua_path
        self.host = host
        self.port = port
        self.reset_lua_path = reset_lua_path
        
        self.server_socket = None
        self.conn = None
        self.emulator_process = None # Track the subprocess
        
        self._start_emulator_bridge()

    def _start_emulator_bridge(self):
        """Binds the socket and launches the emulator."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"Python ML Server actively listening on {self.host}:{self.port}...")
        
        print("Launching BizHawk as a subprocess...")
        self.emulator_process = subprocess.Popen([
            self.bizhawk_path, 
            self.rom_path, 
            f"--socket_ip={self.host}", 
            f"--socket_port={self.port}",
            f"--lua={self.lua_path}"
        ])
        
        self.conn, addr = self.server_socket.accept()
        print(f"Connection established with BizHawk at {addr}")
            

    def send_command(self, command: str):
        """Standardized protocol for sending a command to Lua."""
        try:
            formatted_reply = f"{len(command)} {command}"
            self.conn.sendall(formatted_reply.encode('utf-8'))
        except (ConnectionResetError, BrokenPipeError):
            pass # Socket is already dead

    def receive_payload(self) -> str:
        """Blocks and waits for the next payload from Lua."""
        try:
            return self.conn.recv(1024).decode('utf-8')
        except (ConnectionResetError, ConnectionAbortedError):
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
        