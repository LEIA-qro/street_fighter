import socket
import subprocess
import random

import os, sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_TESTING_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(CODE_TESTING_DIR)
sys.path.append(PROJECT_ROOT)

# Ensure to place the project folder inside the BizHawk directory for correct relative paths
BIZHAWK_FOLDER_DIR = os.path.dirname(PROJECT_ROOT) 

BIZHAWK_PATH = os.path.join(BIZHAWK_FOLDER_DIR, "EmuHawk.exe")
LUA_SCRIPT_PATH = os.path.join(CURRENT_DIR, "test_telemetry_v1.lua")
ROM_PATH = os.path.join(PROJECT_ROOT, "roms", "Street Fighter II' - Special Champion Edition (USA).md")

# Check if BizHawk executable exists at the specified path
if not os.path.exists(BIZHAWK_PATH):
    print(f"ERROR: BizHawk executable not found at {BIZHAWK_PATH}. Please check the path and try again.")
    sys.exit(1)



HOST = '127.0.0.1'
PORT = 9999

def test_telemetry():
    # 1. Python MUST act as the Server. We bind and listen FIRST.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        # Allow immediate reuse of the port to prevent "Address already in use" errors
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print(f"Python ML Server actively listening on {HOST}:{PORT}...")

        # 2. Launch BizHawk (which acts as the Client)
        print("Launching BizHawk as a subprocess...")
        subprocess.Popen([
            BIZHAWK_PATH,
            ROM_PATH, 
            f"--socket_ip={HOST}", 
            f"--socket_port={PORT}",
            f"--lua={LUA_SCRIPT_PATH}"
            ])

        # 3. Block and wait for BizHawk's internal engine to connect
        print("Waiting for BizHawk to establish connection...")
        print("ACTION REQUIRED: Load ROM -> Open Lua Console -> Run Script")
        
        conn, addr = server_socket.accept()
        
        with conn:
            print(f"SUCCESS: BizHawk connected from {addr}!")

            # Debugging: Initialize frame count
            frame_count = 0
            
            while True:
                try:
                    # 1. Receive RAM state from Lua
                    data = conn.recv(1024).decode('utf-8')
                    if data:

                        #print(f"RAM State: {data.strip()}")
                        # data looks like "30 176,65535,380..."
                        # We split strictly on the first space to isolate the CSV payload
                        parts = data.split(" ", 1)

                        if len(parts) == 2:
                            payload = parts[1]

                            # Dedugging
                            if frame_count % 60 == 0:
                                print(f"Clean Payload: {payload}")
                                # Now you can safely split by comma to get your integers
                                # ram_values = [int(x) for x in payload.split(',')]


                        
                        # 2. Send dummy action (12 buttons, all zeros)
                        # 1. Generate a 12-character string of random 1s and 0s
                        random_action_array = [str(random.choice([0, 1])) for _ in range(12)]
                        action_string = "".join(random_action_array) + "\n"
                        reply = action_string
                        
                        # 3. BizHawk comm protocol: Prepend length of the reply
                        formatted_reply = f"{len(reply)} {reply}"
                        conn.sendall(formatted_reply.encode('utf-8'))

                        # Deugging
                        frame_count += 1
                        
                except ConnectionResetError:
                    print("BizHawk closed the connection.")
                    break

if __name__ == "__main__":
    test_telemetry()