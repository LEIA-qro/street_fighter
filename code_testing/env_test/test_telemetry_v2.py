import socket
import subprocess
import random

import os, sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_TESTING_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(CODE_TESTING_DIR)
sys.path.append(PROJECT_ROOT)

# Ensure to place the project folder inside the BizHawk directory for correct relative paths
# BIZHAWK_FOLDER_DIR = os.path.dirname(PROJECT_ROOT) 

import src.config as config 



EXPECTED_DIMS = config.ACTION_DIM + 2  # ← FIX #1: The correct number of CSV values from Lua

def recv_line(conn: socket.socket) -> str:
    # Data may arrive in chunks, so we need to buffer until we get a full line (ending with '\n').
    data = b""
    while True:
        chunk = conn.recv(1)
        if not chunk or chunk == b'\n':
            break
        data += chunk
    return data.decode('utf-8')

def test_telemetry():
    # 1. Python MUST act as the Server. We bind and listen FIRST.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        # Allow immediate reuse of the port to prevent "Address already in use" errors
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((config.HOST, config.PORT))
        server_socket.listen(1)
        print(f"Python ML Server actively listening on {config.HOST}:{config.PORT}...")

        # 2. Launch BizHawk (which acts as the Client)
        print("Launching BizHawk as a subprocess...")
        # Pass the ROM path as the FIRST argument after the executable
        subprocess.Popen([
            config.BIZHAWK_PATH, 
            config.ROM_PATH, 
            f"--socket_ip={config.HOST}", 
            f"--socket_port={config.PORT}",
            f"--lua={config.MATCH_TEST_ENV_CLIENT_LUA_PATH}" # Optional: Automatically load and run the Lua script
        ])

        # 3. Block and wait for BizHawk's internal engine to connect
        print("Waiting for BizHawk to establish connection...")
        
        conn, addr = server_socket.accept()
        
        with conn:
            print(f"SUCCESS: BizHawk connected from {addr}!")

            # Debugging: Initialize step count
            step_count = 0
            
            while True:
                try:
                    # 1. Receive RAM state from Lua
                    raw_line = recv_line(conn)
                    if not raw_line:
                        continue
                    csv_string = raw_line.strip().split(" ")[-1]
                    parts = csv_string.split(",")

                    if len(parts) == EXPECTED_DIMS:
                        # Now you can safely split by comma to get your integers
                        ram_values = [int(x) for x in parts]

                        p1_hp, p2_hp = ram_values[0], ram_values[1]
                        p1_x, p2_x = ram_values[2], ram_values[3]
                        p1_y, p2_y = ram_values[4], ram_values[5]
                        p1_act, p2_act = ram_values[6], ram_values[7]
                        p1_proj, p2_proj = ram_values[8], ram_values[9]
                        p1_char, p2_char = ram_values[10], ram_values[11]

                        p1_hp = 0 if p1_hp >= 65535 else p1_hp
                        p2_hp = 0 if p2_hp >= 65535 else p2_hp

                        if p1_hp <= 0 or p2_hp <= 0:
                            print(f"Match Over at Step {step_count}! P1 HP: {p1_hp}, P2 HP: {p2_hp}.")
                        
                        action_string = "".join('1' if random.random() > 0.5 else '0' for _ in range(config.ACTION_DIM)) # Generate a random action string of length ACTION_DIM (10)
                        reply = action_string + "\n"

                        # Dedugging
                        if step_count % 240 == 0: # Print eevery 16 seconds
                            print(f"[Step {step_count}] State: HP={p1_hp}/{p2_hp} | Action: {reply.strip()}")               
                    
                        # 3. BizHawk comm protocol: Prepend length of the reply
                        formatted_reply = f"{len(reply)} {reply}"
                        conn.sendall(formatted_reply.encode('utf-8'))

                            # Deugging
                        step_count += 1
                    else: print(f"[WARN] Malformed packet. Expected {EXPECTED_DIMS} dims, got {len(parts)}. Raw: '{raw_line}'") 
                        
                except (ConnectionResetError, BrokenPipeError):
                    print("BizHawk closed the connection.")
                    break
                except ValueError as e:
                    print(f"[ERROR] Failed to parse int from packet: '{raw_line}' | {e}")

if __name__ == "__main__":
    test_telemetry()