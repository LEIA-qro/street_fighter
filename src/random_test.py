import socket
import subprocess
import random
import os
import config # Imports your new config.py file

def random_test_telemetry():
    # 1. Python MUST act as the Server. We bind and listen FIRST.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        # Allow immediate reuse of the port to prevent "Address already in use" errors
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((config.HOST, config.PORT))
        server_socket.listen(1)
        print(f"Python ML Server actively listening on {config.HOST}:{config.PORT}...")

        # 2. Launch BizHawk (which acts as the Client)
        print("Launching BizHawk as a subprocess...")
        # Pass the ROM path as the FIRST argument after the executable, along with socket and Lua script parameters
        subprocess.Popen([
            config.BIZHAWK_PATH, 
            config.ROM_PATH, 
            f"--socket_ip={config.HOST}", 
            f"--socket_port={config.PORT}",
            f"--lua={config.ENV_CLIENT_LUA_SCRIPT_PATH}"
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
                    data = conn.recv(1024).decode('utf-8')
                    if data:

                        #print(f"RAM State: {data.strip()}")
                        # data looks like "30 176,65535,380..."
                        # We split strictly on the first space to isolate the CSV payload
                        parts = data.split(" ", 1)

                        if len(parts) == 2:
                            payload = parts[1]
                            # Now you can safely split by comma to get your integers
                            ram_values = [int(x) for x in payload.split(',')]

                            p1_hp, p2_hp, p1_x, p2_x, p1_y, p2_y, p1_action_id, p2_action_id, p1_proj_x, p2_proj_x= ram_values

                            p1_hp = 0 if p1_hp >= 65535 else p1_hp
                            p2_hp = 0 if p2_hp >= 65535 else p2_hp

                            if p1_hp <= 0 or p2_hp <= 0:
                                # Randomly select a new savestate from the AVAILABLE_STATES list in config.py
                                chosen_state_file = random.choice(config.AVAILABLE_STATES)
                                # 2. Construct the FULL absolute path using os.path.join
                                full_state_path = os.path.join(config.STATES_DIR, chosen_state_file)
                                print(f"Match Over at Step {step_count}! P1 HP: {p1_hp}, P2 HP: {p2_hp}. Sending RESET.")
                                print(f"Loading new state: {chosen_state_file}")
                                # Append the state name to the RESET command
                                reply = f"RESET {full_state_path}\n"
                            else:

                                # Generate a 10-character string of random 1s and 0s
                                # In production, this will be replaced by your neural network's output
                                action_string = "".join('1' if random.random() > 0.5 else '0' for _ in range(config.ACTION_DIM))
                                reply = action_string + "\n"

                            # Dedugging
                            if step_count % 240 == 0: # Print eevery 16 seconds
                                print(f"BizHawk Connected and Received Clean Payload: {payload}")                
                        
                            # 3. BizHawk comm protocol: Prepend length of the reply
                            formatted_reply = f"{len(reply)} {reply}"
                            conn.sendall(formatted_reply.encode('utf-8'))

                             # Deugging
                            step_count += 1
                        
                except ConnectionResetError:
                    print("BizHawk closed the connection.")
                    print("Resetting configurations to default")
                    subprocess.Popen([
                        config.BIZHAWK_PATH, 
                        f"--lua={config.RESET_CONFIG_LUA_SCRIPT_PATH}"
                    ])
                    break

if __name__ == "__main__":
    random_test_telemetry()