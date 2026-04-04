import socket
import subprocess
import random

BIZHAWK_PATH = r"C:\\Users\Diego Perea\Documents\Apps\BizHawk-2.8-win-x64\\EmuHawk.exe"
HOST = '127.0.0.1'
PORT = 9999

def test_telemetry2():
    # 1. Python MUST act as the Server. We bind and listen FIRST.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        # Allow immediate reuse of the port to prevent "Address already in use" errors
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print(f"Python ML Server actively listening on {HOST}:{PORT}...")

        # 2. Launch BizHawk (which acts as the Client)
        print("Launching BizHawk as a subprocess...")
        subprocess.Popen([BIZHAWK_PATH, f"--socket_ip={HOST}", f"--socket_port={PORT}"])

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
                            ram_values = [int(x) for x in payload.split(',')]
                        
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
    test_telemetry2()