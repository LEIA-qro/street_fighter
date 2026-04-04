import socket
import subprocess

# Configuration (Ensure this path is perfectly correct)
BIZHAWK_PATH = r"C:\\Users\Diego Perea\Documents\Apps\BizHawk-2.8-win-x64\\EmuHawk.exe"
HOST = '127.0.0.1'
PORT = 9999

def start_pipeline():
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
        conn, addr = server_socket.accept()
        
        with conn:
            print(f"SUCCESS: BizHawk connected from {addr}!")
            
            while True:
                try:
                    # 4. Listen for Lua's PING
                    data = conn.recv(1024).decode('utf-8')
                    if data:
                        print(f"BizHawk Lua says: {data.strip()}")
                        
                        # 5. Send PONG back (Prefixing length for BizHawk's strict protocol)
                        reply = "PONG\n"
                        formatted_reply = f"{len(reply)} {reply}"
                        conn.sendall(formatted_reply.encode('utf-8'))
                except ConnectionResetError:
                    print("BizHawk closed the connection.")
                    break

if __name__ == "__main__":
    start_pipeline()