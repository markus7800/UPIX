import subprocess
import sys
import time
from multiprocessing.connection import Client, Listener
import threading

s = """
import sys
from multiprocessing.connection import Client, Listener
import time

address = int(sys.argv[1])
print(f"WORKER {address=}")

# conn = None
# while True:
#     try:
#         conn = Client(f"{address}.socket")
#         print("WORKER", conn)
#         break
#     except ConnectionRefusedError:
#         time.sleep(1)
# print("WORKER", "connected")

conn = Client(f"{address}.socket")
conn.send("Connected.")

while True:
    msg = conn.recv()
    print("WORKER", msg)
    conn.send({msg: "=)"})
    if msg == 'close':
        conn.close()
        break
"""
# print(sys.argv)
address = 6000

def spawn_worker():
    p = subprocess.Popen(
        [sys.executable, "-c", s, str(address)],
    )
    # time.sleep(5)
    listener = Listener(f"{address}.socket")
    conn = listener.accept()
    print('HOST connection accepted from', listener.last_accepted)
    msg = conn.recv()
    print("HOST", msg)
    
    conn.send("Hello")
    msg = conn.recv()
    print("HOST", msg)
    conn.send("close")
    listener.close()

    p.wait()

spawn_worker()

