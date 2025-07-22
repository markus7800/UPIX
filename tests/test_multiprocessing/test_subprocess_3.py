import subprocess
import sys
import time
from multiprocessing.connection import Client, Listener
import threading

s = """
import sys
from multiprocessing.connection import Client, Listener
address = int(sys.argv[1])
print(f"WORKER {address=}")

conn = Client(('localhost', address))

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

    # client_conn.send("Hello")
    # msg = listen_conn.recv()
    # print(msg)
    # client_conn.send("close")
    # listen_conn.close()

    p.wait()
    
def listen():
    listener = Listener(('localhost', address))
    conn = listener.accept()
    print('HOST connection accepted from', listener.last_accepted)
    
    conn.send("Hello")
    msg = conn.recv()
    print("HOST", msg)
    conn.send("close")
    conn.close()
    listener.close()
    
listen_thread = threading.Thread(target=listen)
listen_thread.start()

spawn_worker()


