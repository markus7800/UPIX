import subprocess
import sys
import time
from multiprocessing.connection import Client, Listener
import threading

s = """
import sys
from multiprocessing.connection import Client, Listener
listen_address = int(sys.argv[1])
client_address = int(sys.argv[2])
print(f"WORKER {listen_address=} {client_address=}")

client_conn = Client(('localhost', client_address))

listener = Listener(('localhost', listen_address))
listen_conn = listener.accept()
print('WORKER connection accepted from', listener.last_accepted)


while True:
    msg = listen_conn.recv()
    print("WORKER", msg)
    client_conn.send({msg: "=)"})
    if msg == 'close':
        listen_conn.close()
        break
listener.close()
"""
# print(sys.argv)
listen_address = 6000
client_address = 6001

def spawn_worker():
    p = subprocess.Popen(
        [sys.executable, "-c", s, str(client_address), str(listen_address)],
    )

    # client_conn.send("Hello")
    # msg = listen_conn.recv()
    # print(msg)
    # client_conn.send("close")
    # listen_conn.close()

    p.wait()
    
def listen():
    listener = Listener(('localhost', listen_address))
    listen_conn = listener.accept()
    print('HOST connection accepted from', listener.last_accepted)
    
    client_conn = Client(('localhost', client_address))
    client_conn.send("Hello")
    msg = listen_conn.recv()
    print("HOST", msg)
    client_conn.send("close")
    listen_conn.close()
    
listen_thread = threading.Thread(target=listen)
listen_thread.start()

spawn_worker()


