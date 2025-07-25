import subprocess
import sys
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.export
import jax.numpy as jnp
import pickle
import threading
from typing import IO, List
from queue import Queue, ShutDown
import time

@jax.jit
def step(carry, any):
  val, key = carry
  key1, key2 = jax.random.split(key)
  val = val + jax.random.normal(key1)
  return (val,key2), None
  
@jax.jit
def f(seed):
  return jax.lax.scan(step, (jnp.array(0.,float), seed), length=10**7)[0][0]


def read_transport_layer(reader: IO[bytes]):
    msg = reader.readline().decode("utf8").rstrip()
    print(msg)
    if msg != "OK":
        error = pickle.load(reader)
        raise Exception(error)
    obj = pickle.load(reader)
    return obj

def write_task_transport_layer(writer: IO[bytes], obj):
    writer.write("task\r\n".encode("utf8"))
    writer.flush()
    pickle.dump(obj, writer)
    writer.flush()

def write_close_transport_layer(writer: IO[bytes]):
    writer.write("close\r\n".encode("utf8"))
    writer.flush()


def worker(in_queue: Queue, out_queue: Queue, pin: int | None):
    p = subprocess.Popen(
        (["taskset",  "-c", str(pin)] if pin is not None else []) + [sys.executable,  "test_multiprocessing/worker.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE
    )
    assert p.stdin is not None
    assert p.stdout is not None
    while True:
        try:
            task = in_queue.get()
            jitted_fn, args = task
            assert isinstance(args, tuple)
            exported_fn = jax.export.export(jitted_fn)(*args) # only traces
            write_task_transport_layer(p.stdin, (exported_fn.serialize(), args))
            response = read_transport_layer(p.stdout)
            print("got response:", response)
            out_queue.put(response)
            in_queue.task_done()
        except ShutDown:
            # By default, get() on a shut down queue will only raise once the queue is empty
            print("shutdown")
            break
        except Exception as e:
            print("Worker error:", e)
            in_queue.task_done()

    write_close_transport_layer(p.stdin)
    p.wait()
    
def main():
    t0 = time.time()
    in_queue = Queue()
    out_queue = Queue()

    num_threads = 1
    num_tasks = 2
    
    threads: List[threading.Thread] = []
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(in_queue, out_queue, i), daemon=True)
        t.start()
        threads.append(t)
    
    for i in range(num_tasks):
        in_queue.put((f, (jax.random.key(i),)))
    in_queue.shutdown()
    
    in_queue.join()
    for t in threads:
        t.join()
    t1 = time.time()
    print(f"Finished in {t1-t0:.3f}s.")
    
    out_queue.shutdown()
    results = []
    while True:
        try:
            results.append(out_queue.get_nowait())
        except ShutDown:
            break
    print("result:", results)
        
main()