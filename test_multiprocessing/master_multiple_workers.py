from io import BufferedReader, BufferedWriter
import subprocess
import sys
import jax
import jax.export
import jax.numpy as jnp
import pickle
import threading
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


def read_transport_layer(reader: BufferedReader):
    msg = reader.readline().decode("utf8").rstrip()
    print(msg)
    if msg != "OK":
        raise Exception
    obj = pickle.load(reader)
    return obj

def write_task_transport_layer(writer: BufferedWriter, obj):
    writer.write("task\r\n".encode("utf8"))
    writer.flush()
    pickle.dump(obj, writer)
    writer.flush()

def write_close_transport_layer(writer: BufferedWriter):
    writer.write("close\r\n".encode("utf8"))
    writer.flush()

def worker(in_queue: Queue, out_queue: Queue):
    p = subprocess.Popen(
        [sys.executable,  "test_multiprocessing/worker.py"],
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
            write_task_transport_layer(p.stdin, (exported_fn.serialize(), args)) # type: ignore
            response = read_transport_layer(p.stdout) # type: ignore
            print("got response:", response)
            in_queue.task_done()
            out_queue.put(response)
        except ShutDown:
            # By default, get() on a shut down queue will only raise once the queue is empty
            write_close_transport_layer(p.stdin) # type: ignore
            break
    p.wait()
    
def main():
    t0 = time.time()
    in_queue = Queue()
    out_queue = Queue()

    num_threads = 5
    num_tasks = 5
    
    for _ in range(num_threads):
        t = threading.Thread(target=worker, args=(in_queue, out_queue), daemon=True)
        t.start()
    
    for i in range(num_tasks):
        in_queue.put((f, (jax.random.PRNGKey(i),)))
    
    in_queue.join()
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