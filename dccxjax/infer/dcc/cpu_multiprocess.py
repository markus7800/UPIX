worker_script = """
import sys
import pickle
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.export
from typing import IO

import jax.numpy as jnp
jax.jit(jax.lax.linalg.cholesky).trace(jnp.eye(1)).lower()

def read_transport_layer(reader: IO[bytes]):
    msg = reader.readline().decode("utf8").rstrip()
    # print("worker read msg:", msg, file=sys.stderr)
    if msg == "":
        return None
    if msg == "close":
        return None
    obj = pickle.load(reader)
    return obj

def write_transport_layer(writer: IO[bytes], obj):
    # print("worker write ok:", file=sys.stderr)
    writer.write("OK\\r\\n".encode("utf8"))
    writer.flush()
    pickle.dump(obj, writer)
    writer.flush()
    
def write_error_transport_layer(writer: IO[bytes], obj):
    # print("write error:", file=sys.stderr)
    writer.write("ERROR\\r\\n".encode("utf8"))
    writer.flush()
    pickle.dump(obj, writer)
    writer.flush()

WORKER_ID = sys.argv[1]
# print(f"Starting worker ", WORKER_ID, "with pid", os.getpid(), file=sys.stderr)
while True:
    obj = read_transport_layer(sys.stdin.buffer)
    if obj is None:
        # print("Terminating worker.", file=sys.stderr)
        break
    try:
        jax_serialised_fn, args = obj 
        jax_fn = jax.export.deserialize(jax_serialised_fn)
        # args will be on CPU device, of course
        # print("worker", WORKER_ID, [arg.device for arg in args], file=sys.stderr)
        out = jax_fn.call(*args) # this will always compile
        # out will be on CPU device, of course
        # print("worker", WORKER_ID, out[0].device, file=sys.stderr)
        write_transport_layer(sys.stdout.buffer, out)
        del obj
        del out
        del jax_fn
    except Exception as e:
        print("exception", e, file=sys.stderr)
        write_error_transport_layer(sys.stdout.buffer, e)
"""

import jax
import pickle
from typing import IO
from queue import Queue
import time
import subprocess
import sys
from dccxjax.infer.dcc.dcc_types import JaxTask, ExportedJaxTask
from jax.tree_util import tree_flatten, tree_unflatten
from tqdm.auto import tqdm
from dccxjax.parallelisation import ParallelisationConfig, ParallelisationType, VectorisationType

__all__ = [
    
]
    

def read_transport_layer(reader: IO[bytes]):
    msg = reader.readline().decode("utf8").rstrip()
    # print("host read", msg)
    if msg != "OK":
        if msg != "":
            error = pickle.load(reader)
            raise Exception(error)
        else:
            raise Exception("Read empty response")
    obj = pickle.load(reader)
    return obj

def write_task_transport_layer(writer: IO[bytes], obj):
    # print("host write task")
    writer.write("task\r\n".encode("utf8"))
    writer.flush()
    pickle.dump(obj, writer)
    writer.flush()

def write_close_transport_layer(writer: IO[bytes]):
    # print("host write close")
    writer.write("close\r\n".encode("utf8"))
    writer.flush()



def start_worker_process(in_queue: Queue, out_queue: Queue, worker_id: int, pconfig: ParallelisationConfig):
    assert pconfig.parallelisation == ParallelisationType.MultiProcessingCPU
    # print("Start worker with id", worker_id)
    p = subprocess.Popen(
        (["taskset",  "-c", str(worker_id)] if pconfig.cpu_affinity else []) + [sys.executable,  "-c", worker_script, str(worker_id)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env = pconfig.environ
    )
    if pconfig.verbose:
        tqdm.write(f"Starting worker {worker_id} with pid {p.pid}" + (" (pinned)" if pconfig.cpu_affinity else ""))
        
    assert p.stdin is not None
    assert p.stdout is not None
    while True:
        try:
            task, task_aux = in_queue.get()
            assert isinstance(task, (JaxTask, ExportedJaxTask))
            exported_task = task.export() if isinstance(task, JaxTask) else task
            if pconfig.verbose:
                pre_info = exported_task.pre_info()
                if pre_info: tqdm.write(f"Worker {worker_id}: " + pre_info)
            
            t0 = time.monotonic()
            
            flat_args, _in_tree = tree_flatten(exported_task.args)
            assert _in_tree == exported_task.in_tree
                    
            write_task_transport_layer(p.stdin, (exported_task.exported_fn.serialize(), tuple(flat_args)))

            response = read_transport_layer(p.stdout)
            # print("got response:", response)
            # response will be copied to default device not host (cpu)!
            # tqdm.write(f"Response device {response[0].device}")
            
            result = tree_unflatten(exported_task.out_tree, response)
            
            elapsed_time = time.monotonic() - t0
        
            if pconfig.verbose:
                post_info = exported_task.post_info(result)
                if post_info: tqdm.write(f"Worker {worker_id}: " + post_info + f"\n    finished in {elapsed_time:.3f}s")

            out_queue.put((result, task_aux, elapsed_time))
            in_queue.task_done()
            del task
            del exported_task

        # except ShutDown:
        #     # By default, get() on a shut down queue will only raise once the queue is empty
        #     # print("shutdown")
        #     break
        except Exception as e:
            print("Worker error:", e)
            in_queue.task_done()
            
def start_worker_thread(in_queue: Queue, out_queue: Queue, worker_id: int, pconfig: ParallelisationConfig):
    assert pconfig.parallelisation == ParallelisationType.MultiThreadingJAXDevices
    thread_device = jax.devices()[worker_id]

    while True:
        task, task_aux = in_queue.get()
        assert isinstance(task, JaxTask)
        if pconfig.verbose:
            pre_info = task.pre_info()
            tqdm.write(f"Thread {worker_id}: starting task. " + pre_info)

        t0 = time.monotonic()
        device_args = jax.device_put(task.args, thread_device)
        device_results = task.f(*device_args)
        leaf = jax.tree.leaves(device_results)[0]
        assert leaf.device == thread_device
        results = jax.device_get(device_results)
        del device_results
        elapsed_time = time.monotonic() - t0
        if pconfig.verbose:
            post_info = task.post_info(results)
            if post_info: tqdm.write(f"Thread {worker_id}: " + post_info + f"\n    finished in {elapsed_time:.3f}s on device {thread_device}")
    
        out_queue.put((results, task_aux, elapsed_time))
        in_queue.task_done()

        del task
