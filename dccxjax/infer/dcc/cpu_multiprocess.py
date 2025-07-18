
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
print(f"Starting worker ", WORKER_ID, "with", os.getpid(), file=sys.stderr)
while True:
    obj = read_transport_layer(sys.stdin.buffer)
    if obj is None:
        # print("Terminating worker.", file=sys.stderr)
        break
    try:
        jax_serialised_fn, args = obj 
        jax_fn = jax.export.deserialize(jax_serialised_fn)
        out = jax_fn.call(*args) # this will always compile
        write_transport_layer(sys.stdout.buffer, out)
        del obj
        del out
        del jax_fn
    except Exception as e:
        print("exception", e, file=sys.stderr)
        write_error_transport_layer(sys.stdout.buffer, e)
"""

import jax
import jax.export
import jax.numpy as jnp
import pickle
import threading
from typing import IO, List, Callable, Tuple, Dict, Generic, TypeVar
from queue import Queue
import time
import subprocess
import sys
from dccxjax.infer.dcc.dcc_types import JaxTask, ExportedJaxTask
from dataclasses import dataclass, field
from enum import Enum
import os
from jax.tree_util import tree_flatten, tree_unflatten
from tqdm.auto import tqdm


__all__ = [
    "ParallelisationType",
    "ParallelisationConfig"
]
    
class ParallelisationType(Enum):
    Sequential = 0
    MultiProcessingCPU = 1
    MultiThreadingJAXDevices = 2

@dataclass
class ParallelisationConfig:
    type: ParallelisationType = ParallelisationType.Sequential
    num_workers: int = os.cpu_count() or 1
    cpu_affinity: bool = False
    environ: Dict[str, str] = field(default_factory= lambda: {
        "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "JAX_PLATFORMS": "cpu"
    })
    verbose: bool = True
    

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



def process_worker(in_queue: Queue, out_queue: Queue, worker_id: int, config: ParallelisationConfig):
    # print("Start worker with id", worker_id)
    p = subprocess.Popen(
        (["taskset",  "-c", str(worker_id)] if config.cpu_affinity else []) + [sys.executable,  "-c", worker_script, str(worker_id)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env = config.environ
    )
    assert p.stdin is not None
    assert p.stdout is not None
    while True:
        try:
            task, task_aux = in_queue.get()
            assert isinstance(task, (JaxTask, ExportedJaxTask))
            exported_task = task.export() if isinstance(task, JaxTask) else task
            if config.verbose:
                pre_info = exported_task.pre_info()
                if pre_info: tqdm.write(f"Worker {worker_id}: " + pre_info)
                
            flat_args, _in_tree = tree_flatten(exported_task.args)
            assert _in_tree == exported_task.in_tree
                    
            write_task_transport_layer(p.stdin, (exported_task.exported_fn.serialize(), tuple(flat_args)))

            response = read_transport_layer(p.stdout)
            # print("got response:", response)
            
            result = tree_unflatten(exported_task.out_tree, response)
            if config.verbose:
                post_info = exported_task.post_info(result)
                if post_info: tqdm.write(f"Worker {worker_id}: " + post_info)

            out_queue.put((result, task_aux))
            in_queue.task_done()

        # except ShutDown:
        #     # By default, get() on a shut down queue will only raise once the queue is empty
        #     # print("shutdown")
        #     break
        except Exception as e:
            print("Worker error:", e)
            in_queue.task_done()
    write_close_transport_layer(p.stdin)
    p.wait()