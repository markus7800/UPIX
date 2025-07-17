
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
    print("worker read msg:", msg, file=sys.stderr)
    if msg == "":
        return None
    if msg == "close":
        return None
    obj = pickle.load(reader)
    return obj

def write_transport_layer(writer: IO[bytes], obj):
    print("worker write ok:", file=sys.stderr)
    writer.write("OK\\r\\n".encode("utf8"))
    writer.flush()
    pickle.dump(obj, writer)
    writer.flush()
    
def write_error_transport_layer(writer: IO[bytes], obj):
    print("write error:", file=sys.stderr)
    writer.write("ERROR\\r\\n".encode("utf8"))
    writer.flush()
    pickle.dump(obj, writer)
    writer.flush()

WORKER_ID = sys.argv[1]
print(f"Starting worker ", WORKER_ID, "with", os.getpid(), file=sys.stderr)
while True:
    print("Worker loop.", file=sys.stderr)
    obj = read_transport_layer(sys.stdin.buffer)
    if obj is None:
        print("Terminating worker.", file=sys.stderr)
        break
    try:
        jax_serialised_fn, args = obj 
        print("in", args, file=sys.stderr)
        jax_fn = jax.export.deserialize(jax_serialised_fn)
        print("jax_fn", jax_fn, file=sys.stderr)
        out = jax_fn.call(*args) # this will always compile
        print("out", out, file=sys.stderr)
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
from typing import IO, List, Callable, Tuple
from queue import Queue, ShutDown
import time
import subprocess
import sys
from dccxjax.infer.dcc.dcc_types import InferenceResult, InferenceTask, LogWeightEstimate
import jax.flatten_util



def read_transport_layer(reader: IO[bytes]):
    msg = reader.readline().decode("utf8").rstrip()
    print("host read", msg)
    if msg != "OK":
        if msg != "":
            error = pickle.load(reader)
            raise Exception(error)
        else:
            raise Exception("Read empty response")
    obj = pickle.load(reader)
    return obj

def write_task_transport_layer(writer: IO[bytes], obj):
    print("host write task")
    writer.write("task\r\n".encode("utf8"))
    writer.flush()
    pickle.dump(obj, writer)
    writer.flush()

def write_close_transport_layer(writer: IO[bytes]):
    print("host write close")
    writer.write("close\r\n".encode("utf8"))
    writer.flush()


from jax.tree_util import tree_flatten, tree_unflatten
from jax._src.linear_util import Store

from jax.stages import Traced, Wrapped
from typing import Sequence, Any
from jax._src.export._export import DisabledSafetyCheck, Exported, default_export_platform, check_symbolic_scope_errors, _export_lowered
import jax._src.interpreters.mlir as mlir
import jax._src.config as jax_config

def trace_to_flat(f: Callable, args: Tuple) -> Tuple[Traced, Any,Any]:
    args_flat, in_tree = tree_flatten(args)

    tree_store = Store()
    @jax.jit
    def f_flat(*_args_flat):
        _args = tree_unflatten(in_tree, _args_flat)
        out = f(*_args)
        out_flat, out_tree = tree_flatten(out)
        tree_store.store((in_tree, out_tree))
        return out_flat
    
    traced = f_flat.trace(*args_flat)
    # print(traced.jaxpr)
    assert isinstance(tree_store.val, Tuple)

    check_symbolic_scope_errors(f_flat, args_flat, {})

    return traced, *tree_store.val


def _export_flat(
    fun: Callable,
    platforms: Sequence[str] | None = None,
    disabled_checks: Sequence[DisabledSafetyCheck] = (),
    _device_assignment_for_internal_jax2tf_use_only=None,
    override_lowering_rules=None,
    ) -> Callable[..., Tuple[Exported,Any,Any]]:
  
  
  def do_export(*args_specs) -> Tuple[Exported, Any,Any]:
    if platforms is not None:
      actual_lowering_platforms = tuple(platforms)
    else:
      actual_lowering_platforms = (default_export_platform(),)

    # check_symbolic_scope_errors(fun_jit, args_specs, kwargs_specs)

    traced, in_tree, out_tree = trace_to_flat(fun, args_specs)


    lowered = traced.lower(
        lowering_platforms=actual_lowering_platforms,
        _private_parameters=mlir.LoweringParameters(
            override_lowering_rules=override_lowering_rules,
            for_export=True,
            export_ignore_forward_compatibility=jax_config.export_ignore_forward_compatibility.value))
    export_lowered = _export_lowered(
        lowered, traced.jaxpr, traced.fun_name,
        disabled_checks=disabled_checks,
        _device_assignment_for_internal_jax2tf_use_only=_device_assignment_for_internal_jax2tf_use_only)
    return export_lowered, in_tree, out_tree
  return do_export

def process_worker(in_queue: Queue, out_queue: Queue, worker_id: int, pin: int | None):
    print("Start worker with pin", pin)
    p = subprocess.Popen(
        (["taskset",  "-c", str(pin)] if pin is not None else []) + [sys.executable,  "-c", worker_script, str(worker_id)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE
    )
    assert p.stdin is not None
    assert p.stdout is not None
    while True:
        try:
            work_aux, work = in_queue.get()

            with open("tmp.pickle", "wb") as f:
                pickle.dump(work, f)
                
            write_task_transport_layer(p.stdin, work)

            response = read_transport_layer(p.stdout)
            print("got response:", response)

            out_queue.put((work_aux, response))
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