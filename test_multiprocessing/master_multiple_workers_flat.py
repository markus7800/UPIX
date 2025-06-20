import subprocess
import sys
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.export
import jax.numpy as jnp
import pickle
import threading
from typing import IO, List, Callable, Tuple, NamedTuple
from queue import Queue, ShutDown
import time

@jax.jit
def step(carry, any):
  val, key = carry
  key1, key2 = jax.random.split(key)
  val = val + jax.random.normal(key1)
  return (val,key2), None
  
class A(NamedTuple):
    input: jax.Array

class B(NamedTuple):
    output: jax.Array

@jax.jit
def f(seed: A):
  return B(jax.lax.scan(step, (jnp.array(0.,float), seed.input), length=10**7)[0][0])


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
    assert isinstance(tree_store.val, Tuple)
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

    # TODO: move to `lower`
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
            # ERROR: cannot serialise
            # task = in_queue.get()
            # jitted_fn, args = task
            # assert isinstance(args, tuple)
            # exported_fn = jax.export.export(jitted_fn)(*args) # only traces
            # write_task_transport_layer(p.stdin, (exported_fn.serialize(), args))
            # response = read_transport_layer(p.stdout)
            # print("got response:", response)
            # out_queue.put(response)
            # in_queue.task_done()

            task = in_queue.get()
            fn, args = task
            assert isinstance(args, tuple)
            print(args)
            exported_fn, in_tree, out_tree = _export_flat(fn, None, (), None)(*args)
            flat_args, _in_tree = tree_flatten(args)
            assert _in_tree == in_tree
            print("flat args:", flat_args, in_tree)

            write_task_transport_layer(p.stdin, (exported_fn.serialize(), flat_args))
            flat_response = read_transport_layer(p.stdout)
            print("flat response:", flat_response)
            response = tree_unflatten(out_tree, flat_response)
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
        in_queue.put((f, (A(jax.random.PRNGKey(i)),)))
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