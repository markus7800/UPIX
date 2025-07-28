
import logging
import jax
import contextlib
from typing import Sequence, TypeVar, List, Optional, Callable, Dict, Any
from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm.auto import tqdm
import os
from jax._src.config import trace_context

__all__ = [
    "bcolors",
    "get_backend",
    "get_default_device",
    "setup_logging",
    "track_compilation_time",
    "CompilationTimeTracker",
    "broadcast_jaxtree",
    "timed"
]

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

import jax.extend.backend
def get_backend():
    return jax.extend.backend.get_backend()
def get_default_device():
    return jax.extend.backend.get_default_device()

logger = logging.getLogger("dccxjax")

def setup_logging(level: int | str):
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s: %(message)s'))
    logger.addHandler(handler)

class JitVariationTracker:
    def __init__(self, name: str) -> None:
        self.name = name
        self.variations: Dict[str, List[str]] = dict()
    def add_variation(self, axis_env: str, variation: str):
        if axis_env not in self.variations:
            self.variations[axis_env] = []
        self.variations[axis_env].append(variation)
    def has_variation(self, axis_env: str):
        if axis_env not in self.variations:
            return False
        return len(self.variations[axis_env]) > 0
    def get_variations(self, axis_env: str) -> List[str]:
        if axis_env not in self.variations:
            return []
        return self.variations[axis_env]


def maybe_jit_warning(tracker: JitVariationTracker, input: Any):
    axis_env_str = str(trace_context()[0])
    input_str = get_dtype_shape_str_of_tree(input)
    msg = f"Compile {tracker.name} and for input: {input_str} and axis-env: {axis_env_str}"
    with logging_redirect_tqdm(loggers=[logger]):
        if tracker.has_variation(axis_env_str):
            if logger.level <= logging.DEBUG:
                tabs = " " * (len(msg) - len(input) + len("dccxjax - WARNING: ") - 9)
                msg += "".join([f"\n{tabs}prev-input: {prev_input}" for prev_input in tracker.get_variations(axis_env_str)])
            logger.warning("Re-" + msg)
        else:
            logger.debug(msg)
    tracker.add_variation(axis_env_str, input_str)

def get_dtype_shape_str_of_tree(tree):
    def _dtype_shape(v):
        return str(jax.typeof(v))
    s = repr(jax.tree.map(_dtype_shape, tree))
    return s.replace("'", "")

JAX_TREE = TypeVar("JAX_TREE")
def broadcast_jaxtree(tree: JAX_TREE, sizes: Sequence[int]) -> JAX_TREE:
    return jax.tree.map(lambda v: jax.lax.broadcast(v, sizes), tree)

from jax._src.monitoring import EventDurationListenerWithMetadata, _unregister_event_duration_listener_by_callback
from jax._src.dispatch import JAXPR_TRACE_EVENT, JAXPR_TO_MLIR_MODULE_EVENT, BACKEND_COMPILE_EVENT


# _create_pjit_jaxpr logs JAXPR_TRACE_EVENT and calls to
# -> trace_to_jaxpr_dynamic

# stage_parallel_callable logs JAXPR_TRACE_EVENT and calls to
# -> trace_to_jaxpr_dynamic

# lower_parallel_callable logs JAXPR_TO_MLIR_MODULE_EVENT and calls to
# -> lower_jaxpr_to_module

# _cached_lowering_to_hlo logs JAXPR_TO_MLIR_MODULE_EVENT and calls to
# -> lower_jaxpr_to_module

# UnlaodedPmapExecutable.from_hlo logs BACKEND_COMPILE_EVENT and calls to 
# -> compiler.compile_or_get_cached

# _cached_compilation logs BACKEND_COMPILE_EVENT and calls to 
# -> compiler.compile_or_get_cached

class CompilationTimeTracker(EventDurationListenerWithMetadata):
    def __init__(self) -> None:
        self.trace_time = 0.
        self.lower_time = 0.
        self.compile_time = 0.
    def get_trace_time_secs(self):
        return self.trace_time
    def get_lower_time_secs(self):
        return self.lower_time
    def get_compile_time_secs(self):
        return self.compile_time
    def get_total_time_secs(self):
        return self.trace_time + self.lower_time + self.compile_time
    def __call__(self, event: str, duration_secs: float,
               **kwargs: str | int) -> None:
        if event == JAXPR_TRACE_EVENT:
            self.trace_time += duration_secs
        if event == JAXPR_TO_MLIR_MODULE_EVENT:
            self.lower_time += duration_secs
        if event == BACKEND_COMPILE_EVENT:
            # fun_name = kwargs["fun_name"]
            # tqdm.write(f"Compiled {fun_name} in {duration_secs:.3f}s")
            self.compile_time += duration_secs

@contextlib.contextmanager
def track_compilation_time():
    tracker = CompilationTimeTracker()
    try:
        jax.monitoring.register_event_duration_secs_listener(tracker)
        yield tracker
    finally:
        _unregister_event_duration_listener_by_callback(tracker)

import time
RETURN_VAL = TypeVar("RETURN_VAL")
def timed(f: Callable[...,RETURN_VAL], compilation: bool = True) -> Callable[...,RETURN_VAL]:
    def wrapped(*args, **kwargs) -> RETURN_VAL:
        compilation_time_tracker = CompilationTimeTracker()
        if compilation:
            jax.monitoring.register_event_duration_secs_listener(compilation_time_tracker)
        start_wall = time.perf_counter()
        start_cpu = time.process_time()
        out = f(*args, **kwargs)
        repr(out) # block until ready
        end_wall = time.perf_counter()
        end_cpu = time.process_time()
        # computing the metric and displaying it
        wall_time = end_wall - start_wall
        cpu_time = end_cpu - start_cpu
        cpu_count = os.cpu_count()
        print(f"cpu usage {cpu_time/wall_time:.1f}/{cpu_count} wall_time:{wall_time:.1f}s")
        if compilation:
            comp_time = compilation_time_tracker.get_total_time_secs()
            print(f"Total compilation time: {comp_time:.3f}s ({comp_time / (wall_time) * 100:.2f}%)", end="")
            print(f"    (trace={compilation_time_tracker.get_trace_time_secs():.3f}s, lower={compilation_time_tracker.get_lower_time_secs():.3f}s, compile={compilation_time_tracker.get_compile_time_secs():.3f}s)")
            _unregister_event_duration_listener_by_callback(compilation_time_tracker)
        return out
    return wrapped
