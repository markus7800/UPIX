
import logging
import jax
from jax._src.core import full_lower
import contextlib
from typing import Sequence, TypeVar, List, Optional
from tqdm.contrib.logging import logging_redirect_tqdm
import os
import re

__all__ = [
    "setup_logging",
    "track_compilation_time",
    "CompilationTimeTracker",
    "broadcast_jaxtree",
    "set_platform",
    "set_host_device_count",
]

logger = logging.getLogger("dccxjax")

def setup_logging(level: int | str):
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s: %(message)s'))
    logger.addHandler(handler)

class JitVariationTracker:
    def __init__(self, name: str) -> None:
        self.name = name
        self.variations: List[str] = []
    def add_variation(self, variation: str):
        self.variations.append(variation)
    def has_variation(self):
        return len(self.variations) > 0

def maybe_jit_warning(tracker: JitVariationTracker, input: str):
    msg = f"Compile {tracker.name} and for input: {input}"
    with logging_redirect_tqdm(loggers=[logger]):
        if tracker.has_variation():
            if logger.level <= logging.DEBUG:
                tabs = " " * (len(msg) - len(input) + len("dccxjax - WARNING: ") - 9)
                msg += "".join([f"\n{tabs}prev-input: {prev_input}" for prev_input in tracker.variations])
            logger.warning("Re-" + msg)
        else:
            logger.debug(msg)
    tracker.add_variation(input)

# def to_shaped_arrays(tree):
#     return jax.tree.map(lambda v: full_lower(v).aval, tree)

# def to_shaped_arrays_str_short(tree):
#     return jax.tree.map(lambda v: full_lower(v).str_short(), tree)

def pprint_dtype_shape_of_tree(tree):
    def _dtype_shape(v):
        v = full_lower(v)
        shape_str = ",".join(f"{d:_}" for d in v.shape)
        return f"{v.dtype}[{shape_str}]"
    s = repr(jax.tree.map(_dtype_shape, tree))
    return s.replace("'", "")

def to_shaped_arrays_str_short(tree):
    return pprint_dtype_shape_of_tree(tree)

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
        self.total_time = 0.
    def get_total_compilation_time_secs(self):
        return self.total_time
    def __call__(self, event: str, duration_secs: float,
               **kwargs: str | int) -> None:
        if event in (JAXPR_TRACE_EVENT, JAXPR_TO_MLIR_MODULE_EVENT, BACKEND_COMPILE_EVENT):
            self.total_time += duration_secs

@contextlib.contextmanager
def track_compilation_time():
    tracker = CompilationTimeTracker()
    try:
        jax.monitoring.register_event_duration_secs_listener(tracker)
        yield tracker
    finally:
        _unregister_event_duration_listener_by_callback(tracker)


# from https://github.dev/pyro-ppl/numpyro

def set_platform(platform: Optional[str] = None) -> None:
    if platform is None:
        platform = os.getenv("JAX_PLATFORM_NAME", "cpu")
    jax.config.update("jax_platform_name", platform)


def set_host_device_count(n: int) -> None:
    xla_flags_str = os.getenv("XLA_FLAGS", "")
    xla_flags = re.sub(
        r"--xla_force_host_platform_device_count=\S+", "", xla_flags_str
    ).split()
    os.environ["XLA_FLAGS"] = " ".join(
        ["--xla_force_host_platform_device_count={}".format(n)] + xla_flags
    )