
import logging
import jax
from jax.core import full_lower
import contextlib
from typing import Sequence, TypeVar

__all__ = [
    "setup_logging",
    "track_compilation_time",
    "CompilationTimeTracker",
    "broadcast_jaxtree",
]

logger = logging.getLogger("dccxjax")

def setup_logging(level: int | str):
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s: %(message)s'))
    logger.addHandler(handler)

def maybe_jit_warning(obj, attr, fname, short_repr, input):
    msg = f"Compile {fname} for {short_repr} and {input}"
    if obj is not None:
        if not getattr(obj, attr):
            setattr(obj, attr, True)
            logger.debug(msg)
        else:
            logger.warning("Re-" + msg)
    else:
        logger.debug(msg)


def to_shaped_arrays(tree):
    return jax.tree.map(lambda v: full_lower(v).aval, tree)

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
