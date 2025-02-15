
import logging
from .types import Trace
import jax
from jax.core import full_lower

__all__ = [
    "setup_logging",
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


def to_shaped_array_trace(X: Trace):
    return {address: value.aval for address, value in X.items()}

def to_shaped_arrays(tree):
    return jax.tree.map(lambda v: full_lower(v).aval, tree)