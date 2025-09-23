from dccxjax.types import Trace
from dataclasses import dataclass
from typing import Callable, Optional, Set, Tuple, Any
import jax
from jax.flatten_util import ravel_pytree

__all__ = [
    "TraceTansformation",
    "read_discrete",
    "write_discrete",
    "read_continuous",
    "write_continuous",
    "copy_at_address"
]

class TraceTansformation:
    f: Callable[[Trace,Trace,Trace,Trace],None]
    continuous_reads: Set[str]
    continuous_writes: Set[str]
    def __init__(self, f: Callable[[Trace,Trace,Trace,Trace],None]) -> None:
        self.f = f
        self.continuous_reads = set()
        self.continuous_writes = set()
    def __enter__(self):
        global TRACE_TRANSFORM_CONTEXT
        TRACE_TRANSFORM_CONTEXT.set(self)
        return self
    def __exit__(self, *args):
        global TRACE_TRANSFORM_CONTEXT
        TRACE_TRANSFORM_CONTEXT.set(None)
    def apply(self, model_trace: Trace, aux_trace: Trace) -> Tuple[Trace,Trace]:
        self.continuous_reads.clear()
        self.continuous_writes.clear()
        new_model_trace: Trace = dict()
        new_aux_trace: Trace = dict()
        self.f(model_trace, aux_trace, new_model_trace, new_aux_trace)
        return new_model_trace, new_aux_trace
    def jacobian(self, model_trace: Trace, aux_trace: Trace) -> jax.Array:
        diff_model_trace: Trace = {addr: model_trace[addr] for addr in model_trace.keys() if addr in self.continuous_reads}
        no_diff_model_trace: Trace = {addr: model_trace[addr] for addr in model_trace.keys() if addr not in self.continuous_reads}
        diff_aux_trace: Trace = {addr: aux_trace[addr] for addr in aux_trace.keys() if addr in self.continuous_reads}
        no_diff_aux_trace: Trace = {addr: aux_trace[addr] for addr in aux_trace.keys() if addr not in self.continuous_reads}
        diff_X, unravel_fn = ravel_pytree((diff_model_trace, diff_aux_trace))
        # print(self.continuous_reads, self.continuous_writes)
        def flat_diff_transform(X):
            _mtr, _atr = unravel_fn(X)
            model_tr = _mtr | no_diff_model_trace
            aux_tr = _atr | no_diff_aux_trace
            new_model_tr: Trace = dict()
            new_aux_tr: Trace = dict()
            self.f(model_tr, aux_tr, new_model_tr, new_aux_tr)
            diff_new_model_tr = {addr: new_model_tr[addr] for addr in new_model_tr.keys() if addr in self.continuous_writes}
            diff_new_aux_tr = {addr: new_aux_tr[addr] for addr in new_aux_tr.keys() if addr in self.continuous_writes}
            diff_out, _ = ravel_pytree((diff_new_model_tr, diff_new_aux_tr))
            return diff_out
        return jax.jacfwd(flat_diff_transform)(diff_X)
    

import threading
class TraceTransformContextStore(threading.local):
    def __init__(self):
        self.ctx: Optional[TraceTansformation] = None
    def get(self):
        return self.ctx
    def set(self, ctx: Optional["TraceTansformation"]):
        self.ctx = ctx
        
TRACE_TRANSFORM_CONTEXT = TraceTransformContextStore()

def read_discrete(old_trace: Trace, address: str):
    global TRACE_TRANSFORM_CONTEXT
    ctx = TRACE_TRANSFORM_CONTEXT.get()
    assert ctx is not None
    return old_trace[address]

def write_discrete(new_trace: Trace, address: str, value):
    global TRACE_TRANSFORM_CONTEXT
    ctx = TRACE_TRANSFORM_CONTEXT.get()
    assert ctx is not None
    new_trace[address] = value
    
def read_continuous(old_trace: Trace, address: str):
    global TRACE_TRANSFORM_CONTEXT
    ctx = TRACE_TRANSFORM_CONTEXT.get()
    assert ctx is not None
    ctx.continuous_reads.add(address)
    return old_trace[address]

def write_continuous(new_trace: Trace, address: str, value):
    global TRACE_TRANSFORM_CONTEXT
    ctx = TRACE_TRANSFORM_CONTEXT.get()
    assert ctx is not None
    ctx.continuous_writes.add(address)
    new_trace[address] = value
    
def copy_at_address(old_trace: Trace, new_trace: Trace, address: str):
    global TRACE_TRANSFORM_CONTEXT
    ctx = TRACE_TRANSFORM_CONTEXT.get()
    assert ctx is not None
    new_trace[address] = old_trace[address]
