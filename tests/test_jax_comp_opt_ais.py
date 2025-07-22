import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import contextlib


from jax._src.monitoring import EventListenerWithMetadata, EventDurationListenerWithMetadata, EventTimeSpanListenerWithMetadata
from jax._src.monitoring import _unregister_event_listener_by_callback, _unregister_event_duration_listener_by_callback, _unregister_event_time_span_listener_by_callback
from jax._src.dispatch import JAXPR_TRACE_EVENT, JAXPR_TO_MLIR_MODULE_EVENT, BACKEND_COMPILE_EVENT

class CompilationEventTracker(EventListenerWithMetadata):
    def __call__(self, event: str, **kwargs: str | int):
        # if event in (JAXPR_TRACE_EVENT, JAXPR_TO_MLIR_MODULE_EVENT, BACKEND_COMPILE_EVENT):
        print("EVENT", event, kwargs)

class CompilationTimeTracker(EventDurationListenerWithMetadata):
    def __init__(self) -> None:
        self.total_time = 0.
    def get_total_compilation_time_secs(self):
        return self.total_time
    def __call__(self, event: str, duration_secs: float,
               **kwargs: str | int) -> None:
        if event in (JAXPR_TRACE_EVENT, JAXPR_TO_MLIR_MODULE_EVENT, BACKEND_COMPILE_EVENT):
            print("TIME", event, duration_secs, kwargs)
            self.total_time += duration_secs


class CompilationTimespanTracker(EventTimeSpanListenerWithMetadata):
    def __call__(self, event: str, start_time: float, end_time: float, **kwargs: str | int) -> None:
        if event in (JAXPR_TRACE_EVENT, JAXPR_TO_MLIR_MODULE_EVENT, BACKEND_COMPILE_EVENT):
            print("TIMESPAN", event, start_time, end_time, end_time - start_time, kwargs)


@contextlib.contextmanager
def track_compilation_event():
    tracker = CompilationEventTracker()
    try:
        jax.monitoring.register_event_listener(tracker)
        yield tracker
    finally:
        _unregister_event_listener_by_callback(tracker)

@contextlib.contextmanager
def track_compilation_time():
    tracker = CompilationTimeTracker()
    try:
        jax.monitoring.register_event_duration_secs_listener(tracker)
        yield tracker
    finally:
        _unregister_event_duration_listener_by_callback(tracker)

@contextlib.contextmanager
def track_compilation_timespan():
    tracker = CompilationTimespanTracker()
    try:
        jax.monitoring.register_event_time_span_listener(tracker)
        yield tracker
    finally:
        _unregister_event_time_span_listener_by_callback(tracker)

@jax.jit
def log_prob(xs: jax.Array):
    lp = 0.0
    for i in range(100):
        lp += dist.Normal(0., float(i)).log_prob(xs).sum()
    # jax.debug.print("log_prob {x}", x=xs[0])
    return lp

from typing import NamedTuple
class Carry(NamedTuple):
    xs: jax.Array
    lp: jax.Array
    lp_diff: jax.Array

@jax.jit
def step(carry: Carry, rng_key: jax.Array):
    print("Compile step")
    xs = carry.xs + jax.random.normal(rng_key, carry.xs.shape)
    lp = log_prob(xs)
    lp_diff = log_prob(xs) - log_prob(carry.xs)
    return Carry(xs, lp, lp_diff), None


def f1(X) -> Carry:
    print("Compile f1")
    keys = jax.random.split(jax.random.PRNGKey(0), 100)
    return jax.lax.scan(step, Carry(X, 0., 0.), keys)[0]

def f2(X) -> jax.Array:
    print("Compile f2")
    keys = jax.random.split(jax.random.PRNGKey(0), 100)
    return jax.lax.scan(step, Carry(X, 0., 0.), keys)[0].lp

X = jax.random.normal(jax.random.PRNGKey(0), (1_000_000,))

from time import time

# with track_compilation_event(), track_compilation_time() as tracker, track_compilation_timespan(): 
    # print(jax.jit(log_prob)(X))
    # print(tracker.total_time)

print("\nf1\n")
# print(jax.make_jaxpr(f1)(X))
t0 = time()
result = jax.jit(f1)(X)
result.lp.block_until_ready()
t1 = time()
print(f"Finished in {t1-t0:.3f}")

print("\nf2\n")
# print(jax.make_jaxpr(jax.jit(f2))(X))
t0 = time()
result = jax.jit(f2)(X)
result.block_until_ready()
t1 = time()
print(f"Finished in {t1-t0:.3f}")

