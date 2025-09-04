
import logging
import jax
import jax.numpy as jnp
import contextlib
from typing import Sequence, TypeVar, List, Optional, Callable, Dict, Any, Tuple
from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm.auto import tqdm
import os
import sys
from jax._src.config import trace_context
import subprocess
import psutil
import cpuinfo

__all__ = [
    "bcolors",
    "get_backend",
    "get_default_device",
    "setup_logging",
    "track_compilation_time",
    "CompilationTimeTracker",
    "broadcast_jaxtree",
    "timed",
    "log_debug",
    "log_info",
    "log_warn",
    "log_error",
    "log_critical",
    "get_environment_info",
    "write_json_result",
    "get_cpu_count",
    "check_pmap"
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
    # logging.DEBUG 10
    # logging.INFO 20
    # logging.WARN 30
    # logging.ERROR 40
    # logging.CRITICAL 50
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s: %(message)s'))
    logger.addHandler(handler)

def log(level: int, msg: str):
    with logging_redirect_tqdm(loggers=[logger]):
        logger.log(level, msg)
def log_debug(msg: str):
    log(logging.DEBUG, msg)
def log_info(msg: str):
    log(logging.INFO, msg)
def log_warn(msg: str):
    log(logging.WARN, msg)
def log_error(msg: str):
    log(logging.ERROR, msg)
def log_critical(msg: str):
    log(logging.CRITICAL, msg)

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
    msg = f"Compile {tracker.name} with input: {input_str} and axis-env: {axis_env_str}"
    # msg += "\n context:"+repr(trace_context())
    if tracker.has_variation(axis_env_str):
        if logger.level <= logging.DEBUG:
            tabs = " " * (len("dccxjax - WARNING: ") + len(f"Compile {tracker.name} with input:") - 8)
            msg += "".join([f"\n{tabs}prev-input: {prev_input}" for prev_input in tracker.get_variations(axis_env_str)])
        log_warn("Re-" + msg)
    else:
        log_info(msg)
    tracker.add_variation(axis_env_str, input_str)

def get_dtype_shape_str_of_tree(tree):
    def _dtype_shape(v):
        return jax.typeof(v).str_short(short_dtypes=False,mesh_axis_types=True)
    s = str(jax.tree.map(_dtype_shape, tree))
    return s

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

def get_cpu_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        return int(len(os.sched_getaffinity(0))) # type: ignore
    else:
        return int(psutil.cpu_count()) # type: ignore

def get_gpu_names() -> List[str]:
    try:
        return list(
            map(
                lambda l: l.partition(" (UUID")[0],
                subprocess.check_output(['nvidia-smi', '-L']).decode().rstrip().splitlines()
            )
        )
    except:
        return []

import time
from datetime import datetime
RETURN_VAL = TypeVar("RETURN_VAL")
def timed(f: Callable[...,RETURN_VAL], compilation: bool = True) -> Callable[...,Tuple[RETURN_VAL,Dict]]:
    def wrapped(*args, **kwargs) -> Tuple[RETURN_VAL,Dict]:
        compilation_time_tracker = CompilationTimeTracker()
        if compilation:
            jax.monitoring.register_event_duration_secs_listener(compilation_time_tracker)
        start_wall = time.perf_counter()
        start_cpu = time.process_time()
        start_date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        out = f(*args, **kwargs)
        repr(out) # block until ready
        end_wall = time.perf_counter()
        end_cpu = time.process_time()
        # computing the metric and displaying it
        wall_time = end_wall - start_wall
        cpu_time = end_cpu - start_cpu
        cpu_count = get_cpu_count()
        end_date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        print(f"cpu usage {cpu_time/wall_time:.1f}/{cpu_count} wall_time:{wall_time:.1f}s")
        timings = {
            "start_date": start_date,
            "end_date": end_date,
            "wall_time": wall_time,
            "cpu_usage": cpu_time/wall_time,
            "cpu_count": cpu_count
        }

        if compilation:
            jax_total_jit_time = compilation_time_tracker.get_total_time_secs()
            print(f"Total jit time: {jax_total_jit_time:.3f}s ({jax_total_jit_time / (wall_time) * 100:.2f}%)", end="")
            trace_time = compilation_time_tracker.get_trace_time_secs()
            lower_time = compilation_time_tracker.get_lower_time_secs()
            comp_time = compilation_time_tracker.get_compile_time_secs()
            print(f"    (trace={trace_time:.3f}s, lower={lower_time:.3f}s, compile={comp_time:.3f}s)")
            _unregister_event_duration_listener_by_callback(compilation_time_tracker)
            timings["jax_total_jit_time"] = jax_total_jit_time
            timings["jax_trace_time"] = trace_time
            timings["jax_lower_time"] = lower_time
            timings["jax_comp_time"] = comp_time
            
        return out, timings
    
    return wrapped

def _get_last_git_commit() -> str:
    try:
        return subprocess.check_output(['git', 'log',  '--format=%H', '-n', '1']).decode().rstrip()
    except:
        return ""

def get_environment_info() -> Dict:
    info = {
        "platform": jnp.array([]).device.platform, # type: ignore # there has to be a better way,
        "cpu-brand": cpuinfo.get_cpu_info()["brand_raw"],
        "gpu-brand": get_gpu_names(),
        "jax_environment": jax.print_environment_info(True),
        "cpu_count": get_cpu_count(),
        "git_commit": _get_last_git_commit(),
        "command": sys.argv[0]
    }
    info["n_available_devices"] = info["cpu_count"] if info["platform"] == "cpu" else len(info["gpu-brand"])
    return info
    
def write_json_result(json_result: Dict, *folders: str, prefix: str = ""):
    import pathlib, json, uuid
    from datetime import datetime
    platform = json_result["environment_info"]["platform"]
    # n_devices = json_result["environment_info"]["n_available_devices"]
    num_workers = json_result["pconfig"]["num_workers"]
    id_str = str(uuid.uuid4())
    json_result["id"] = id_str
    now = datetime.today().strftime('%Y-%m-%d_%H-%M')
    dev = json_result["environment_info"]["gpu-brand"][0][len("GPU 0: ")].replace(" ", "_") if platform == "gpu" else "cpu"
    fpath = pathlib.Path("experiments", "data", *folders, f"{dev}_{num_workers:02d}", f"{prefix}{platform}_{num_workers:02d}_date_{now}_{id_str[:8]}.json")
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "w") as f:
        json.dump(json_result, f, indent=2)
        
def check_pmap():
    return jax.pmap(lambda x: x + jax.lax.axis_index("batch"), axis_name="batch")(jnp.zeros((jax.device_count(),),int))