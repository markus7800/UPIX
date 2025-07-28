from typing import Dict
import os
from dataclasses import dataclass, field
from enum import Enum, auto
import jax
from jax.sharding import Mesh
from jax.experimental.mesh_utils import create_device_mesh
from dccxjax.utils import bcolors
from typing import Callable, TypeVar, Tuple
from dccxjax.jax_utils import smap_vmap, pmap_vmap

__all__ = [
    "ParallelisationType",
    "ParallelisationConfig",
    "is_sequential",
    "is_parallel",
    "create_default_device_mesh",
    "SHARDING_AXIS",
    "vectorise",
    "parallel_run",
]

class ParallelisationType(Enum):
    Sequential = auto()
    MultiProcessingCPU = auto()
    MultiThreadingJAXDevices = auto()
    
class VectorisationType(Enum):
    LocalVMAP = auto()
    GlobalVMAP = auto()
    PMAP = auto()
    LocalSMAP = auto()
    GlobalSMAP = auto()

@dataclass
class ParallelisationConfig:
    parallelisation: ParallelisationType = ParallelisationType.Sequential
    vectorsisation: VectorisationType = VectorisationType.GlobalVMAP
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


def is_sequential(pconfig: ParallelisationConfig):
    return pconfig.parallelisation == ParallelisationType.Sequential
def is_parallel(pconfig: ParallelisationConfig):
    return pconfig.parallelisation in (
        ParallelisationType.MultiProcessingCPU,
        ParallelisationType.MultiThreadingJAXDevices,
    )
    
SHARDING_AXIS = "s_axis"
    
def create_default_device_mesh(dim: int):
    n_devices = jax.device_count()
    if n_devices == 1:
        print(bcolors.FAIL + "Creating device mesh for sharding, but only have one device." + bcolors.ENDC)
    if dim < n_devices:
        print(bcolors.FAIL + f"Configured {n_devices} devices, but sharding dim is smaller {dim}.\n"
              "Consider using pmap vectorisation instead." + bcolors.ENDC)
    
    if dim % n_devices != 0:
        print(bcolors.FAIL + f"Sharding dim={dim} is not multiple of number of devices={n_devices}." + bcolors.ENDC)
        
    return Mesh(create_device_mesh((n_devices,)), axis_names=(SHARDING_AXIS,))

FUNCTION_TYPE = TypeVar("FUNCTION_TYPE", bound=Callable)
FUNCTION_RET_TYPE = TypeVar("FUNCTION_RET_TYPE")
def vectorise(fn: FUNCTION_TYPE, in_axes, out_axes, batch_axis_size: int, pconfig: ParallelisationConfig) -> FUNCTION_TYPE:
    if pconfig.vectorsisation == VectorisationType.LocalVMAP:
        return jax.jit(fn) # type: ignore
    elif pconfig.vectorsisation == VectorisationType.GlobalVMAP:
        return jax.jit(jax.vmap(fn, in_axes=in_axes, out_axes=out_axes)) # type: ignore
    elif pconfig.vectorsisation == VectorisationType.PMAP:
        device_count = jax.device_count()
        if batch_axis_size <= device_count:
            return jax.pmap(fn, axis_name=SHARDING_AXIS, in_axes=in_axes, out_axes=out_axes)
        else:
            assert batch_axis_size % device_count == 0
            batch_size = batch_axis_size // device_count
            return pmap_vmap(fn, axis_name=SHARDING_AXIS, batch_size=batch_size, in_axes=in_axes, out_axes=out_axes)
    elif pconfig.vectorsisation == VectorisationType.LocalSMAP:
        return jax.jit(fn) # type: ignore
    else:
        assert pconfig.vectorsisation == VectorisationType.GlobalSMAP
        return jax.jit(smap_vmap(fn, axis_name=SHARDING_AXIS, in_axes=in_axes, out_axes=out_axes)) # type: ignore
        
def parallel_run(fn: Callable[..., FUNCTION_RET_TYPE], args: Tuple, batch_axis_size: int, pconfig: ParallelisationConfig) -> FUNCTION_RET_TYPE:
    if pconfig.vectorsisation == VectorisationType.LocalSMAP or pconfig.vectorsisation == VectorisationType.GlobalSMAP:
        with jax.sharding.use_mesh(create_default_device_mesh(batch_axis_size)):
            return fn(*args)
    else:
        return fn(*args)