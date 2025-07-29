from typing import Dict
import os
from dataclasses import dataclass, field
from enum import Enum, auto
import jax
from jax.sharding import Mesh
from jax.experimental.mesh_utils import create_device_mesh
from dccxjax.utils import bcolors
from typing import Callable, TypeVar, Tuple
from dccxjax.jax_utils import smap_vmap, pmap_vmap, batch_func_args, unbatch_output
from dccxjax.types import IntArray

__all__ = [
    "ParallelisationType",
    "ParallelisationConfig",
    "is_sequential",
    "is_parallel",
    "create_default_device_mesh",
    "SHARDING_AXIS",
    "vectorise",
    "vectorise_scan",
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
    
# from tqdm.auto import tqdm
# def typeoftree(tree):
#     return jax.tree.map(jax.typeof, tree)

SCAN_DATA_TYPE = TypeVar("SCAN_DATA_TYPE")
SCAN_RETURN_TYPE = TypeVar("SCAN_RETURN_TYPE")
SCAN_CARRY_TYPE = TypeVar("SCAN_CARRY_TYPE")

from dccxjax.progress_bar import ProgressbarManager, _add_progress_bar
import jax.experimental

def vectorise_scan(step: Callable[[SCAN_CARRY_TYPE,SCAN_DATA_TYPE],Tuple[SCAN_CARRY_TYPE,SCAN_RETURN_TYPE]],
                   carry_axes, pmap_data_axes, batch_axis_size: int, pconfig: ParallelisationConfig,
                   progressbar_mngr: ProgressbarManager | None = None, get_iternum_fn: Callable[[SCAN_CARRY_TYPE], IntArray] | None = None):
    
    device_count = jax.device_count()
    
    def _scan(init: SCAN_CARRY_TYPE, data: SCAN_DATA_TYPE) -> Tuple[SCAN_CARRY_TYPE,SCAN_RETURN_TYPE]:
        if progressbar_mngr is not None:
            progressbar_mngr.start_progress()
            
        if pconfig.vectorsisation == VectorisationType.LocalVMAP:
            _step = step
        elif pconfig.vectorsisation == VectorisationType.GlobalVMAP:
            _step = jax.vmap(step, in_axes=(carry_axes,0), out_axes=(carry_axes,0))
        elif pconfig.vectorsisation == VectorisationType.PMAP:
            if batch_axis_size <= device_count:
                _step = step
            else:
                _step = jax.vmap(step, in_axes=(carry_axes,0), out_axes=(carry_axes,0))
        elif pconfig.vectorsisation == VectorisationType.LocalSMAP:
            _step = step
        else:
            assert pconfig.vectorsisation == VectorisationType.GlobalSMAP
            _step = smap_vmap(step, axis_name=SHARDING_AXIS, in_axes=(carry_axes,0), out_axes=(carry_axes,0))
            
        if progressbar_mngr is not None:
            assert get_iternum_fn is not None
            _step = _add_progress_bar(_step, get_iternum_fn, progressbar_mngr, progressbar_mngr.num_samples)
            jax.experimental.io_callback(progressbar_mngr._init_tqdm, None, get_iternum_fn(init))
 
        return jax.lax.scan(_step, init, data)
        
        
    if pconfig.vectorsisation == VectorisationType.PMAP:
        if batch_axis_size <= device_count:
            return jax.pmap(_scan, axis_name=SHARDING_AXIS, in_axes=(carry_axes,pmap_data_axes), out_axes=(carry_axes,pmap_data_axes))
        else:
            assert batch_axis_size % device_count == 0
            batch_size = batch_axis_size // device_count
            def _batched_scan(init: SCAN_CARRY_TYPE, data: SCAN_DATA_TYPE) -> Tuple[SCAN_CARRY_TYPE,SCAN_RETURN_TYPE]:
                args = (init, data)
                batched_args, num_batches = batch_func_args(args, (carry_axes,pmap_data_axes), batch_size)
                pfun = jax.pmap(_scan, axis_name=SHARDING_AXIS, in_axes=(carry_axes,pmap_data_axes), out_axes=(carry_axes,pmap_data_axes))
                batched_out = pfun(*batched_args)
                return unbatch_output(batched_out, (carry_axes,pmap_data_axes), batch_size, num_batches)
            return _batched_scan
            
    else:
        return _scan
        