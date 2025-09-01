from typing import Dict
import os
from dataclasses import dataclass, field
from enum import Enum, auto, StrEnum
import jax
from jax.sharding import Mesh
from jax.experimental.mesh_utils import create_device_mesh
from dccxjax.utils import maybe_jit_warning, JitVariationTracker, log_warn
from typing import Callable, TypeVar, Tuple
from dccxjax.jax_utils import batched_vmap, smap_vmap, pmap_vmap, batch_func_args, unbatch_output, concatentate_output
from dccxjax.types import IntArray
from functools import partial
from dccxjax.utils import log_critical

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

class ParallelisationType(StrEnum):
    Sequential = auto()
    MultiProcessingCPU = auto()
    MultiThreadingJAXDevices = auto()
    
class VectorisationType(StrEnum):
    LocalVMAP = auto()
    GlobalVMAP = auto()
    PMAP = auto()
    LocalSMAP = auto()
    GlobalSMAP = auto()

@dataclass
class ParallelisationConfig:
    parallelisation: ParallelisationType = ParallelisationType.Sequential
    vectorisation: VectorisationType = VectorisationType.GlobalVMAP
    num_workers: int = None # type: ignore makes sure this is set properly
    cpu_affinity: bool = False
    vmap_batch_size: int = 0
    force_task_order : bool = False
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
    
def create_default_device_mesh(dim: int, n_devices: int):
    devices = jax.devices()[:n_devices]
    device_count = len(devices)
    if device_count == 1:
        log_warn("Creating device mesh for sharding, but only have one device.")
    if dim < device_count:
        log_critical(f"Configured {device_count} devices, but sharding dim is smaller {dim}.\n"
              "Consider using pmap vectorisation instead.")
        assert device_count <= dim
    if dim % device_count != 0:
        log_critical(f"Sharding dim={dim} is not multiple of number of devices={device_count}.")
        assert dim % device_count == 0
    return Mesh(create_device_mesh((device_count,), devices=devices), axis_names=(SHARDING_AXIS,))

FUNCTION_TYPE = TypeVar("FUNCTION_TYPE", bound=Callable)
FUNCTION_RET_TYPE = TypeVar("FUNCTION_RET_TYPE")
def vectorise(fn: FUNCTION_TYPE, in_axes, out_axes, batch_axis_size: int, vectorisation: VectorisationType, n_devices: int, vmap_batch_size: int) -> FUNCTION_TYPE:
    if vectorisation == VectorisationType.LocalVMAP:
        return jax.jit(fn) # type: ignore
    elif vectorisation == VectorisationType.GlobalVMAP:
        if vmap_batch_size > 0:
            return jax.jit(batched_vmap(fn, batch_size=vmap_batch_size, in_axes=in_axes, out_axes=out_axes)) # type: ignore
        else:
            return jax.jit(jax.vmap(fn, in_axes=in_axes, out_axes=out_axes)) # type: ignore
    elif vectorisation == VectorisationType.PMAP:
        devices = jax.devices()[:n_devices]
        device_count = len(devices)
        if batch_axis_size <= device_count:
            return jax.pmap(fn, axis_name=SHARDING_AXIS, in_axes=in_axes, out_axes=out_axes, devices=devices[:batch_axis_size])
        else:
            # assert batch_axis_size % device_count == 0
            return pmap_vmap(fn, axis_name=SHARDING_AXIS, num_batches=device_count, in_axes=in_axes, out_axes=out_axes, devices=devices, vmap_batch_size=vmap_batch_size)
    elif vectorisation == VectorisationType.LocalSMAP:
        return jax.jit(fn) # type: ignore
    else:
        assert vectorisation == VectorisationType.GlobalSMAP
        return jax.jit(smap_vmap(fn, axis_name=SHARDING_AXIS, in_axes=in_axes, out_axes=out_axes, vmap_batch_size=vmap_batch_size)) # type: ignore
        
def parallel_run(fn: Callable[..., FUNCTION_RET_TYPE], args: Tuple, batch_axis_size: int, vectorisation: VectorisationType, n_devices: int) -> FUNCTION_RET_TYPE:
    if vectorisation == VectorisationType.LocalSMAP or vectorisation == VectorisationType.GlobalSMAP:
        with jax.set_mesh(create_default_device_mesh(batch_axis_size, n_devices)):
            return jax.device_get(fn(*args)) # device_get to unshard output
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
                   carry_axes, pmap_data_axes, batch_axis_size: int, vmap_batch_size: int, vectorisation: VectorisationType, n_devices: int,
                   progressbar_mngr: ProgressbarManager | None = None, get_iternum_fn: Callable[[SCAN_CARRY_TYPE], IntArray] | None = None):
    
    if vectorisation == VectorisationType.PMAP:
        return vectorise_scan_pmap(step, carry_axes, pmap_data_axes, batch_axis_size, vmap_batch_size, vectorisation, n_devices, progressbar_mngr, get_iternum_fn)
    
    @jax.jit
    def _scan(init: SCAN_CARRY_TYPE, data: SCAN_DATA_TYPE) -> Tuple[SCAN_CARRY_TYPE,SCAN_RETURN_TYPE]:
        # maybe_jit_warning(tracker, (init, data))
        if progressbar_mngr is not None:
            progressbar_mngr.start_progress()
            
        if vectorisation == VectorisationType.LocalVMAP:
            _step = step
        elif vectorisation == VectorisationType.GlobalVMAP:
            if vmap_batch_size > 0:
                _step = batched_vmap(step, batch_size=vmap_batch_size, in_axes=(carry_axes,0), out_axes=(carry_axes,0))
            else:
                _step = jax.vmap(step, in_axes=(carry_axes,0), out_axes=(carry_axes,0))
        elif vectorisation == VectorisationType.LocalSMAP:
            _step = step
        else:
            assert vectorisation == VectorisationType.GlobalSMAP
            _step = smap_vmap(step, axis_name=SHARDING_AXIS, in_axes=(carry_axes,0), out_axes=(carry_axes,0))
            
        if progressbar_mngr is not None:
            assert get_iternum_fn is not None
            _step = _add_progress_bar(_step, get_iternum_fn, progressbar_mngr, progressbar_mngr.num_samples)
            jax.experimental.io_callback(progressbar_mngr._init_tqdm, None, get_iternum_fn(init))
 
        return jax.lax.scan(_step, init, data)
    
    return _scan
   

def vectorise_scan_pmap(step: Callable[[SCAN_CARRY_TYPE,SCAN_DATA_TYPE],Tuple[SCAN_CARRY_TYPE,SCAN_RETURN_TYPE]],
                   carry_axes, pmap_data_axes, batch_axis_size: int, vmap_batch_size: int, vectorisation: VectorisationType, n_devices: int,
                   progressbar_mngr: ProgressbarManager | None = None, get_iternum_fn: Callable[[SCAN_CARRY_TYPE], IntArray] | None = None):
         
    assert vectorisation == VectorisationType.PMAP
    
    def _scan(init: SCAN_CARRY_TYPE, data: SCAN_DATA_TYPE, vmap_step: bool) -> Tuple[SCAN_CARRY_TYPE,SCAN_RETURN_TYPE]:
        if progressbar_mngr is not None:
            progressbar_mngr.start_progress()
            
        if vmap_step:
            if vmap_batch_size > 0:
                _step = batched_vmap(step, in_axes=(carry_axes,0), out_axes=(carry_axes,0), batch_size=vmap_batch_size)
            else:
                _step = jax.vmap(step, in_axes=(carry_axes,0), out_axes=(carry_axes,0))
        else:
            _step = step
            
        if progressbar_mngr is not None:
            assert get_iternum_fn is not None
            _step = _add_progress_bar(_step, get_iternum_fn, progressbar_mngr, progressbar_mngr.num_samples)
            jax.experimental.io_callback(progressbar_mngr._init_tqdm, None, get_iternum_fn(init))
 
        return jax.lax.scan(_step, init, data)
    
    devices = jax.devices()[:n_devices]
    device_count = len(devices)
    # special case, because we want to keep pmap at top level and do not want to put it into scan
    # we cannot use pmap_vmap at top level, because to use progressbar we cannot vmap scan 
    if batch_axis_size <= device_count:
        return jax.pmap(jax.jit(partial(_scan, vmap_step=False)), axis_name=SHARDING_AXIS, in_axes=(carry_axes,pmap_data_axes), out_axes=(carry_axes,pmap_data_axes), devices=devices[:batch_axis_size])
    else:
        num_batches = device_count
        def _batched_scan(init: SCAN_CARRY_TYPE, data: SCAN_DATA_TYPE) -> Tuple[SCAN_CARRY_TYPE,SCAN_RETURN_TYPE]:
            args = (init, data)
            batched_args, remainder_args, _, batch_size, remainder = batch_func_args(args, (carry_axes,pmap_data_axes), num_batches=num_batches)
            if remainder > 0:
                log_warn(
                    f"Vectorise scan with pmap and positive remainder={remainder}. "
                    "Causes scan to be compiled and run twice. "
                    f"This can be avoided by having {batch_axis_size=} % device_count == 0."
                )
            
            assert batched_args is not None
            if remainder_args is not None:
                if progressbar_mngr is not None: progressbar_mngr.set_msgs_per_update(remainder)
                remainder_pfun = jax.pmap(jax.jit(partial(_scan,vmap_step=False)), axis_name=SHARDING_AXIS, in_axes=(carry_axes,pmap_data_axes), out_axes=(carry_axes,pmap_data_axes), devices=devices[:remainder])
                # remainder_pfun = jax.jit(partial(_scan,vmap_step=True))
                # removing jax.block_until_ready messes up progressbar. pmap can be dispatched asyncronously?
                remainder_out = jax.block_until_ready(remainder_pfun(*remainder_args))
            else:
                remainder_out = None
                
            if progressbar_mngr is not None: progressbar_mngr.set_msgs_per_update(device_count)
            pfun = jax.pmap(jax.jit(partial(_scan,vmap_step=True)), axis_name=SHARDING_AXIS, in_axes=(carry_axes,pmap_data_axes), out_axes=(carry_axes,pmap_data_axes), devices=devices)
            batched_out = pfun(*batched_args)
            out_axes = (carry_axes,pmap_data_axes)
            unbatched_out = unbatch_output(batched_out, out_axes, batch_size, num_batches)
            
            if remainder_out is not None:
                return concatentate_output(unbatched_out, remainder_out, out_axes)
            else:
                return unbatched_out
            
        return _batched_scan
            