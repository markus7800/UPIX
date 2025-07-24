from typing import Dict
import os

from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "ParallelisationType",
    "ParallelisationConfig",
    "is_sequential",
    "is_parallel",
    "create_default_device_mesh",
]
    
class ParallelisationType(Enum):
    SequentialVMAP = 0
    SequentialPMAP = 1
    SequentialSMAP = 2
    SequentialGlobalSMAP = 3
    MultiProcessingCPU = 4
    MultiThreadingJAXDevices = 5
    
def is_sequential(parallelisation_type: ParallelisationType):
    return parallelisation_type in (
        ParallelisationType.SequentialVMAP,
        ParallelisationType.SequentialPMAP,
        ParallelisationType.SequentialSMAP,
        ParallelisationType.SequentialGlobalSMAP,
    )
def is_parallel(parallelisation_type: ParallelisationType):
    return parallelisation_type in (
        ParallelisationType.MultiProcessingCPU,
        ParallelisationType.MultiThreadingJAXDevices,
    )

@dataclass
class ParallelisationConfig:
    type: ParallelisationType = ParallelisationType.SequentialVMAP
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

import jax
from jax.sharding import Mesh
from jax.experimental.mesh_utils import create_device_mesh
from dccxjax.utils import bcolors

def create_default_device_mesh(dim: int):
    n_devices = jax.device_count()
    if n_devices == 1:
        print(bcolors.FAIL + "Creating device mesh for sharding, but only have one device." + bcolors.ENDC)
    if dim < n_devices:
        print(bcolors.FAIL + f"Configured {n_devices} devices, but sharding dim is smaller {dim}.\n"
              "Consider using pmap vectorisation instead." + bcolors.ENDC)
    
    if dim % n_devices != 0:
        print(bcolors.FAIL + f"Sharding dim={dim} is not multiple of number of devices={n_devices}." + bcolors.ENDC)
        
    return Mesh(create_device_mesh((n_devices,)), axis_names=("i",))