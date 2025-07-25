
from dccxjax.infer.dcc.parallelisation import ParallelisationConfig, ParallelisationType
import jax
import os

def get_parallelisation_config(args) -> ParallelisationConfig:
    parallelisation: str = args.parallelisation
    vectorisation: str = args.vectorisation
    num_workers: int = int(args.num_workers)
    assert parallelisation in ("sequential", "cpu_multiprocess", "jax_devices")
    assert vectorisation in ("vmap", "vmap_global", "pmap", "smap", "smap_global")
    if parallelisation != "sequential" and vectorisation != "vmap":
        print(f"Ignoring vectoristion '{vectorisation}' for non-sequential parallisation '{parallelisation}'")
    if parallelisation == "sequential":
        if num_workers > 0:
            print(f"Ignoring num_workers '{vectorisation}' for non-sequential parallisation '{parallelisation}'")
        if vectorisation == "vmap":
            return ParallelisationConfig(type=ParallelisationType.SequentialVMAP)
        elif vectorisation == "vmap_global":
            return ParallelisationConfig(type=ParallelisationType.SequentialGlobalVMAP)
        else:
            if jax.device_count() == 1:
                print(f"Warning: Vectorisation is set to'{vectorisation}' but only 1 jax devices is available.")
        if vectorisation == "pmap":
            return ParallelisationConfig(type=ParallelisationType.SequentialPMAP)
        if vectorisation == "smap":
            return ParallelisationConfig(type=ParallelisationType.SequentialSMAP)
        if vectorisation == "smap_global":
            return ParallelisationConfig(type=ParallelisationType.SequentialGlobalSMAP)
            
    elif parallelisation == "cpu_multiprocess":
        return ParallelisationConfig(
            type=ParallelisationType.MultiProcessingCPU,
            num_workers=num_workers or os.cpu_count() or 1
        )
    else:
        assert parallelisation == "jax_devices"
        assert num_workers <= jax.device_count()
        return ParallelisationConfig(
            type=ParallelisationType.MultiThreadingJAXDevices,
            num_workers=num_workers or jax.device_count()
        )