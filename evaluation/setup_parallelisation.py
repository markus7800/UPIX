
from dccxjax.parallelisation import ParallelisationConfig, ParallelisationType, VectorisationType
import jax
import os

def get_parallelisation_config(args) -> ParallelisationConfig:
    parallelisation: str = args.parallelisation
    vectorisation: str = args.vectorisation
    num_workers: int = int(args.num_workers)
    assert parallelisation in ("sequential", "cpu_multiprocess", "jax_devices")
    assert vectorisation in ("vmap", "vmap_local", "vmap_global", "pmap", "smap", "smap_global", "smap_local")
    if parallelisation != "sequential" and vectorisation != "vmap":
        print(f"Ignoring vectoristion '{vectorisation}' for non-sequential parallisation '{parallelisation}'")
    if parallelisation == "sequential":
        if num_workers > 0:
            print(f"Ignoring num_workers '{vectorisation}' for non-sequential parallisation '{parallelisation}'")
        if vectorisation == "vmap":
            return ParallelisationConfig(parallelisation=ParallelisationType.Sequential, vectorsisation=VectorisationType.GlobalVMAP)
        elif vectorisation == "vmap_local":
            return ParallelisationConfig(parallelisation=ParallelisationType.Sequential, vectorsisation=VectorisationType.LocalVMAP)
        elif vectorisation == "vmap_global":
            return ParallelisationConfig(parallelisation=ParallelisationType.Sequential, vectorsisation=VectorisationType.GlobalVMAP)
        else:
            if jax.device_count() == 1:
                print(f"Warning: Vectorisation is set to'{vectorisation}' but only 1 jax devices is available.")
        if vectorisation == "pmap":
            return ParallelisationConfig(parallelisation=ParallelisationType.Sequential, vectorsisation=VectorisationType.PMAP)
        if vectorisation == "smap":
            return ParallelisationConfig(parallelisation=ParallelisationType.Sequential, vectorsisation=VectorisationType.GlobalSMAP)
        if vectorisation == "smap_local":
            return ParallelisationConfig(parallelisation=ParallelisationType.Sequential, vectorsisation=VectorisationType.LocalSMAP)
        if vectorisation == "smap_global":
            return ParallelisationConfig(parallelisation=ParallelisationType.Sequential, vectorsisation=VectorisationType.GlobalSMAP)
    else:
        assert vectorisation in ("vmap", "vmap_local", "vmap_global")
        vectorisation_type = VectorisationType.LocalVMAP if vectorisation == "vmap_local" else VectorisationType.GlobalVMAP
        if parallelisation == "cpu_multiprocess":
            return ParallelisationConfig(
                parallelisation=ParallelisationType.MultiProcessingCPU,
                vectorsisation=vectorisation_type,
                num_workers=num_workers or os.cpu_count() or 1
            )
        else:
            assert parallelisation == "jax_devices"
            assert num_workers <= jax.device_count()
            return ParallelisationConfig(
                parallelisation=ParallelisationType.MultiThreadingJAXDevices,
                vectorsisation=vectorisation_type,
                num_workers=num_workers or jax.device_count()
            )