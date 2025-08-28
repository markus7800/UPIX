
from dccxjax.parallelisation import ParallelisationConfig, ParallelisationType, VectorisationType
import jax
import os

def get_parallelisation_config(args) -> ParallelisationConfig:
    parallelisation: str = args.parallelisation
    vectorisation: str = args.vectorisation
    num_workers: int = int(args.num_workers)
    assert parallelisation in ("sequential", "cpu_multiprocess", "jax_devices")
    assert vectorisation in ("vmap_local", "vmap_global", "pmap", "smap_global", "smap_local")
    if parallelisation != "sequential" and not vectorisation.startswith("vmap"):
        print(f"Ignoring vectorisation '{vectorisation}' for non-sequential parallisation '{parallelisation}'")
    if vectorisation not in ("vmap_local", "vmap_global") and args.vmap_batch_size > 0:
        print(f"Ignoring batch_size {args.vmap_batch_size} for vectorisation '{vectorisation}'")
    if parallelisation == "sequential":
        if num_workers > 0:
            print(f"Ignoring num_workers '{vectorisation}' for non-sequential parallisation '{parallelisation}'")
        if vectorisation == "vmap_local":
            return ParallelisationConfig(
                parallelisation=ParallelisationType.Sequential, 
                vectorisation=VectorisationType.LocalVMAP,
                vmap_batch_size=args.vmap_batch_size,
                num_workers=1
            )
        if vectorisation == "vmap_global":
            return ParallelisationConfig(
                parallelisation=ParallelisationType.Sequential,
                vectorisation=VectorisationType.GlobalVMAP,
                vmap_batch_size=args.vmap_batch_size,
                num_workers=1
        )
        if jax.device_count() == 1:
            print(f"Warning: Vectorisation is set to'{vectorisation}' but only 1 jax devices is available.")
        if vectorisation == "pmap":
            return ParallelisationConfig(
                parallelisation=ParallelisationType.Sequential,
                vectorisation=VectorisationType.PMAP,
                num_workers=num_workers or jax.device_count(),
            )
        if vectorisation == "smap_local":
            return ParallelisationConfig(
                parallelisation=ParallelisationType.Sequential,
                vectorisation=VectorisationType.LocalSMAP,
                num_workers=num_workers or jax.device_count(),
            )
        if vectorisation == "smap_global":
            return ParallelisationConfig(
                parallelisation=ParallelisationType.Sequential,
                vectorisation=VectorisationType.GlobalSMAP,
                num_workers=num_workers or jax.device_count(),
            )
    else:
        assert vectorisation in ("vmap_local", "vmap_global")
        vectorisation_type = VectorisationType.LocalVMAP if vectorisation == "vmap_local" else VectorisationType.GlobalVMAP
        if parallelisation == "cpu_multiprocess":
            return ParallelisationConfig(
                parallelisation=ParallelisationType.MultiProcessingCPU,
                vectorisation=vectorisation_type,
                num_workers=num_workers or os.cpu_count() or 1,
                cpu_affinity=args.cpu_affinity,
                force_task_order = args.force_task_order,
                vmap_batch_size=args.vmap_batch_size
            )
        else:
            assert parallelisation == "jax_devices"
            assert num_workers <= jax.device_count()
            return ParallelisationConfig(
                parallelisation=ParallelisationType.MultiThreadingJAXDevices,
                vectorisation=vectorisation_type,
                num_workers=num_workers or jax.device_count(),
                force_task_order = True,
                vmap_batch_size=args.vmap_batch_size
            )