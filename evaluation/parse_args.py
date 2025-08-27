
import os
import argparse
from dccxjax.backend import set_host_device_count, set_platform

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("parallelisation", default="sequential", help="sequential | cpu_multiprocess | jax_devices")
    parser.add_argument("vectorisation", default="vmap", help="vmap_local | vmap_global | pmap | smap_glibal | smap_local")
    parser.add_argument("-host_device_count", default=1, type=int, required=False)
    parser.add_argument("-num_workers", default=0, type=int, required=False)
    parser.add_argument("-vmap_batch_size", default=0, type=int, required=False)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--cpu_affinity", action="store_true")
    parser.add_argument("-omp", default=0, type=int, help="Number of OMP threads to use")
    parser.add_argument("--force_task_order", action="store_true")

    return parser

def setup_devices_from_args(args):
    if args.omp > 0:
         os.environ["OMP_NUM_THREADS"] = str(args.omp)

    if args.cpu:
        print("Force run on CPU.")
        set_platform("cpu")
        
    host_device_count = int(args.host_device_count)
    
    if host_device_count > 1:
        set_host_device_count(host_device_count)
        
    return args
        
    
    
    