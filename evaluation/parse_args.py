
import os
import argparse
from dccxjax.backend import set_host_device_count, set_platform

def parse_args_and_setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("parallelisation", default="sequential", help="sequential | cpu_multiprocess | jax_devices")
    parser.add_argument("vectorisation", default="vmap", help="vmap | pmap | smap | global_smap")
    parser.add_argument("-host_device_count", default=1, required=False)
    parser.add_argument("-num_workers", default=0, required=False)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    # print(args)
    
    if args.cpu:
        print("Force run on CPU.")
        set_platform("cpu")
        
    host_device_count = int(args.host_device_count)
    
    if host_device_count > 1:
        set_host_device_count(host_device_count)
        
    return args
        
    
    
    