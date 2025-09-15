
import argparse

def get_scale_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("platform", help="cpu | cuda")
    parser.add_argument("ndevices", type=int)
    parser.add_argument("minpow", type=int)
    parser.add_argument("maxpow", type=int)
    parser.add_argument("parallelisation")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--no_save", action="store_true")
    args = parser.parse_args()
    
    assert args.platform in ("cpu", "cuda")

    flags = ""
    if args.no_progress:
        flags += "--no_progress "
    if args.no_save:
        flags += "--no_save "

    platform = str(args.platform)
    ndevices = int(args.ndevices)
    minpow = int(args.minpow)
    maxpow = int(args.maxpow)
    parallelisation = str(args.parallelisation)
    
    return platform, ndevices, minpow, maxpow, parallelisation, flags