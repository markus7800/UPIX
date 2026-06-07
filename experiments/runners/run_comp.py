
import subprocess
import argparse
from time import monotonic

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=["pedestrian", "gp-vi", "gp-smc", "gmm", "urn", "all", "dice"],  help="Model to run")
parser.add_argument("ncpu", type=int)
parser.add_argument("-repetitions", type=int, default=5)
parser.add_argument("--smoketest", action="store_true")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

smoketest = args.smoketest
repetitions = 1 if smoketest else args.repetitions
ncpu = args.ncpu
stdout_behavior = None if args.verbose else subprocess.DEVNULL
stderr_behavior = None if args.verbose else subprocess.DEVNULL
    
print(f"{smoketest=} {repetitions=} {ncpu=}")
    
def run_pedestrian_npdhmc():
    t0 = monotonic()
    print(f"{'Testing' if smoketest else 'Running'} Pedestrian NPDHMC ... ", end="" if smoketest else "\n", flush=True)
    
    d = 10 if smoketest else 1
    
    for rep in range(repetitions):
        cmd = [
            "uv", "run", 
            "-p", "python3.10", 
            "--no-project", 
            "--with-requirements=evaluation/pedestrian/nonparametric-hmc/requirements.txt", 
            "evaluation/pedestrian/nonparametric-hmc/pedestrian.py", 
            "NP-DHMC", str(ncpu), str(1000 // d), "100", 
            "-n_processes", str(ncpu), 
            "--disable_bar", 
            "-seed", str(rep)
        ]
        subprocess.run(cmd, stdout=stdout_behavior, stderr=stderr_behavior, check=True)
        if repetitions > 1: print(f"{rep+1}/{repetitions}")
    if smoketest: print("ok. ", end="")
    print(f"Finished in {monotonic()-t0:.3f}s")
        
def run_pedestrian_upix():
    t0 = monotonic()
    print(f"{'Testing' if smoketest else 'Running'} Pedestrian UPIX MCMC-DCC ... ", end="" if smoketest else "\n", flush=True)
    
    d = 10 if smoketest else 1
    
    for rep in range(repetitions):
        cmd = [
            "uv", "run", 
            "-p", "python3.13", 
            "--frozen", 
            "--extra=cpu", 
            "evaluation/pedestrian/run_comp.py", 
            "sequential", "pmap", 
            "-n_chains", str(ncpu),
            "-n_samples_per_chain", str(25_000 // d),
            "-host_device_count", str(ncpu), 
            "-seed", str(rep)
        ]
        subprocess.run(cmd, stdout=stdout_behavior, stderr=stderr_behavior, check=True)
        if repetitions > 1: print(f"{rep+1}/{repetitions}")
    if smoketest: print("ok. ", end="")
    print(f"Finished in {monotonic()-t0:.3f}s")

        
if args.model in ["pedestrian", "all"]:
    run_pedestrian_npdhmc()
    run_pedestrian_upix()
    
    
def run_gp_sdvi(): 
    t0 = monotonic()
    print(f"{'Testing' if smoketest else 'Running'} SDVI  ... ", end="" if smoketest else "\n", flush=True)

    d = 100 if smoketest else 1
    
    for rep in range(repetitions):
        cmd = [
            "bash", 
            "evaluation/gp/sdvi/run_comp.sh", 
            str(ncpu), 
            str(rep),
            str(1_000_000 // d)
        ]
        subprocess.run(cmd, stdout=stdout_behavior, stderr=stderr_behavior, check=True)
        if repetitions > 1: print(f"{rep+1}/{repetitions}")
    if smoketest: print("ok. ", end="")
    print(f"Finished in {monotonic()-t0:.3f}s")


def run_gp_vi_upix():
    t0 = monotonic()
    print(f"{'Testing' if smoketest else 'Running'} GP UPIX VI-DCC ... ", end="" if smoketest else "\n", flush=True)
    
    for rep in range(repetitions):
        cmd = [
            "uv", "run", 
            "-p", "python3.13", 
            "--frozen", 
            "--extra=cpu", 
            "--with", "pandas", 
            "evaluation/gp/run_comp_vi.py", 
            "cpu_multiprocess", "vmap_local",
            "-sh_iterations", str(1_000_000),
            "-num_workers", str(ncpu), 
            "-seed", str(rep)
        ]
        subprocess.run(cmd, stdout=stdout_behavior, stderr=stderr_behavior, check=True)
        if repetitions > 1: print(f"{rep+1}/{repetitions}")
    if smoketest: print("ok. ", end="")
    print(f"Finished in {monotonic()-t0:.3f}s")

if args.model in ["gp-vi", "all"]:
    run_gp_sdvi()
    run_gp_vi_upix()
    
    
    
def intall_julia_gen():
    t0 = monotonic()
    print(f"Installing Gen.jl (if needed) ... ", end="" if smoketest else "\n", flush=True)
    cmd = [
        "julia",
        "--project=evaluation/gmm/gen",
        "-e", "import Pkg; Pkg.instantiate()"
    ]
    subprocess.run(cmd, stdout=stdout_behavior, stderr=stderr_behavior, check=True)
    if smoketest: print("ok. ", end="")
    print(f"Finished in {monotonic()-t0:.3f}s")
    
def run_gmm_gen():
    t0 = monotonic()
    print(f"{'Testing' if smoketest else 'Running'} Gen RJMCMC ... ", end="" if smoketest else "\n", flush=True)
    
    d = 10 if smoketest else 1

    for rep in range(repetitions):
        cmd = [
            "julia",
            "-t", str(ncpu),
            "--project=evaluation/gmm/gen",
            "evaluation/gmm/gen/gmm.jl",
            str(ncpu), 
            str(25_000 // d),
            str(rep),
        ]
        subprocess.run(cmd, stdout=stdout_behavior, stderr=stderr_behavior, check=True)
        if repetitions > 1: print(f"{rep+1}/{repetitions}")
    if smoketest: print("ok. ", end="")
    print(f"Finished in {monotonic()-t0:.3f}s")


def run_gmm_upix():
    t0 = monotonic()
    print(f"{'Testing' if smoketest else 'Running'} GP UPIX RJ-DCC ... ", end="" if smoketest else "\n", flush=True)
    
    d = 10 if smoketest else 1

    for rep in range(repetitions):
        cmd = [
            "uv", "run", 
            "-p", "python3.13", 
            "--frozen", 
            "--extra=cpu", 
            "evaluation/gmm/run_comp.py", 
            "sequential", "pmap",
            "-n_chains", str(ncpu),
            "-n_samples_per_chain", str(25_000 // d),
            "-host_device_count", str(ncpu), 
            "-seed", str(rep)
        ]
        subprocess.run(cmd, stdout=stdout_behavior, stderr=stderr_behavior, check=True)
        if repetitions > 1: print(f"{rep+1}/{repetitions}")
    if smoketest: print("ok. ", end="")
    print(f"Finished in {monotonic()-t0:.3f}s")
    
    
if args.model in ["gmm", "all"]:
    intall_julia_gen()
    run_gmm_gen()
    run_gmm_upix()
    
    
    
    
def intall_julia_autogp():
    t0 = monotonic()
    print(f"Installing AutoGP.jl (if needed) ... ", end="" if smoketest else "\n", flush=True)
    cmd = [
        "julia",
        "--project=evaluation/gp/autogp",
        "-e", "import Pkg; Pkg.instantiate()"
    ]
    subprocess.run(cmd, stdout=stdout_behavior, stderr=stderr_behavior, check=True)
    if smoketest: print("ok. ", end="")
    print(f"Finished in {monotonic()-t0:.3f}s")
    
def run_gp_autogp():
    t0 = monotonic()
    print(f"{'Testing' if smoketest else 'Running'} AutoGP.jl ... ", end="" if smoketest else "\n", flush=True)
    
    d = 16 if smoketest else 1

    for rep in range(repetitions):
        cmd = [
            "julia",
            "-t", str(ncpu),
            "--project=evaluation/gp/autogp",
            "evaluation/gp/autogp/main.jl",
            str(128 // d), 
            "false",
            str(rep),
        ]
        subprocess.run(cmd, stdout=stdout_behavior, stderr=stderr_behavior, check=True)
        if repetitions > 1: print(f"{rep+1}/{repetitions}")
    if smoketest: print("ok. ", end="")
    print(f"Finished in {monotonic()-t0:.3f}s")


def run_gp_smc_upix():
    t0 = monotonic()
    print(f"{'Testing' if smoketest else 'Running'} GP UPIX SMC-DCC ... ", end="" if smoketest else "\n", flush=True)
    
    d = 16 if smoketest else 1

    for rep in range(repetitions):
        cmd = [
            "uv", "run", 
            "-p", "python3.13", 
            "--frozen", 
            "--extra=cpu",
            "--with=pandas",
            "evaluation/gp/run_comp_smc.py", 
            "sequential", "smap_local",
            "-n_particles", str(128 // d),
            "-host_device_count", str(ncpu), 
            "-seed", str(rep)
        ]
        subprocess.run(cmd, stdout=stdout_behavior, stderr=stderr_behavior, check=True)
        if repetitions > 1: print(f"{rep+1}/{repetitions}")
    if smoketest: print("ok. ", end="")
    print(f"Finished in {monotonic()-t0:.3f}s")
    
    
if args.model in ["gp-smc", "all"]:
    intall_julia_autogp()
    run_gp_autogp()
    run_gp_smc_upix()
    
    
def run_urn_dice():
    t0 = monotonic()
    print(f"{'Testing' if smoketest else 'Running'} Urn Dice ... ", end="" if smoketest else "\n", flush=True)
    
    for _ in range(repetitions):
        cmd = [
            "uv", "run", 
            "-p", "python3.13", 
            "--frozen", 
            "--extra=cpu",
            "--with=numpy",
            "evaluation/urn/dice/run.py", 
            "10" if smoketest else "19"
        ]
        subprocess.run(cmd, stdout=stdout_behavior, stderr=stderr_behavior, check=True)
    if smoketest: print("ok. ", end="")
    print(f"Finished in {monotonic()-t0:.3f}s")
    
if args.model in ["dice"]:
    run_urn_dice()
    
def run_urn_upix():
    t0 = monotonic()
    print(f"{'Testing' if smoketest else 'Running'} Urn UPIX VE-DCC ... ", end="" if smoketest else "\n", flush=True)
    
    d = 2 if smoketest else 1

    for rep in range(repetitions):
        cmd = [
            "uv", "run", 
            "-p", "python3.13", 
            "--frozen", 
            "--extra=cpu",
            "--with=pandas",
            "evaluation/urn/run_comp.py", 
            "sequential", "vmap_local",
            str(20 // d), 
            "--jit_inf",
            "-seed", str(rep)
        ]
        subprocess.run(cmd, stdout=stdout_behavior, stderr=stderr_behavior, check=True)
        if repetitions > 1: print(f"{rep+1}/{repetitions}")
    if smoketest: print("ok. ", end="")
    print(f"Finished in {monotonic()-t0:.3f}s")
    

if args.model in ["urn", "all"]:
    run_urn_upix()