import pickle
import sys
import torch
import os

import pyro
import pyro.infer.mcmc as pyromcmc  # type: ignore
from torch.distributions import Normal, Uniform

from infer import run_inference, importance_resample, run_inference_icml2022
from ppl import ProbCtx, run_prob_prog
from time import monotonic

distance_limit = 10


def walk_model(ctx: ProbCtx) -> float:
    """Random walk model.

    Mak et al. (2020): Densities of almost-surely terminating probabilistic programs are differentiable almost everywhere.
    """
    distance = torch.tensor(0.0, requires_grad=True)
    start = ctx.sample(Uniform(0, 3), is_cont=False)
    position = start
    while position > 0 and distance < distance_limit:
        step = ctx.sample(Uniform(-1, 1), is_cont=False)
        distance = distance + torch.abs(step)
        position = position + step
    ctx.observe(distance, Normal(1.1, 0.1))
    return start.item()


def pyro_walk_model() -> float:
    """The same model written in Pyro."""
    start = pyro.sample("start", pyro.distributions.Uniform(0, 3))
    t = 0
    position = start
    distance = torch.tensor(0.0)
    while position > 0 and position < distance_limit:
        step = pyro.sample(f"step_{t}", pyro.distributions.Uniform(-1, 1))
        distance = distance + torch.abs(step)
        position = position + step
        t = t + 1
    pyro.sample("obs", pyro.distributions.Normal(1.1, 0.1), obs=distance)
    return start.item()

import torch.multiprocessing as mp

def target_NP_LA_DHMC(rep, burnin, count, bar_pos=None, disable_bar=False):
    # pick best from paper
    eps = 0.1
    L = 5
    alpha = 0.1
    K = 2
    run_inference_icml2022(
        lambda trace: run_prob_prog(walk_model, trace=trace),
        name=f"walk_model_{rep}",
        count=count,
        burnin=burnin,
        eps=eps,
        L=L,
        K=K,
        alpha=alpha,
        seed=rep,
        bar_pos=bar_pos,
        disable_bar=disable_bar,
        save_samples=False
    )

def target_NP_DHMC(rep, burnin, count, bar_pos=None, disable_bar=False):
    # pick from paper
    eps = 0.1
    num_steps = 50
    run_inference(
        lambda trace: run_prob_prog(walk_model, trace=trace),
        name=f"walk_model{rep}",
        count=count,
        burnin=burnin,
        eps=eps,
        leapfrog_steps=num_steps,
        seed=rep,
        bar_pos=bar_pos,
        disable_bar=disable_bar,
        save_samples=False
    )

import argparse
from tqdm import tqdm

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("algorithm", help="NP-LA-DHMC | NP-DHMC")
    parser.add_argument("repetitions", default=10, type=int)
    parser.add_argument("n_iter", default=1000, type=int)
    parser.add_argument("burnin", default=100, type=int)
    parser.add_argument("-n_processes", default=1, type=int)
    parser.add_argument("--disable_bar", action="store_true")
    args = parser.parse_args()

    n_iter = args.n_iter
    burnin = args.burnin
    repetitions = args.repetitions
    disable_bar=args.disable_bar

    assert args.n_processes <= repetitions
    
    t0 = monotonic()
    
    if args.algorithm == "NP-LA-DHMC":
        if args.n_processes > 1:
            processes = []
            with mp.Pool(args.n_processes) as p:
                p.starmap(target_NP_LA_DHMC, [(rep, burnin, n_iter, rep % args.n_processes, disable_bar) for rep in range(repetitions)])
        else:
            for rep in range(repetitions):
                print(
                    f"REPETITION {rep+1}/{repetitions}"
                )
                target_NP_LA_DHMC(rep, n_iter, burnin, rep)

    elif args.algorithm == "NP-DHMC":
        if args.n_processes > 1:
            # takes ~ 2:30s for 10 * 1_000 samples
            with mp.Pool(args.n_processes) as p:
                p.starmap(target_NP_DHMC, [(rep, n_iter, burnin, rep % args.n_processes, disable_bar) for rep in range(repetitions)])
        else:
            for rep in range(repetitions):
                print(f"REPETITION {rep+1}/{repetitions}")
                target_NP_DHMC(rep, n_iter, burnin, rep)
                
    inference_time = monotonic() - t0
    tqdm.write(f"\nFinished in {inference_time:.3f}s.")
    
    import pathlib, json, uuid
    from datetime import datetime
    import cpuinfo
    
    def get_cpu_count() -> int:
        if hasattr(os, "sched_getaffinity"):
            return int(len(os.sched_getaffinity(0))) # type: ignore
        else:
            return int(os.cpu_count()) # type: ignore
    def _get_last_git_commit() -> str:
        try:
            return subprocess.check_output(['git', 'log',  '--format=%H', '-n', '1']).decode().rstrip()
        except:
            return ""
    
    platform = "cpu"
    num_workers = args.n_processes
    id_str = str(uuid.uuid4())
    json_result = {
        "id": id_str,
        "workload": {
            "n_chains": repetitions,
            "n_samples_per_chain": n_iter,
            "burnin": burnin
        },
        "timings": {
            "inference_time": inference_time
        },
        "environment_info": {
            "platform": "cpu",
            "cpu-brand": cpuinfo.get_cpu_info()["brand_raw"],
            "cpu_count": get_cpu_count(),
            "git_commit": _get_last_git_commit(),
            "command": sys.argv[0]
        }
    }
    now = datetime.today().strftime('%Y-%m-%d_%H-%M')
    prefix = "npdhmc" if args.algorithm == "NP-DHMC" else "npladhmc"
    fpath = pathlib.Path(
        "experiments", "data", "pedestrian", "comp",
        f"npdhmc_{repetitions}_{platform}_{num_workers:02d}_date_{now}_{id_str[:8]}.json")
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "w") as f:
        json.dump(json_result, f, indent=2)
    