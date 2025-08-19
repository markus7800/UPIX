import pickle
import sys
import torch

import pyro
import pyro.infer.mcmc as pyromcmc  # type: ignore
from torch.distributions import Normal, Uniform

from infer import run_inference, importance_resample, run_inference_icml2022
from ppl import ProbCtx, run_prob_prog

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

def target_NP_LA_DHMC(rep, bar_pos=None):
    # pick best from paper
    count = 1_000
    eps = 0.1
    L = 5
    alpha = 0.1
    K = 2
    run_inference_icml2022(
        lambda trace: run_prob_prog(walk_model, trace=trace),
        name=f"walk_model_{rep}",
        count=count,
        burnin=0,  # 100,
        eps=eps,
        L=L,
        K=K,
        alpha=alpha,
        seed=rep,
        bar_pos=bar_pos
    )

def target_NP_DHMC(rep, bar_pos=None):
    # pick from paper
    count = 1_000
    eps = 0.1
    num_steps = 50
    run_inference(
        lambda trace: run_prob_prog(walk_model, trace=trace),
        name=f"walk_model{rep}",
        count=count,
        burnin=100,
        eps=eps,
        leapfrog_steps=num_steps,
        seed=rep,
        bar_pos=bar_pos
    )

import argparse


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("algorithm", help="NP-LA-DHMC | NP-DHMC")
    parser.add_argument("repetitions", default=10, type=int)
    parser.add_argument("-n_processes", default=1, type=int)
    args = parser.parse_args()

    count = 1_000
    repetitions = args.repetitions

    assert len(sys.argv) > 1
    
    if args.algorithm == "NP-LA-DHMC":
        if args.n_processes > 1:
            processes = []
            with mp.Pool(args.n_processes) as p:
                p.starmap(target_NP_LA_DHMC, [(rep, rep) for rep in range(repetitions)])
        else:
            for rep in range(repetitions):
                print(
                    f"REPETITION {rep+1}/{repetitions}"
                )
                target_NP_LA_DHMC(rep)

    elif args.algorithm == "NP-DHMC":
        if args.n_processes > 1:
            # takes ~ 2:30s for 10 * 1_000 samples
            with mp.Pool(args.n_processes) as p:
                p.starmap(target_NP_DHMC, [(rep, rep % args.n_processes) for rep in range(repetitions)])
        else:
            for rep in range(repetitions):
                print(f"REPETITION {rep+1}/{repetitions}")
                target_NP_DHMC(rep)