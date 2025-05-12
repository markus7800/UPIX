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


if __name__ == "__main__":
    count = 1_000
    repetitions = 10

    assert len(sys.argv) > 1
    
    if sys.argv[1] == "NP-LA-DHMC":
        # configs = [
        #     (L, alpha, K, eps)
        #     for L in [5]
        #     for eps in [0.1]
        #     for alpha in [1.0, 0.5, 0.1]
        #     for K in [0, 1, 2]
        # ]
        # pick best from paper
        eps = 0.1
        L = 5
        alpha = 0.1
        K = 2
        for rep in range(repetitions):
            print(
                f"REPETITION {rep+1}/{repetitions} ({eps=}, {L=}, {alpha=}, {K=})"
            )
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
            )
    elif sys.argv[1] == "NP-DHMC":
        eps = 0.1
        num_steps = 50
        for rep in range(repetitions):
            print(f"REPETITION {rep+1}/{repetitions}")
            run_inference(
                lambda trace: run_prob_prog(walk_model, trace=trace),
                name=f"walk_model{rep}",
                count=count,
                burnin=100,
                eps=eps,
                leapfrog_steps=num_steps,
                seed=rep,
            )
            