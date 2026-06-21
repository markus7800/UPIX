import jax
from typing import Set, Tuple
from upix.types import Trace, PRNGKey
from upix.core.model_slp import Model, SLP
from upix.core.samplecontext import LogprobCtx, GenerateCtx
from upix.core.concretize_tracer import Decisions, track_decisions

__all__ = [
    "sample_from_prior",
    "sample_from_prior_with_decisions",
    "slp_from_decision_representative",
    "SLP_from_branchless_model",
]


def slp_from_decision_representative(model: Model, decision_representative: Trace) -> SLP:
    def f(X: Trace):
        with LogprobCtx(X) as ctx:
            model()
            return ctx.log_prior + ctx.log_likelihood

    _, decisions = track_decisions(f)(decision_representative)

    return SLP(model, decision_representative, decisions)


def sample_from_prior(model: Model, rng_key: PRNGKey) -> Trace:
    ctx = GenerateCtx(rng_key)
    with ctx:
        model()
    return ctx.X

def sample_from_prior_with_decisions(model: Model, rng_key: PRNGKey) -> Tuple[Trace,Decisions]:
    def f(rng_key: PRNGKey):
        ctx = GenerateCtx(rng_key)
        with ctx:
            model()
        return ctx.X
    return track_decisions(f)(rng_key)
    

# assumes model has no branching
def SLP_from_branchless_model(model: Model) -> SLP:
    X = sample_from_prior(model, jax.random.key(0))
    slp = slp_from_decision_representative(model, X)
    assert len(slp.decisions) == 0
    return slp