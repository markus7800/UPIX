import jax
from typing import Set, Tuple
from .model_slp import Model, SLP
from ..types import Trace, PRNGKey
from .samplecontext import LogprobCtx, GenerateCtx
from .branching_tracer import BranchingDecisions, trace_branching
from .sexpr import replace_constants_with_svars

__all__ = [
    "sample_from_prior",
    "sample_from_prior_with_decisions",
    "slp_from_decision_representative",
    "convert_branchless_model_to_SLP",
]


def slp_from_decision_representative(model: Model, decision_representative: Trace) -> SLP:
    def f(X: Trace):
        with LogprobCtx(X) as ctx:
            model()
            return ctx.log_prob

    _, branching_decisions = trace_branching(f, decision_representative)

    return SLP(model, decision_representative, branching_decisions)


def sample_from_prior(model: Model, rng_key: PRNGKey) -> Trace:
    ctx = GenerateCtx(rng_key)
    with ctx:
        model()
    return ctx.X

def sample_from_prior_with_decisions(model: Model, rng_key: PRNGKey) -> Tuple[Trace,BranchingDecisions]:
    def f(rng_key: PRNGKey):
        ctx = GenerateCtx(rng_key)
        with ctx:
            model()
        return ctx.X
    return trace_branching(f, rng_key)
    

# assumes model has no branching
def convert_branchless_model_to_SLP(model: Model) -> SLP:
    X = sample_from_prior(model, jax.random.PRNGKey(0))
    slp = slp_from_decision_representative(model, X)
    assert len(slp.branching_decisions.decisions) == 0
    return slp