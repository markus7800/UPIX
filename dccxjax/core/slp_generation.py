import jax
from typing import Set
from .model_slp import Model, SLP
from ..types import Trace, PRNGKey
from .samplecontext import LogprobCtx, GenerateCtx
from .branching_tracer import BranchingDecisions, trace_branching
from .sexpr import replace_constants_with_svars

__all__ = [
    "sample_from_prior",
    "slp_from_decision_representative",
    "convert_branchless_model_to_SLP",
]


def slp_from_decision_representative(model: Model, decision_representative: Trace) -> SLP:
    def f(X: Trace):
        with LogprobCtx(X) as ctx:
            model()
            return ctx.log_prob

    branching_decisions = trace_branching(f, decision_representative)

    sexpr_object_ids_to_name = {id(array): addr for addr, array in decision_representative.items()}
    branching_variables: Set[str] = set()
    branching_decisions.decisions = [(replace_constants_with_svars(sexpr_object_ids_to_name, sexpr, branching_variables), val) for sexpr, val in branching_decisions.decisions]


    return SLP(model, decision_representative, branching_decisions, branching_variables)


def sample_from_prior(model: Model, rng_key: PRNGKey) -> Trace:
    ctx = GenerateCtx(rng_key)
    with ctx:
        model()
    return ctx.X


# assumes model has no branching
def convert_branchless_model_to_SLP(model: Model) -> SLP:
    X = sample_from_prior(model, jax.random.PRNGKey(0))
    slp = slp_from_decision_representative(model, X)
    assert len(slp.branching_decisions.decisions) == 0
    return slp