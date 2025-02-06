from .samplecontext import GenerateCtx, LogprobCtx, Model
from .tracer import trace_branching, BranchingDecisions, SExpr, SVar, SConstant, SOp, BranchingTracer
import jax
import jax.numpy as jnp
from jax.core import full_lower
from typing import Dict, Callable, Set, NamedTuple
from .types import Trace, PRNGKey
from .utils import maybe_jit_warning, to_shaped_array_trace, to_shaped_arrays

def replace_constants_with_svars(constant_object_ids_to_name: Dict[int, str], sexpr: SExpr) -> SExpr:
    if isinstance(sexpr, SConstant):
        if isinstance(sexpr.constant, jax.Array) and id(sexpr.constant) in constant_object_ids_to_name:
            return SVar(constant_object_ids_to_name[id(sexpr.constant)])
        return sexpr
    elif isinstance(sexpr, SOp):
        sexpr.args = list(map(lambda arg: replace_constants_with_svars(constant_object_ids_to_name, arg), sexpr.args))
        return sexpr
    else:
        assert isinstance(sexpr, SVar)
        return sexpr
    
def replace_sexpr_with_svars(sexpr_object_ids_to_name: Dict[int, str], sexpr: SExpr) -> SExpr:
    if id(sexpr) in sexpr_object_ids_to_name:
        return SVar(sexpr_object_ids_to_name[id(sexpr)])
    if isinstance(sexpr, SConstant):
        return sexpr
    elif isinstance(sexpr, SOp):
        sexpr.args = list(map(lambda arg: replace_sexpr_with_svars(sexpr_object_ids_to_name, arg), sexpr.args))
        return sexpr
    else:
        assert isinstance(sexpr, SVar)
        return sexpr



class SLP:
    model: Model
    decision_representative: Trace
    branching_decisions: BranchingDecisions

    def __init__(self, model: Model, decision_representative: Trace, branching_decisions: BranchingDecisions) -> None:
        self.model = model
        self.decision_representative = decision_representative
        self.branching_decisions = branching_decisions
        self._jitted_path_indicator = False
        self._path_indicator = self.make_slp_path_indicator(branching_decisions)
        self._jitted_log_prob = False
        self._log_prob = self.make_slp_logprob(model, branching_decisions)
        self._jitted_gen_likelihood_weight = False
        self._gen_likelihood_weight = self.make_slp_gen_likelihood_weight(model, branching_decisions)

    def make_slp_path_indicator(self, branching_decisions: BranchingDecisions) -> Callable[[Trace], float]:
        @jax.jit
        def _path_indicator(X: Trace):
            maybe_jit_warning(self, "_jitted_path_indicator", "_path_indicator", self.short_repr(), to_shaped_array_trace(X))

            b = jnp.array(True)
            for sexpr, val in branching_decisions.decisions:
                b = b & (sexpr.eval(X) == val)
            return b
        return _path_indicator

    def make_slp_logprob(self, model: Model, branching_decisions: BranchingDecisions) -> Callable[[Trace], float]:
        @jax.jit
        def _log_prob(X: Trace):
            maybe_jit_warning(self, "_jitted_log_prob", "_slp_log_prob", self.short_repr(), to_shaped_array_trace(X))
                            
            slp_model_logprob = trace_branching(model.log_prob, branching_decisions, retrace=True)
            # However, when transformed with vmap() to operate over a batch of predicates, cond is converted to select().
            # return jax.lax.cond(path_indicator(X), model_logprob, lambda _: -jnp.inf, X)
            
            return jax.lax.select(self._path_indicator(X), slp_model_logprob(X), -jnp.inf)
   
        return _log_prob
    
    def make_slp_gen_likelihood_weight(self, model: Model, branching_decisions: BranchingDecisions) -> Callable[[PRNGKey], float]:
        @jax.jit
        def _gen_likelihood_weight(key: PRNGKey):
            maybe_jit_warning(self, "_jitted_gen_likelihood_weight", "slp_gen_likelihood_weight", self.short_repr(), to_shaped_arrays(key))

            # TODO: figure out why @jax.jit does not work here in combination with trace_branching
            def _gen_log_likelihood_and_X_from_prior(key: PRNGKey):
                with GenerateCtx(key) as ctx:
                    model()
                    return ctx.log_likelihood, ctx.X
                            
            gen_log_likelihood_and_X_from_prior = trace_branching(_gen_log_likelihood_and_X_from_prior, branching_decisions, retrace=True)
            log_likelihood, X = gen_log_likelihood_and_X_from_prior(key)
            
            return jax.lax.select(self._path_indicator(X), log_likelihood, -jnp.inf)
   
        return _gen_likelihood_weight

    def __repr__(self) -> str:
        s = "SLP {"
        s += "\n  " + repr(self.model)
        s += "\n  " + repr(self.decision_representative)
        s += "\n  " + repr(self.branching_decisions.decisions)
        s += "\n}"
        return s
    
    def short_repr(self) -> str:
        s = f"<SLP at {hex(id(self))}>"
        return s
    
    def path_indicator(self, X: Dict[str,jax.Array]):
        if self.decision_representative.keys() != X.keys():
            return -jnp.array(0)
        else:
            return self._path_indicator(X)

    
    def log_prob(self, X: Dict[str,jax.Array]):
        if self.decision_representative.keys() != X.keys():
            return -jnp.inf
        return self._log_prob(X)
    

def estimate_Z_for_SLP(slp: SLP, rng_keys: PRNGKey):
    lps = jax.vmap(slp._gen_likelihood_weight)(rng_keys)
    # print(lps)
    return jnp.mean(jnp.exp(lps))

def sample_from_prior(model: Model, rng_key: PRNGKey) -> Dict[str, jax.Array]:
    ctx = GenerateCtx(rng_key)
    with ctx:
        model()
    return ctx.X


def slp_from_X(model: Model, decision_representative: Dict[str, jax.Array]) -> SLP:
    def f(X: Dict[str, jax.Array]):
        with LogprobCtx(X) as ctx:
            model()
            return ctx.log_prob

    branching_decisions = BranchingDecisions()
    traced_f = trace_branching(f, branching_decisions)
    traced_f(decision_representative)

    sexpr_object_ids_to_name = {id(array): addr for addr, array in decision_representative.items()}
    branching_decisions.decisions = [(replace_constants_with_svars(sexpr_object_ids_to_name, sexpr), val) for sexpr, val in branching_decisions.decisions]


    return SLP(model, decision_representative, branching_decisions)


# assumes model has no branching
def convert_model_to_SLP(model: Model) -> SLP:
    X = sample_from_prior(model, jax.random.PRNGKey(0))
    slp = slp_from_X(model, X)
    assert len(slp.branching_decisions.decisions) == 0
    return slp