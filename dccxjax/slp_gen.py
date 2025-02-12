from .samplecontext import GenerateCtx, LogprobCtx, Model, SampleContext
from .tracer import trace_branching, BranchingDecisions, SExpr, SVar, SConstant, SOp, BranchingTracer
import jax
import jax.numpy as jnp
from jax.core import full_lower
from typing import Dict, Callable, Set, NamedTuple, Tuple, Optional
from .types import Trace, PRNGKey
from .utils import maybe_jit_warning, to_shaped_array_trace, to_shaped_arrays
from jax.flatten_util import ravel_pytree
import numpyro.distributions as dist

def replace_constants_with_svars(constant_object_ids_to_name: Dict[int, str], sexpr: SExpr, variables: Set[str]) -> SExpr:
    if isinstance(sexpr, SConstant):
        if isinstance(sexpr.constant, jax.Array) and id(sexpr.constant) in constant_object_ids_to_name:
            svar = SVar(constant_object_ids_to_name[id(sexpr.constant)])
            variables.add(svar.name)
            return svar
        return sexpr
    elif isinstance(sexpr, SOp):
        sexpr.args = list(map(lambda arg: replace_constants_with_svars(constant_object_ids_to_name, arg, variables), sexpr.args))
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
    branching_variables: Set[str]

    def __init__(self, model: Model, decision_representative: Trace, branching_decisions: BranchingDecisions, branching_variables: Set[str]) -> None:
        self.model = model
        self.decision_representative = decision_representative
        self.branching_decisions = branching_decisions
        self.branching_variables = branching_variables
        self._jitted_path_indicator = False
        self._path_indicator = self.make_slp_path_indicator(branching_decisions)
        self._jitted_log_prob = False
        self._log_prob = self.make_slp_log_prob(model, branching_decisions)
        self._jitted_grad_log_prob = False
        self._grad_log_prob = self.make_slp_log_prob_and_grad()
        self._jitted_gen_likelihood_weight = False
        self._gen_likelihood_weight = self.make_slp_gen_likelihood_weight(model, branching_decisions)

    def make_slp_path_indicator(self, branching_decisions: BranchingDecisions) -> Callable[[Trace], bool]:
        @jax.jit
        def _path_indicator(X: Trace):
            maybe_jit_warning(self, "_jitted_path_indicator", "_path_indicator", self.short_repr(), to_shaped_array_trace(X))

            b = jnp.array(True)
            for sexpr, val in branching_decisions.decisions:
                b = b & (sexpr.eval(X) == val)
            return b
        return _path_indicator

    def make_slp_log_prob(self, model: Model, branching_decisions: BranchingDecisions) -> Callable[[Trace], float]:
        @jax.jit
        def _log_prob(X: Trace):
            maybe_jit_warning(self, "_jitted_log_prob", "_slp_log_prob", self.short_repr(), to_shaped_array_trace(X))
                            
            slp_model_logprob = trace_branching(model.log_prob, branching_decisions, retrace=True)
            # However, when transformed with vmap() to operate over a batch of predicates, cond is converted to select().
            # return jax.lax.cond(path_indicator(X), model_logprob, lambda _: -jnp.inf, X)
            lp = slp_model_logprob(X)
            return jax.lax.select(self._path_indicator(X), lp, jax.lax.full_like(lp, -jnp.inf))
   
        return _log_prob
    
    def make_slp_log_prob_and_grad(self) -> Callable[[Trace], Tuple[float,Trace]]:
        @jax.jit
        def _grad_log_prob(X: Trace):
            maybe_jit_warning(self, "_jitted_grad_log_prob", "_slp_grad_log_prob", self.short_repr(), to_shaped_array_trace(X))
            return jax.value_and_grad(self._log_prob)(X)
        return _grad_log_prob
    
    def make_slp_gen_likelihood_weight(self, model: Model, branching_decisions: BranchingDecisions) -> Callable[[PRNGKey], float]:
        @jax.jit
        def _gen_likelihood_weight(key: PRNGKey):
            maybe_jit_warning(self, "_jitted_gen_likelihood_weight", "slp_gen_likelihood_weight", self.short_repr(), to_shaped_arrays(key))

            # cannot do @jax.jit here because model can do branching and has to be controlled by trace_branching transformation, before jitting
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
            return False
        else:
            return self._path_indicator(X)

    
    def log_prob(self, X: Dict[str,jax.Array]):
        if self.decision_representative.keys() != X.keys():
            return -jnp.inf
        return self._log_prob(X)
    

def estimate_Z_for_SLP_from_prior(slp: SLP, N: int, rng_key: PRNGKey):
    rng_keys = jax.random.split(rng_key, N)
    weights = jax.vmap(slp._gen_likelihood_weight)(rng_keys)
    return jnp.mean(jnp.exp(weights))

def estimate_Z_for_SLP_from_mcmc(slp: SLP, scale: float, samples_per_trace: int, seed: PRNGKey, Xs: Trace):
    @jax.jit
    def _log_IS_weight(rng_key: PRNGKey, X: Trace):
        X_flat, unravel_fn = ravel_pytree(X)
        Q = dist.Normal(X_flat, scale) # type: ignore
        X_prime_flat = Q.sample(rng_key)
        X_prime = unravel_fn(X_prime_flat)
        return slp._log_prob(X_prime) - Q.log_prob(X_prime_flat).sum()
    
    _, some_entry = next(iter(Xs.items()))
    N = some_entry.shape[0]
    @jax.jit
    def _weight_sum_for_Xs(rng_key: PRNGKey):
        rng_keys = jax.random.split(rng_key, N)
        return jnp.sum(jnp.exp(jax.vmap(_log_IS_weight)(rng_keys, Xs)))
                       
    weights = jax.vmap(_weight_sum_for_Xs)(jax.random.split(seed, samples_per_trace))
    return jnp.sum(weights) / (N * samples_per_trace)
    

class DecisionRepresentativeCtx(SampleContext):
    def __init__(self, partial_X: Trace, rng_key: PRNGKey) -> None:
        self.rng_key = rng_key
        self.partial_X = partial_X
        self.X: Trace = dict()
    def sample(self, address: str, distribution: dist.Distribution, observed: Optional[jax.Array] = None) -> jax.Array:
        if observed is not None:
            return observed
        if address in self.partial_X:
            value = self.partial_X[address]
        else:
            self.rng_key, sample_key = jax.random.split(self.rng_key)
            value = distribution.sample(sample_key)
        self.X[address] = value
        return value
    
def decision_representative_from_partial_trace(model: Model, partial_X: Trace, rng_key: PRNGKey):
    with DecisionRepresentativeCtx(partial_X, rng_key) as ctx:
        model()
        return ctx.X


def sample_from_prior(model: Model, rng_key: PRNGKey) -> Trace:
    ctx = GenerateCtx(rng_key)
    with ctx:
        model()
    return ctx.X


def slp_from_decision_representative(model: Model, decision_representative: Trace) -> SLP:
    def f(X: Trace):
        with LogprobCtx(X) as ctx:
            model()
            return ctx.log_prob

    branching_decisions = BranchingDecisions()
    traced_f = trace_branching(f, branching_decisions)
    traced_f(decision_representative)

    sexpr_object_ids_to_name = {id(array): addr for addr, array in decision_representative.items()}
    branching_variables: Set[str] = set()
    branching_decisions.decisions = [(replace_constants_with_svars(sexpr_object_ids_to_name, sexpr, branching_variables), val) for sexpr, val in branching_decisions.decisions]


    return SLP(model, decision_representative, branching_decisions, branching_variables)


# assumes model has no branching
def convert_model_to_SLP(model: Model) -> SLP:
    X = sample_from_prior(model, jax.random.PRNGKey(0))
    slp = slp_from_decision_representative(model, X)
    assert len(slp.branching_decisions.decisions) == 0
    return slp