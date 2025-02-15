import jax
import jax.numpy as jnp
from typing import Callable, Optional, Any, Set, Tuple, Dict
from .samplecontext import LogprobCtx, GenerateCtx
from ..types import Trace, PRNGKey
from ..utils import maybe_jit_warning, to_shaped_arrays, to_shaped_array_trace
from .branching_tracer import BranchingDecisions, trace_branching

__all__ = [
    "Model",
    "model",
    "SLP",
    "HumanReadableDecisionsFormatter"
]

class Model:
    def __init__(self, f: Callable, args, kwargs) -> None:
        self.f = f
        self.args = args
        self.kwargs = kwargs

        self._jitted_log_prob = False
        self.slp_formatter: Optional[Callable[["SLP"],str]] = None
        self.slp_sort_key: Optional[Callable[["SLP"], Any]] = None
        # self._log_prob = self.make_model_logprob()

    def __call__(self) -> Any:
        return self.f(*self.args, **self.kwargs)
    
    # not jitted
    def log_prob(self, X: Trace) -> float:
        with LogprobCtx(X) as ctx:
            self.f(*self.args, **self.kwargs)
            return ctx.log_prob

    def __repr__(self) -> str:
        return f"Model({self.f.__name__}, {self.args}, {self.kwargs})"
    
    def short_repr(self) -> str:
        return f"Model({self.f.__name__} at {hex(id(self))})"
    
    def set_slp_formatter(self, formatter: Callable[["SLP"],str]):
        self.slp_formatter = formatter

    def set_slp_sort_key(self, key: Callable[["SLP"], Any]):
        self.slp_sort_key = key

def model(f:Callable):
    def _f(*args, **kwargs):
        return Model(f, args, kwargs)
    return _f

class SLP:
    model: Model
    decision_representative: Trace
    branching_decisions: BranchingDecisions
    branching_variables: Set[str]

    def __init__(self,
                 model: Model,
                 decision_representative: Trace,
                 branching_decisions: BranchingDecisions,
                 branching_variables: Set[str]) -> None:
        
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
    
    def formatted(self) -> str:
        if self.model.slp_formatter is not None:
            return self.model.slp_formatter(self)
        else:
            return self.short_repr()
        
    def sort_key(self):
        if self.model.slp_sort_key is not None:
            return self.model.slp_sort_key(self)
        else:
            return self.short_repr()

    def path_indicator(self, X: Dict[str,jax.Array]):
        if self.decision_representative.keys() != X.keys():
            return False
        else:
            return self._path_indicator(X)

    
    def log_prob(self, X: Dict[str,jax.Array]):
        if self.decision_representative.keys() != X.keys():
            return -jnp.inf
        return self._log_prob(X)
    

def HumanReadableDecisionsFormatter():
    def _formatter(slp: SLP):
        return slp.branching_decisions.to_human_readable()
    return _formatter