import jax
import jax.numpy as jnp
from typing import Callable, Optional, Any, Set, Tuple, Dict
from .samplecontext import LogprobCtx, GenerateCtx, UnconstrainedLogprobCtx, TransformToUnconstrainedCtx, TransformToConstrainedCtx, CollectDistributionTypesCtx
from ..types import Trace, PRNGKey
from ..utils import maybe_jit_warning, to_shaped_arrays, to_shaped_array_trace
from .branching_tracer import BranchingDecisions, trace_branching, retrace_branching

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

def _make_slp_path_indicator(slp: "SLP",  model: Model, branching_decisions: BranchingDecisions) -> Callable[[Trace], bool]:
    @jax.jit
    def _path_indicator(X: Trace):
        maybe_jit_warning(slp, "_jitted_path_indicator", "_path_indicator", slp.short_repr(), to_shaped_array_trace(X))
        slp_model_logprob = retrace_branching(model.log_prob, branching_decisions)
        lp, path_condition = slp_model_logprob(X)
        return path_condition


    return _path_indicator

def _make_slp_log_prob(slp: "SLP", model: Model, branching_decisions: BranchingDecisions) -> Callable[[Trace], float]:
    @jax.jit
    def _log_prob(X: Trace):
        maybe_jit_warning(slp, "_jitted_log_prob", "_slp_log_prob", slp.short_repr(), to_shaped_array_trace(X))
                        
        slp_model_logprob = retrace_branching(model.log_prob, branching_decisions)
        # However, when transformed with vmap() to operate over a batch of predicates, cond is converted to select().
        # return jax.lax.cond(path_indicator(X), model_logprob, lambda _: -jnp.inf, X)
        lp, path_condition = slp_model_logprob(X)
        return jax.lax.select(path_condition, lp, jax.lax.full_like(lp, -jnp.inf))

    return _log_prob


def _make_slp_log_prob_and_grad(slp: "SLP") -> Callable[[Trace], Tuple[float,Trace]]:
    @jax.jit
    def _grad_log_prob(X: Trace):
        maybe_jit_warning(slp, "_jitted_grad_log_prob", "_slp_grad_log_prob", slp.short_repr(), to_shaped_array_trace(X))
        return jax.value_and_grad(slp._log_prob)(X)
    return _grad_log_prob

def _make_slp_unconstrained_log_prob(slp: "SLP", model: Model, branching_decisions: BranchingDecisions) -> Callable[[Trace], Tuple[float, Trace]]:
    @jax.jit
    def _unconstrained_log_prob(X_unconstrained: Trace):
        maybe_jit_warning(slp, "_jitted_unconstrained_log_prob", "_slp_unconstrained_log_prob", slp.short_repr(), to_shaped_array_trace(X_unconstrained))
        
        def _unconstrained_log_prob_with_ctx(_X_unconstrained: Trace):
            with UnconstrainedLogprobCtx(_X_unconstrained) as ctx:
                model()
                return ctx.log_prob, ctx.X_constrained

        traced_f = retrace_branching(_unconstrained_log_prob_with_ctx, branching_decisions)
        
        (log_prob, X_constrained), path_condition = traced_f(X_unconstrained)

        return jax.lax.select(path_condition, log_prob, jax.lax.full_like(log_prob, -jnp.inf)), X_constrained

    return _unconstrained_log_prob

# def _make_slp_transform_unconstrained_to_support(slp: "SLP", model: Model, branching_decisions: BranchingDecisions) -> Callable[[Trace], Trace]:
#     # this is not expected to be called often, and thus not jitted to save (?) compile time
#     # vmap does not require jit
#     def _transform_unconstrained_to_support(X_unconstrained: Trace):
#         def _transform_unconstrained_to_support_with_ctx(_X_unconstrained: Trace):
#             with TransformToUnconstrainedCtx(_X_unconstrained) as ctx:
#                 model()
#                 return ctx.X_unconstrained

#         traced_f = trace_branching(_transform_unconstrained_to_support_with_ctx, branching_decisions, retrace=True)
        
#         X_unconstrained = traced_f(X_unconstrained)

#         return X_unconstrained
    
#     return _transform_unconstrained_to_support

def _make_slp_gen_likelihood_weight(slp: "SLP", model: Model, branching_decisions: BranchingDecisions) -> Callable[[PRNGKey], float]:
    @jax.jit
    def _gen_likelihood_weight(key: PRNGKey):
        maybe_jit_warning(slp, "_jitted_gen_likelihood_weight", "slp_gen_likelihood_weight", slp.short_repr(), to_shaped_arrays(key))

        # cannot do @jax.jit here because model can do branching and has to be controlled by trace_branching transformation, before jitting
        def _gen_log_likelihood_and_X_from_prior(key: PRNGKey):
            with GenerateCtx(key) as ctx:
                model()
                return ctx.log_likelihood
                        
        gen_log_likelihood_and_X_from_prior = retrace_branching(_gen_log_likelihood_and_X_from_prior, branching_decisions)
        log_likelihood, path_condition = gen_log_likelihood_and_X_from_prior(key)
        
        return jax.lax.select(path_condition, log_likelihood, -jnp.inf)

    return _gen_likelihood_weight

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
        self._path_indicator = _make_slp_path_indicator(self, model, branching_decisions)

        self._jitted_log_prob = False
        self._log_prob = _make_slp_log_prob(self, model, branching_decisions)

        self._jitted_grad_log_prob = False
        self._grad_log_prob = _make_slp_log_prob_and_grad(self)

        self._jitted_gen_likelihood_weight = False
        self._gen_likelihood_weight = _make_slp_gen_likelihood_weight(self, model, branching_decisions)

        self._jitted_unconstrained_log_prob = False
        self._unconstrained_log_prob = _make_slp_unconstrained_log_prob(self, model, branching_decisions)

        # this is not jitted
        # self._transform_unconstrained_to_support = _make_slp_transform_unconstrained_to_support(self, model, branching_decisions)

        self.is_discrete_map: Optional[Dict[str,bool]] = None
        self._all_discrete: Optional[bool] = None
        self._all_continuous: Optional[bool] = None

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

    def path_indicator(self, X: Trace):
        if self.decision_representative.keys() != X.keys():
            return False
        else:
            return self._path_indicator(X)

    def log_prob(self, X: Trace):
        if self.decision_representative.keys() != X.keys():
            return -jnp.inf
        return self._log_prob(X)
    
    def unconstrained_log_prob(self, X_unconstrained: Trace) -> Tuple[float, Trace]:
        if self.decision_representative.keys() != X_unconstrained.keys():
            return -jnp.inf, self.transform_to_constrained(X_unconstrained)
        return self._unconstrained_log_prob(X_unconstrained)
    
    def transform_to_constrained(self, X_unconstrained: Trace) -> Trace:
        def _transform_to_constrained(X_unconstrained: Trace):
            with TransformToConstrainedCtx(X_unconstrained) as ctx:
                self.model()
                return ctx.X_constrained
        return retrace_branching(_transform_to_constrained, self.branching_decisions)(X_unconstrained)[0]

    def transform_to_unconstrained(self, X: Trace) -> Trace:
        def _transform_to_unconstrained(X: Trace):
            with TransformToUnconstrainedCtx(X) as ctx:
                self.model()
                return ctx.X_unconstrained
        return retrace_branching(_transform_to_unconstrained, self.branching_decisions)(X)[0]
        
    def get_is_discrete_map(self):
        if self.is_discrete_map is None:
            with CollectDistributionTypesCtx(self.decision_representative) as ctx:
                self.model()
                self.is_discrete_map = ctx.is_discrete
        return self.is_discrete_map
    
    def all_discrete(self):
        if self._all_discrete is None:
            self._all_discrete = all(b for _, b in self.get_is_discrete_map().items())
        return self._all_discrete
    
    def all_continuous(self):
        if self._all_continuous is None:
            self._all_continuous = not any(b for _, b in self.get_is_discrete_map().items())
        return self._all_continuous


def HumanReadableDecisionsFormatter():
    def _formatter(slp: SLP):
        return slp.branching_decisions.to_human_readable()
    return _formatter