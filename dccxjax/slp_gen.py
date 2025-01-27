from .samplecontext import GenerateCtx, LogprobCtx, Model
from .tracer import trace_branching, BranchingDecisions, SExpr, SVar, SConstant, SOp, BranchingTracer
import jax
import jax.numpy as jnp
from jax.core import full_lower
from typing import Dict, Callable, Set, NamedTuple

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
    

def make_model_logprob(model: Model) -> Callable:
    def _model_logprob(X: Dict[str,jax.Array]):
        with LogprobCtx(X) as ctx:
            model.f(*model.args, **model.kwargs)
            return ctx.log_prob
    return _model_logprob
    
def make_slp_path_indicator(branching_decisions: BranchingDecisions) -> Callable:
    def _path_indicator(X: Dict[str, jax.Array]):
        print("compile _path_indicator for", X)
        b = jnp.array(True)
        for sexpr, val in branching_decisions.boolean_decisions:
            b = b & (sexpr.eval(X) == val)
        for sexpr, val in  branching_decisions.index_decisions:
            b = b & (sexpr.eval(X) == val)
        return b
    return _path_indicator

def make_slp_logprob(model: Model, branching_decisions: BranchingDecisions) -> Callable:
    def _log_prob(X: Dict[str,jax.Array]):
        print("compile _log_prob for", X)
        
        path_indicator = make_slp_path_indicator(branching_decisions)
        model_logprob = make_model_logprob(model)
            
        slp_model_logprob = trace_branching(model_logprob, branching_decisions, retrace=True)
        # However, when transformed with vmap() to operate over a batch of predicates, cond is converted to select().
        # return jax.lax.cond(path_indicator(X), model_logprob, lambda _: -jnp.inf, X)
        
        return jax.lax.select(path_indicator(X), slp_model_logprob(X), -jnp.inf)

        
    return _log_prob

class SLP:
    model: Model
    decision_representative: Dict[str,jax.Array]
    branching_decisions: BranchingDecisions

    def __init__(self, model: Model, decision_representative: Dict[str,jax.Array], branching_decisions: BranchingDecisions) -> None:
        self.model = model
        self.decision_representative = decision_representative
        self.branching_decisions = branching_decisions
        self._path_indicator = jax.jit(make_slp_path_indicator(branching_decisions))
        self._log_prob = jax.jit(make_slp_logprob(model, branching_decisions))

    def __repr__(self) -> str:
        s = "SLP {"
        s += "\n  " + repr(self.model)
        s += "\n  " + repr(self.decision_representative)
        s += "\n  " + repr(self.branching_decisions.boolean_decisions)
        s += "\n  " + repr(self.branching_decisions.index_decisions)
        s += "\n}"
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
    

def sample_from_prior(model: Model, rng_key: jax.Array) -> Dict[str, jax.Array]:
    ctx = GenerateCtx(rng_key)
    with ctx:
        model()
    return ctx.X


def slp_from_X(model: Model, decision_representative: Dict[str, jax.Array]):
    def f(X: Dict[str, jax.Array]):
        ctx = LogprobCtx(X)
        with ctx:
            model()

    decisions = BranchingDecisions()
    traced_f = trace_branching(f, decisions)
    traced_f(decision_representative)

    sexpr_object_ids_to_name = {id(array): addr for addr, array in decision_representative.items()}
    decisions.boolean_decisions = [(replace_constants_with_svars(sexpr_object_ids_to_name, sexpr), val) for sexpr, val in decisions.boolean_decisions]
    decisions.index_decisions = [(replace_constants_with_svars(sexpr_object_ids_to_name, sexpr), val) for sexpr, val in decisions.index_decisions]
    # print(X)
    # print(decisions.boolean_decisions)
    # print(decisions.index_decisions)


    return SLP(model, decision_representative, decisions)
