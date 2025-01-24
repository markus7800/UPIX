from .samplecontext import GenerateCtx, LogprobCtx, Model
from .tracer import trace_branching, BranchingDecisions, SExpr, SVar, SConstant, SOp, BranchingTracer
import jax
import jax.numpy as jnp
from jax.core import full_lower
from typing import Dict, Callable, Set, NamedTuple

def replace_sexpr_with_svars(sexpr_object_ids_to_name: Dict[int, str], sexpr: SExpr) -> SExpr:
    if id(sexpr) in sexpr_object_ids_to_name:
        return  SVar(sexpr_object_ids_to_name[id(sexpr)])
    if isinstance(sexpr, SConstant):
        return sexpr
    elif isinstance(sexpr, SOp):
        sexpr.args = list(map(lambda arg: replace_sexpr_with_svars(sexpr_object_ids_to_name, arg), sexpr.args))
        return sexpr
    else:
        assert isinstance(sexpr, SVar)
        return sexpr


class SLP(NamedTuple):
    model: Model
    decision_representative: Dict[str,jax.Array]
    branching_decisions: BranchingDecisions

    def __repr__(self) -> str:
        s = "SLP {"
        s += "\n  " + repr(self.model)
        s += "\n  " + repr(self.decision_representative)
        s += "\n  " + repr(self.branching_decisions.boolean_decisions)
        s += "\n  " + repr(self.branching_decisions.index_decisions)
        s += "\n}"
        return s
    
    def log_prob(self, X: Dict[str,jax.Array]):
        print("slp log_prob", X, self.decision_representative)
        if self.decision_representative.keys() != X.keys():
            return -jnp.inf
        
        @jax.jit
        def path_indicator(X: Dict[str, jax.Array]):
            print("compile path_indicator for", X)
            b = jnp.array(True)
            for sexpr, val in self.branching_decisions.boolean_decisions:
                b = b & (sexpr.eval(X) == val)
            for sexpr, val in self.branching_decisions.index_decisions:
                b = b & (sexpr.eval(X) == val)
            return b
    
        if path_indicator(X) == 0.:
            return -jnp.inf
        
        def model_logprob(X: Dict[str,jax.Array]):
            print("compile model_logprob for", X)
            with LogprobCtx(X) as ctx:
                self.model.f(*self.model.args, **self.model.kwargs)
                return ctx.log_prob
            
        return jax.jit(trace_branching(model_logprob, self.branching_decisions, retrace=True))(X)
    

def slp_from_prior(model: Model, rng_key: jax.Array):
    ctx = GenerateCtx(rng_key)
    decisions = BranchingDecisions()
    traced_f = trace_branching(model.f, decisions)
    with ctx:
        traced_f(*model.args, **model.kwargs)

    ctx_X: Dict[str, BranchingTracer] = ctx.X # type: ignore

    decision_representative = {addr: full_lower(val) for addr, val in ctx_X.items()}

    sexpr_object_ids_to_name = {id(tracer.sexpr): addr for addr, tracer in ctx_X.items()}
    decisions.boolean_decisions = [(replace_sexpr_with_svars(sexpr_object_ids_to_name, sexpr), val) for sexpr, val in decisions.boolean_decisions]
    decisions.index_decisions = [(replace_sexpr_with_svars(sexpr_object_ids_to_name, sexpr), val) for sexpr, val in decisions.index_decisions]
    # print(X)
    # print(decisions.boolean_decisions)
    # print(decisions.index_decisions)


    return SLP(model, decision_representative, decisions)
