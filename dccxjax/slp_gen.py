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
    decision_representative: Dict[str,jax.Array]
    branching_decisions: BranchingDecisions
    path_indicator: Callable
    log_prob: Callable

    def __repr__(self) -> str:
        s = "SLP {"
        s += "\n  " + repr(self.decision_representative)
        s += "\n  " + repr(self.branching_decisions.boolean_decisions)
        s += "\n  " + repr(self.branching_decisions.index_decisions)
        s += "\n}"
        return s

def slp_from_prior(model: Model, rng_key: jax.Array):
    ctx = GenerateCtx(rng_key)
    decisions = BranchingDecisions()
    traced_f = trace_branching(model.f, decisions)
    with ctx:
        traced_f(*model.args, **model.kwargs)

    ctx_X: Dict[str, BranchingTracer] = ctx.X # type: ignore
    addresses = set(ctx_X.keys())

    decision_representative = {addr: full_lower(val) for addr, val in ctx_X.items()}

    sexpr_object_ids_to_name = {id(tracer.sexpr): addr for addr, tracer in ctx_X.items()}
    decisions.boolean_decisions = {replace_sexpr_with_svars(sexpr_object_ids_to_name, sexpr): val for sexpr, val in decisions.boolean_decisions.items()}
    decisions.index_decisions = {replace_sexpr_with_svars(sexpr_object_ids_to_name, sexpr): val for sexpr, val in decisions.index_decisions.items()}
    # print(X)
    # print(decisions.boolean_decisions)
    # print(decisions.index_decisions)

       


    @jax.jit
    def path_indicator(X: Dict[str, jax.Array]):
        print("compile path_indicator for", X)
        b = jnp.array(False)
        for sexpr, val in decisions.boolean_decisions.items():
            b = b & (sexpr.eval(X) == val)
        for sexpr, val in decisions.index_decisions.items():
            b = b & (sexpr.eval(X) == val)
        return b
    
    print(f"{decision_representative=}")
    path_indicator(decision_representative)

    print()
    print()
    

    
    @jax.jit
    def log_prob(X: Dict[str, jax.Array]):
        print("compile log_prob for", X)
        decisions.object_id_to_sym_name = {id(val): addr for addr, val in X.items()}
        print("object_id_to_sym_name", decisions.object_id_to_sym_name)
        retraced_f = trace_branching(model.f, decisions, retrace=True)
        ctx = LogprobCtx(X)
        with ctx:
            retraced_f(*model.args, **model.kwargs)
            # model()
        return ctx.log_prob
    
    log_prob(decision_representative)

    return SLP(decision_representative, decisions, path_indicator, log_prob)
