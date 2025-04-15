from dccxjax.core import SLP
from .variable_selector import VariableSelector
from typing import Set, Optional, Tuple, Callable
import jax
from ..types import Trace, FloatArray, BoolArray
import jax.numpy as jnp

# def make_gibbs_log_prior_likeli_pathcond(slp: SLP, conditional_variables: Set[str]) -> Callable[[Trace,Trace], Tuple[FloatArray,FloatArray,BoolArray]]:
#     def _gibbs_log_prior_likeli_pathcond(X: Trace, Y: Trace):
#         for address in conditional_variables:
#             X[address] = Y[address]
#         return slp._log_prior_likeli_pathcond(X)
#     return _gibbs_log_prior_likeli_pathcond

class GibbsModel:
    def __init__(self, slp: SLP, variable_selector: VariableSelector, Y: Optional[Trace] = None) -> None:
        self.slp = slp
        self.variables: Set[str] = set()
        self.conditional_variables: Set[str] = set()
        for address in slp.decision_representative.keys():
            if variable_selector.contains(address):
                self.variables.add(address)
            else:
                self.conditional_variables.add(address)
        # TODO: check if jit is slower or faster compilation here
        # self._gibbs_log_prior_likeli_pathcond = make_gibbs_log_prior_likeli_pathcond(slp, self.conditional_variables)
        if Y is not None:
            assert Y.keys() == self.conditional_variables
            self.Y = Y
        else:
            self.Y = {address: slp.decision_representative[address] for address in self.conditional_variables}

    def split_trace(self, t: Trace) -> Tuple[Trace,Trace]:
        X: Trace = {address: t[address] for address in self.variables}
        Y: Trace = {address: t[address] for address in self.conditional_variables}
        return (X, Y)
    
    def combine_to_trace(self, X: Trace, Y: Trace) -> Trace:
        assert X.keys() == self.variables
        assert Y.keys() == self.conditional_variables
        return X | Y

    def set_Y(self, Y: Trace):
        assert Y.keys() == self.conditional_variables
        self.Y = Y

    def log_prior_likeli_pathcond(self, X: Trace) -> Tuple[FloatArray,FloatArray,BoolArray]:
        assert X.keys() == self.variables
        return self.slp._log_prior_likeli_pathcond(X | self.Y)
    
    def tempered_log_prob(self, temperature: FloatArray) -> Callable[[Trace], FloatArray]:
        def _log_prob(X: Trace) -> FloatArray:
            assert X.keys() == self.variables
            log_prior, log_likelihood, path_condition = self.log_prior_likeli_pathcond(X)
            return jax.lax.select(path_condition, log_prior + temperature * log_likelihood, -jnp.inf)
        return _log_prob

    