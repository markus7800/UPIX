from upix.core.model_slp import SLP, AnnealingMask
from upix.infer.variable_selector import VariableSelector
from typing import Set, Optional, Tuple, Callable, Dict
import jax
from upix.types import Trace, FloatArray, BoolArray
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

__all__ = [
    
]

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
            self.Y: Trace = Y
        else:
            self.Y: Trace = {address: slp.decision_representative[address] for address in self.conditional_variables}
        self.X_representative: Trace = {address: value for address, value in slp.decision_representative.items() if address not in self.conditional_variables}

    def split_trace(self, t: Trace) -> Tuple[Trace,Trace]:
        X: Trace = {address: t[address] for address in self.variables}
        Y: Trace = {address: t[address] for address in self.conditional_variables}
        return (X, Y)
    
    def combine_to_trace(self, X: Trace, Y: Trace) -> Trace:
        self._check_variables(X)
        assert Y.keys() == self.conditional_variables
        return X | Y

    def set_Y(self, Y: Trace):
        assert Y.keys() == self.conditional_variables
        self.Y = Y

    def _check_variables(self, X: Trace):
        assert X.keys() == self.variables, f"Attempted to evaluate GibbsModel with variables {self.variables} at {X}"
    # methods should not be called on batched traces -> use vmap instead
    def log_prior_likeli_pathcond(self, X: Trace, data_annealing: AnnealingMask) -> Tuple[FloatArray,FloatArray,BoolArray]:
        self._check_variables(X)
        return self.slp._log_prior_likeli_pathcond(X | self.Y, data_annealing)
    
    def tempered_log_prob(self, temperature: FloatArray, data_annealing: AnnealingMask) -> Callable[[Trace], FloatArray]:
        def _log_prob(X: Trace) -> FloatArray:
            self._check_variables(X)
            log_prior, log_likelihood, path_condition = self.log_prior_likeli_pathcond(X, data_annealing)
            return jax.lax.select(path_condition, log_prior + temperature * log_likelihood, -jnp.inf)
        return _log_prob
    
    def unraveled_tempered_log_prob(self, temperature: FloatArray, data_annealing: AnnealingMask, unravel_fn: Optional[Callable[[jax.Array],Trace]] = None) -> Callable[[jax.Array], FloatArray]:
        if unravel_fn is None:
            _, unravel_fn = ravel_pytree(self.X_representative)

        def _log_prob(X_flat: jax.Array) -> FloatArray:
            X = unravel_fn(X_flat)
            self._check_variables(X)
            log_prior, log_likelihood, path_condition = self.log_prior_likeli_pathcond(X, data_annealing)
            return jax.lax.select(path_condition, log_prior + temperature * log_likelihood, -jnp.inf)
        return _log_prob
    

    def unconstrained_log_prior_likeli_pathcond(self, X: Trace, data_annealing: AnnealingMask) -> Tuple[FloatArray,FloatArray,BoolArray, Trace]:
        self._check_variables(X)
        return self.slp._unconstrained_log_prior_likeli_pathcond(X | self.Y, data_annealing)

    def tempered_unconstrained_log_prob(self, temperature: FloatArray, data_annealing: AnnealingMask) -> Callable[[Trace], FloatArray]:
        def _log_prob(X: Trace) -> FloatArray:
            self._check_variables(X)
            log_prior, log_likelihood, path_condition, X_constrained = self.unconstrained_log_prior_likeli_pathcond(X, data_annealing)
            return jax.lax.select(path_condition, log_prior + temperature * log_likelihood, -jnp.inf)
        return _log_prob
    
    def unraveled_unconstrained_tempered_log_prob(self, temperature: FloatArray, data_annealing: AnnealingMask, unravel_fn: Optional[Callable[[jax.Array],Trace]] = None) -> Callable[[jax.Array], FloatArray]:
        if unravel_fn is None:
            _, unravel_fn = ravel_pytree(self.X_representative)

        def _log_prob(X_flat: jax.Array) -> FloatArray:
            X = unravel_fn(X_flat)
            self._check_variables(X)
            log_prior, log_likelihood, path_condition, X_constrained = self.unconstrained_log_prior_likeli_pathcond(X, data_annealing)
            return jax.lax.select(path_condition, log_prior + temperature * log_likelihood, -jnp.inf)
        return _log_prob
