from .slp_gen import SLP
from .variable_selector import VariableSelector
from typing import Dict, Set, Optional, Tuple
import jax
from .types import Trace


def make_gibbs_log_prob(slp: SLP, conditional_variables: Set[str]):
    def _gibbs_log_prob(X: Trace, Y: Trace):
        for address in conditional_variables:
            X[address] = Y[address]
        return slp._log_prob(X)
    return _gibbs_log_prob

class GibbsModel:
    def __init__(self, slp: SLP, variable_selector: VariableSelector, Y: Optional[Trace] = None) -> None:
        self.variables: Set[str] = set()
        self.conditional_variables: Set[str] = set()
        for address in slp.decision_representative.keys():
            if variable_selector.contains(address):
                self.variables.add(address)
            else:
                self.conditional_variables.add(address)
        self._gibbs_log_prob = jax.jit(make_gibbs_log_prob(slp, self.conditional_variables))
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

    def log_prob(self, X: Trace):
        assert X.keys() == self.variables
        return self._gibbs_log_prob(X, self.Y)

    