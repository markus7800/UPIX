
from abc import ABC, abstractmethod
from typing import Callable, TypeVar, Dict

__all__ = [
    "AllVariables",
    "SingleVariable",
    "VariableSet",
    "PrefixSelector",
    "PredicateSelector"
]

class VariableSelector(ABC):
    @abstractmethod
    def contains(self, variable: str) -> bool:
        raise NotImplementedError

class AllVariables(VariableSelector):
    def contains(self, variable: str) -> bool:
        return True

class SingleVariable(VariableSelector):
    def __init__(self, variable: str) -> None:
        self.variable = variable
    def contains(self, variable: str) -> bool:
        return self.variable == variable
    
class VariableSet(VariableSelector):
    def __init__(self, *variables: str) -> None:
        self.variable_set = set(variables)
    def contains(self, variable: str) -> bool:
        return variable in self.variable_set
    
class PrefixSelector(VariableSelector):
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
    def contains(self, variable: str) -> bool:
        return variable.startswith(self.prefix)
    
class PredicateSelector(VariableSelector):
    def __init__(self, predicate: Callable[[str], bool]) -> None:
        self.predicate = predicate
    def contains(self, variable: str) -> bool:
        return self.predicate(variable)
    

T = TypeVar("T")
def find_or_error(d: Dict[VariableSelector, T], address: str) -> T:
    for selector, item in d.items():
        if selector.contains(address):
            return item
    raise ValueError()