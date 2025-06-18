
from abc import ABC, abstractmethod
from typing import Callable, TypeVar, Dict

__all__ = [
    "AllVariables",
    "SingleVariable",
    "VariableSet",
    "PrefixSelector",
    "SuffixSelector",
    "RegexSelector",
    "PredicateSelector",
    "ComplementSelector",
    "VariableSelector"
]

class VariableSelector(ABC):
    @abstractmethod
    def contains(self, variable: str) -> bool:
        raise NotImplementedError

class AllVariables(VariableSelector):
    def contains(self, variable: str) -> bool:
        return True
    def __repr__(self) -> str:
        return "<all variables>"

class SingleVariable(VariableSelector):
    def __init__(self, variable: str) -> None:
        self.variable = variable
    def contains(self, variable: str) -> bool:
        return self.variable == variable
    def __repr__(self) -> str:
        return f"<{self.variable}>"
    
class VariableSet(VariableSelector):
    def __init__(self, *variables: str) -> None:
        self.variable_set = set(variables)
    def contains(self, variable: str) -> bool:
        return variable in self.variable_set
    def __repr__(self) -> str:
        return f"<{sorted(self.variable_set)}>"
    
class PrefixSelector(VariableSelector):
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
    def contains(self, variable: str) -> bool:
        return variable.startswith(self.prefix)
    def __repr__(self) -> str:
        return f"<{self.prefix}.*>"
    
class SuffixSelector(VariableSelector):
    def __init__(self, suffix: str) -> None:
        self.suffix = suffix
    def contains(self, variable: str) -> bool:
        return variable.endswith(self.suffix)
    def __repr__(self) -> str:
        return f"<.*{self.suffix}>"
    
import re
class RegexSelector(VariableSelector):
    def __init__(self, pattern: str | re.Pattern[str]) -> None:
        self.pattern = pattern
    def contains(self, variable: str) -> bool:
        return re.match(self.pattern, variable) is not None
    def __repr__(self) -> str:
        return f"<r\"{self.pattern}\">"
    
class PredicateSelector(VariableSelector):
    def __init__(self, predicate: Callable[[str], bool]) -> None:
        self.predicate = predicate
    def contains(self, variable: str) -> bool:
        return self.predicate(variable)
    def __repr__(self) -> str:
        return f"<predicate {self.predicate}>"
    
class ComplementSelector(VariableSelector):
    def __init__(self, selector: VariableSelector) -> None:
        self.selector = selector
    def contains(self, variable: str) -> bool:
        return not self.selector.contains(variable)
    def __repr__(self) -> str:
        return f"<complement {self.selector}>"


T = TypeVar("T")
def find_or_error(d: Dict[VariableSelector, T], address: str) -> T:
    for selector, item in d.items():
        if selector.contains(address):
            return item
    raise ValueError()