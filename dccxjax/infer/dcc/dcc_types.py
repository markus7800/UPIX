from abc import ABC, abstractmethod
from typing import NamedTuple, Callable, Tuple
from dataclasses import dataclass
from dccxjax.core import SLP

__all__ = [
    "InferenceResult",
    "LogWeightEstimate",

]

class InferenceResult(ABC):
    @abstractmethod
    def combine_results(self, other: "InferenceResult") -> "InferenceResult":
        # fold left
        raise NotImplementedError


class LogWeightEstimate(ABC):
    @abstractmethod
    def combine_estimates(self, other: "LogWeightEstimate") -> "LogWeightEstimate":
        raise NotImplementedError
    
class InferenceTask:
    def __init__(self,f: Callable[...,InferenceResult], args: Tuple):
        self.f = f
        self.args = args
        
    # pre_run_info: Callable[[SLP], str]
    # post_run_info: Callable[[InferenceResult],str]
    def run(self) -> InferenceResult:
        return self.f(*self.args)

    