from abc import ABC, abstractmethod
from typing import NamedTuple, Callable, Tuple, TypeVar, Generic
from dataclasses import dataclass
from dccxjax.core import SLP
from jax.export import Exported
from .export import export_flat

__all__ = [
    "InferenceResult",
    "LogWeightEstimate",
    "InferenceTask",
    "EstimateLogWeightTask",
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
    
TASK_RESULT = TypeVar("TASK_RESULT")


class ExportedJaxTask(Generic[TASK_RESULT]):
    def __init__(self, exported_fn: Exported, args: Tuple, in_tree, out_tree, pre_info: Callable[[],str] | None, post_info: Callable[[TASK_RESULT],str] | None) -> None:
        self.exported_fn = exported_fn
        self.args = args
        self.in_tree = in_tree
        self.out_tree = out_tree
        self._pre_info = pre_info
        self._post_info = post_info
        
    def pre_info(self) -> str:
        if self._pre_info is None:
            return ""
        return self._pre_info()
        
    def post_info(self, result: TASK_RESULT) -> str:
        if self._post_info is None:
            return ""
        return self._post_info(result)
        
# callable does not have to be jit-wrapped
class JaxTask(Generic[TASK_RESULT]):
    def __init__(self, 
                 f: Callable[...,TASK_RESULT], args: Tuple, 
                 pre_info: Callable[[],str] | None = None,
                 post_info: Callable[[TASK_RESULT],str] | None = None):
        self.f = f
        self.args = args
        self._pre_info = pre_info
        self._post_info = post_info
        
    def run(self) -> TASK_RESULT:
        return self.f(*self.args)
    
    def pre_info(self) -> str:
        if self._pre_info is None:
            return ""
        return self._pre_info()
        
    def post_info(self, result: TASK_RESULT) -> str:
        if self._post_info is None:
            return ""
        return self._post_info(result)
        
    def export(self) -> ExportedJaxTask[TASK_RESULT]:
        exported_fn, in_tree, out_tree = export_flat(self.f, ("cpu",), (), None)(*self.args)
        return ExportedJaxTask[TASK_RESULT](
            exported_fn, self.args, in_tree, out_tree, self._pre_info, self._post_info
        )
    
InferenceTask = JaxTask[InferenceResult]
EstimateLogWeightTask = JaxTask[LogWeightEstimate]