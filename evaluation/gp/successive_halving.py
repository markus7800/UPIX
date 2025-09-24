

import math
import heapq
from typing import List, Tuple
from upix.core import SLP

class SuccessiveHalving:
    def __init__(self, num_total_iterations: int, num_final_arms: int):
        self.num_total_iterations = num_total_iterations
        self.num_final_arms = num_final_arms

    def calculate_num_phases(self, num_arms: int) -> int:
        self.num_phases = 1
        num_active_arms = num_arms
        while num_active_arms > self.num_final_arms:
            num_active_arms = max(math.ceil(num_active_arms / 2), self.num_final_arms)
            self.num_phases += 1

        return self.num_phases

    def calculate_num_optimization_steps(self, num_active_arms: int) -> int:
        return math.floor(self.num_total_iterations / (self.num_phases * num_active_arms))

    def select_active_slps(self, arm2reward: List[Tuple[SLP,float]]) -> List[Tuple[SLP,float]]:
        num_active_arms = len(arm2reward)
        num_to_keep = max(math.ceil(num_active_arms / 2), self.num_final_arms)
        arms_to_keep = heapq.nlargest(num_to_keep, arm2reward, key=lambda v: v[1])
        return arms_to_keep