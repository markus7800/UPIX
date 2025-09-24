from typing import List, Dict, Set
from functools import reduce
from upix.infer.exact.factors import Factor, factor_product, factor_sum, factor_sum_out_addr
import jax
import jax.numpy as jnp

__all__ = [
    "variable_elimination",
]

def variable_elimination(factors: List[Factor], elimination_order: List[str]):
    # print("\nvariable_elimination")
    variable_to_factors: Dict[str, Set[Factor]] = {}
    for factor in factors:
        for addr in factor.addresses:
            if addr not in variable_to_factors:
                variable_to_factors[addr] = set()
            variable_to_factors[addr].add(factor)
            
    tau: Factor = Factor([], jnp.array(0.,float))
    for addr in elimination_order:
        # print(f"eliminate {addr}")
        # pprint(variable_to_factors)
        neighbour_factors = variable_to_factors[addr]
        assert len(neighbour_factors) > 0
        psi = reduce(factor_product, neighbour_factors)
        tau = factor_sum_out_addr(psi, addr)
        # print(f"{tau=}")
        for factor in neighbour_factors:
            for variable in factor.addresses:
                factor_set = variable_to_factors[variable]
                if factor_set is not neighbour_factors:
                    factor_set.discard(factor)
        for variable in tau.addresses:
            variable_to_factors[variable].add(tau)
        del variable_to_factors[addr]
        # print()
        
    if len(variable_to_factors) == 0:
        log_evidence = jax.scipy.special.logsumexp(tau.table)
        return Factor([], jnp.array(0.,float)), log_evidence
    else:
        remaining_factors = reduce(lambda x, y: x | y, variable_to_factors.values())
        result = reduce(factor_product, remaining_factors)
        log_evidence = jax.scipy.special.logsumexp(result.table)
        return result, log_evidence
        
    

        