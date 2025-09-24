
from dataclasses import dataclass
from typing import Set, List, Dict, cast
import math
from upix.infer.exact.factors import Factor
from functools import reduce

__all__ = [
    "get_greedy_elimination_order"
]

@dataclass
class ReductionSize:
    # node to eliminate
    v: str
    # set of variables connected to node v via some factor,
    # if we eliminate v, tau will be a factor of these nodes
    nodes: Set[str]
    
    individual: int # sum of sizes of all factors connected to node v
    combined: int # size of factor resulting from multiplying all factors connected to node v and summing out v (size of tau)
    reduction: int # individual - combined
    
    position: int # position in heap
    metric: int # cached metric
    
    def __lt__(self, other):
        assert isinstance(other, ReductionSize)
        return self.metric < other.metric
    
# 1 indexed
class Heap:
    def __init__(self, l: List[ReductionSize]) -> None:
        self.l = l
        heapify(self)
    def __getitem__(self, key):
        return self.l[key - 1]
    def __setitem__(self, key, value):
        self.l[key - 1] = value
    def __len__(self):
        return len(self.l)
    def resize(self, n: int):
        self.l = self.l[:n]
        assert len(self) == n
    
def compute_metric(r: ReductionSize):
    return r.combined


def heap_left(i: int):
    return 2 * i
def heap_right(i: int):
    return 2 * i + 1
def heap_parent(i: int):
    return i // 2

def heapify_down(A: Heap, i: int, x: ReductionSize, n: int) -> int:
    while True:
        l = heap_left(i)
        r = heap_right(i)
        if not l <= n:
            break
        j = l if (r > n or A[l] < A[r]) else r
        if not A[j] < x:
            break
        A[i] = A[j]
        A[i].position = i
        i = j
    A[i] = x
    x.position = i
    return i

def heapify_up(A: Heap, i: int, x: ReductionSize) -> int:
    while True:
        j = heap_parent(i)
        if not (j >= 1 and x < A[j]):
            break
        A[i] = A[j]
        A[i].position = i
        i = j
    A[i] = x
    x.position = i
    return i
    
def heapify(A: Heap):
    n = len(A)
    for i in range(n, 0, -1): # n,...,1
        heapify_down(A, i, A[i], n)
        
        
def isheap(A: Heap):
    n = len(A)
    for i in range(1, n//2 + 1): # 1,...,n//2
        l = heap_left(i)
        r = heap_right(i)
        if A[l] < A[i] or (r <= n and A[r] < A[i]):
            return False
    return True

def pop_heap(A: Heap):
    n = len(A)
    x = A[1]
    if n > 1:
        y = A[n]
        heapify_down(A, 1, y, n - 1)
    A.resize(n-1)
    return x
        
def initialise_reduction_size_heap(factors: List[Factor], var_to_dim: Dict[str,int], marignal_variables: List[str]):
    reduction_size: Dict[str, ReductionSize] = dict()
    marignal_variables_set = set(marignal_variables)
    reduction_size_list: List[ReductionSize] = []
    assert len(marignal_variables_set.intersection(var_to_dim.keys())) == len(marignal_variables_set)
    
    i = 1
    for v in var_to_dim.keys() - marignal_variables_set:
        r = ReductionSize(v, set(), 0, 1, 0, i, 0)
        for f in factors:
            if v not in f.addresses:
                continue
            r.individual += math.prod([var_to_dim[v] for v in f.addresses])
            for n in f.addresses:
                if n not in r.nodes and n != v:
                    r.nodes.add(n)
                    r.combined *= var_to_dim[n]
        r.reduction = r.combined - r.individual
        r.metric = compute_metric(r)
        
        reduction_size[v] = r
        reduction_size_list.append(r)
        i += 1
        
    reduction_size_heap = Heap(reduction_size_list)
    assert isheap(reduction_size_heap)
    
    return reduction_size, reduction_size_heap


def remove_f_from_reduction_score_of_v(reduction_size: Dict[str, ReductionSize], var_to_dim: Dict[str,int], node: str, v: str, f: Factor):
    if v in reduction_size:
        r = reduction_size[v]
        
        r.individual -= math.prod([var_to_dim[v] for v in f.addresses])
        if node in r.nodes:
            r.combined //= var_to_dim[node]
            r.nodes.discard(node)

def add_tau_to_reduction_score_of_v_and_heapify(reduction_size: Dict[str, ReductionSize], var_to_dim: Dict[str,int], reduction_size_heap: Heap, v: str, tau: Factor):
    if v in reduction_size:
        r = reduction_size[v]
        r.individual += math.prod([var_to_dim[v] for v in tau.addresses])
        for n in tau.addresses:
            if n not in r.nodes and n != v:
                r.nodes.add(n)
                r.combined *= var_to_dim[n]
        r.reduction = r.combined - r.individual
        before = r.metric
        after = compute_metric(r)
        r.metric = after
        
        if before < after:
            heapify_down(reduction_size_heap, r.position, reduction_size_heap[r.position], len(reduction_size_heap))
        else:
            heapify_up(reduction_size_heap, r.position, reduction_size_heap[r.position])
            
        assert isheap(reduction_size_heap)
        
        
def get_greedy_elimination_order(factors: List[Factor], marginal_variables: List[str]):
    # print("\nget_greedy_elimination_order")
    var_to_dim: Dict[str, int] = dict()

    variable_to_factors: Dict[str, Set[Factor]] = {}
    for factor in factors:
        for i, addr in enumerate(factor.addresses):
            if addr not in variable_to_factors:
                variable_to_factors[addr] = set()
            variable_to_factors[addr].add(factor)
            
            if addr in var_to_dim:
                assert var_to_dim[addr] == factor.table.shape[i]
            else:
                var_to_dim[addr] = factor.table.shape[i]
            
    reduction_size, reduction_size_heap = initialise_reduction_size_heap(factors, var_to_dim, marginal_variables)
    # pprint(reduction_size)
    
    elimination_order: List[str] = []
            
    for _ in range(len(var_to_dim) - len(marginal_variables)):
        r = pop_heap(reduction_size_heap)
        addr = r.v
        elimination_order.append(addr)
        # print("eliminate", addr)
        
        neighbour_factors = variable_to_factors[addr]
        assert len(neighbour_factors) > 0
        
        # mock the computation
        tau_neighbours = reduce(lambda x, y: x | set(y.addresses), neighbour_factors, cast(Set[str], set()))
        tau_neighbours.discard(addr)
        tau = Factor(sorted(tau_neighbours), None) # type: ignore
        # print(f"tau=Factor({tau.addresses}, {tuple([var_to_dim[v] for v in tau.addresses])})")
       
        for factor in neighbour_factors:
            for variable in factor.addresses:
                if variable == addr:
                    continue
                factor_set = variable_to_factors[variable]
                if factor_set is not neighbour_factors:
                    factor_set.discard(factor)
                remove_f_from_reduction_score_of_v(reduction_size, var_to_dim, addr, variable, factor)
        
        for variable in tau.addresses:
            variable_to_factors[variable].add(tau)
            add_tau_to_reduction_score_of_v_and_heapify(reduction_size, var_to_dim, reduction_size_heap, variable, tau)
            
        del variable_to_factors[addr]
        del reduction_size[addr]
        
    return elimination_order