
import json
from pprint import pprint
from dataclasses import dataclass
from abc import ABC

# miking-benchmarks/benchmark-suite/benchmarks/ppl/phyl/webppl/phyjs/index.js

class Node(ABC):
    age: float

@dataclass
class Leaf(Node):
    age: float
    index: int
    
@dataclass
class Branch(Node):
    age: float
    left: Node
    right: Node

bisse_32 = Branch(13.016,Branch(10.626,Branch(8.352,Branch(7.679,Branch(5.187,Leaf(0.0,7),Leaf(0.0,22)),Branch(5.196,Leaf(0.0,2),Branch(4.871,Branch(2.601,Leaf(0.0,31),Leaf(0.0,14)),Leaf(0.0,26)))),Branch(7.361,Branch(3.818,Branch(1.143,Branch(0.829,Leaf(0.0,6),Leaf(0.0,9)),Leaf(0.0,16)),Branch(1.813,Branch(0.452,Branch(0.203,Leaf(0.0,15),Leaf(0.0,12)),Leaf(0.0,8)),Leaf(0.0,32))),Branch(1.868,Branch(0.866,Leaf(0.0,23),Branch(0.001,Leaf(0.0,17),Leaf(0.0,24))),Branch(1.06,Leaf(0.0,18),Leaf(0.0,4))))),Branch(10.536,Branch(8.291,Branch(1.396,Branch(0.215,Leaf(0.0,10),Leaf(0.0,29)),Leaf(0.0,21)),Leaf(0.0,27)),Branch(8.192,Branch(0.56,Leaf(0.0,11),Leaf(0.0,19)),Leaf(0.0,3)))),Branch(8.958,Branch(3.748,Leaf(0.0,5),Branch(0.033,Leaf(0.0,20),Leaf(0.0,1))),Branch(7.775,Branch(0.584,Leaf(0.0,28),Leaf(0.0,13)),Branch(1.589,Leaf(0.0,25),Leaf(0.0,30)))))

def count_leaves(node: Node) -> int:
    if isinstance(node, Leaf):
        return 1
    else:
        assert isinstance(node, Branch)
        return count_leaves(node.left) + count_leaves(node.right)
    
import math
def ln_factorial(n: int) -> float:
    if (n == 1):
        return 0.0
    else:
        return math.log(n) + ln_factorial(n-1)

def _is_leaf(node: dict):
    return len(node["children"]) == 0

def _get_left_age(node: dict):
    if _is_leaf(node):
        return 0.
    else:
        child: dict = node["children"][0]
        return child["branch_length"] + _get_left_age(child)

def _construct_tree(j: dict):
    if _is_leaf(j):
        return Leaf(0.0, j["taxon"]+1)
    else:
        return Branch(float(_get_left_age(j)), _construct_tree(j["children"][0]), _construct_tree(j["children"][1]))

def read_phyjson(filename: str):
    with open(filename, "r") as f:
        j = json.load(f)
        return _construct_tree(j["trees"][0]["root"])
        
    

def pretty(n: Node) -> str:
    if isinstance(n, Leaf):
        return "Leaf {age = " + str(n.age) + "}"
    else:
        assert isinstance(n, Branch)
        return "Node {left = " + pretty(n.left) + ", right = " + pretty(n.right) + ", age = " + str(n.age) + "}"
    
    
# print(pretty(read_phyjson("evaluation/phylo/data/Alcedinidae.phyjson")))

# pprint(read_phyjson("evaluation/phylo/data/bisse_32_rounded.phyjson"))

# pprint(bisse_32)