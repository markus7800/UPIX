
from dccxjax import *

regime = MCMCSteps(
    MCMCStep(SingleVariable("x"), MH()),
    MCMCSteps(
        MCMCStep(SingleVariable("a"), MH()),
        MCMCStep(SingleVariable("b"), MH()),
    ),
    MCMCStep(SingleVariable("y"), MH()),
)

for r in regime:
    print(r)