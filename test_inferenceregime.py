
from dccxjax import *

regime = Gibbs(
    InferenceStep(SingleVariable("x"), MH()),
    Gibbs(
        InferenceStep(SingleVariable("a"), MH()),
        InferenceStep(SingleVariable("b"), MH()),
    ),
    InferenceStep(SingleVariable("y"), MH()),
)

for r in regime:
    print(r)