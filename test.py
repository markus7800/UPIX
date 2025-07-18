
from dccxjax.all import *
import dccxjax.distributions as dist
import jax

def test_if():
    B = sample("B", dist.Normal(0., 1.))
    if B < 0:
        return sample("A", dist.Normal(0., 1.))
    else:
        return sample("C", dist.Normal(0., 1.))
    

with GenerateCtx(jax.random.PRNGKey(0)) as ctx:
    test_if()
    print(ctx.X)
