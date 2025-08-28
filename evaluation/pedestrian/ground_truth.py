import sys
sys.path.append("evaluation")
from parse_args import *
parser = get_arg_parser()
parser.add_argument("n_qs", type=int)
parser.add_argument("batch_size", type=int)
parser.add_argument("num_batches", type=int)
parser.add_argument("n_iter", type=int)
parser.add_argument("--show_plots", action="store_true")
parser.add_argument("--old", action="store_true")
args = parser.parse_args()
setup_devices_from_args(args)
from setup_parallelisation import get_parallelisation_config

import jax
import jax.numpy as jnp
from typing import NamedTuple
from time import time
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

class PedWhileState(NamedTuple):
    rng_key: jax.Array
    position: jax.Array
    distance: jax.Array
    t: jax.Array

def cond(state: PedWhileState):
    return (state.position > 0) & (state.distance < 10)

def body_fun(state: PedWhileState):
    new_rng_key, sample_key = jax.random.split(state.rng_key)
    step = jax.random.uniform(sample_key) * 2 - 1
    return PedWhileState(new_rng_key, state.position + step, state.distance + jax.lax.abs(step), state.t + 1)
    
@jax.jit
def pedestrian(rng_key: jax.Array):
    rng_key, sample_key = jax.random.split(rng_key)
    start = jax.random.uniform(sample_key) * 3.

    end_state = jax.lax.while_loop(cond, body_fun, PedWhileState(rng_key, start, jax.lax.pvary(jnp.array(0.), "s_axis"), jax.lax.pvary(jnp.array(0), "s_axis")))

    likelihood = jax.scipy.stats.norm.logpdf(1.1, end_state.distance, 0.1)

    return {"start": start, "lp": likelihood}


@jax.jit
def cdf(x, qs, weights: jax.Array):
    def _cdf(q):
        return jnp.where(x < q, weights, jax.numpy.zeros_like(weights)).sum()
    return jax.lax.map(_cdf, qs)

from dccxjax.parallelisation import *
from dccxjax.progress_bar import ProgressbarManager
from dccxjax.parallelisation import VectorisationType

@jax.jit
def cdf(x, qs, weights: jax.Array):
    def _cdf(q):
        return jnp.where(x < q, weights, jax.numpy.zeros_like(weights)).sum()
    return jax.lax.map(_cdf, qs)

def cdf_cruncher_old(qs, n_iter, batch_size):
    rng_keys = jax.random.split(jax.random.key(0), n_iter)
    c = jnp.zeros_like(qs)
    for (i,key) in tqdm(enumerate(rng_keys),total=n_iter):
        result = jax.vmap(pedestrian)(jax.random.split(key,batch_size))
        x = result["start"]
        lp = result["lp"]
        weights = jax.lax.exp(lp - jax.scipy.special.logsumexp(lp))
        c += cdf(x, qs, weights) * 1/n_iter
    return c

def cdf_cruncher(qs, batch_size, num_batches, n_iter, vectorisation):

    bar = tqdm(position=0)
    pmngr = ProgressbarManager("", bar, vectorisation==VectorisationType.PMAP)
    pmngr.set_num_samples(num_batches)

    def step(carry, key):
        i = carry
        result = pedestrian(key)
        x = result["start"]
        lp = result["lp"]
        return i+1, (jnp.searchsorted(qs, x), lp)
    
    scan_f = vectorise_scan(step, None, 1, batch_size, 0, vectorisation, pmngr, lambda x: x)

    def _count(i, _ix, _lps):
        def step(s, data):
            ix, lp = data
            return jnp.logaddexp(s, jax.lax.select(ix <= i, lp, -jnp.inf)), None
        return jax.lax.scan(step, jnp.array(-jnp.inf,float), (_ix, _lps))[0]
    count = jax.vmap(_count, in_axes=(0,None,None), out_axes=0)
    count_fn = vectorise(count, in_axes=(None,1,1), out_axes=0, batch_axis_size=batch_size, vectorisation=vectorisation, vmap_batch_size=0)


    def _get_c(i):
        pmngr.desc = f"{i}"
        rng_keys = jax.random.split(jax.random.key(0), (num_batches,batch_size))
        _, (ixs, lps) = scan_f(jnp.array(0,int), rng_keys)
        
        c = count_fn(jnp.arange(0,qs.size), ixs, lps)
        c = jax.scipy.special.logsumexp(c, axis=0)
        s = jax.scipy.special.logsumexp(lps)
        del ixs
        del lps
    
        return jax.lax.exp(c - s), s
    
    cs = []
    ss = []
    for i in tqdm(range(n_iter), position=1):
        c, s = parallel_run(_get_c, (i+1,), batch_size, pconfig.vectorisation)
        # print(f"{N=} {M=} {ixs.shape=}") # (N,M)
        cs.append(c)
        ss.append(s)

    cs = jnp.vstack(cs)
    return jnp.mean(cs, axis=0)
    # res_c = jax.scipy.special.logsumexp(cs, axis=0)

    # ss = jnp.hstack(ss)
    # res_s = jax.scipy.special.logsumexp(ss)
    # return jax.lax.exp(res_c - res_s)
    

# n_sq batch_size num_batches n_iter

# GTX 1070
# uv run --python python3 --extra cuda --locked --with PyQt6 evaluation/pedestrian/ground_truth.py sequential vmap_global 100 1_000_000 10 1_000 --show_plots

start_linspace = jnp.linspace(0., 3., args.n_qs)
N_SAMPLES = args.num_batches * args.batch_size * args.n_iter
print(f"N_QS = {args.n_qs:_} N_SAMPLES = {N_SAMPLES:_}")

if not args.old:
    pconfig = get_parallelisation_config(args)
    gt_cdf = cdf_cruncher(start_linspace, batch_size=args.batch_size, num_batches=args.num_batches, n_iter=args.n_iter, vectorisation=pconfig.vectorisation)
else:
    assert args.num_batches == 1
    gt_cdf = cdf_cruncher_old(start_linspace, batch_size=args.batch_size, n_iter=args.n_iter)

gt_pdf = jnp.hstack([jnp.array(0.),jnp.diff(gt_cdf)]) / (start_linspace[1] - start_linspace[0])

jnp.save(f"evaluation/pedestrian/gt_xs-{args.n_qs}.npy", start_linspace)
jnp.save(f"evaluation/pedestrian/gt_pdf_est-{args.n_qs}-{N_SAMPLES:_}.npy", gt_pdf)
jnp.save(f"evaluation/pedestrian/gt_cdf-{args.n_qs}-{N_SAMPLES:_}.npy", gt_cdf)

if args.show_plots:
    t0 = time()
    # result = pedestrian_batched(jax.random.key(0), 1_000_000, 1) # not batching while loop is faster on CPU
    result = jax.vmap(pedestrian)(jax.random.split(jax.random.key(0), 10_000_000))
    result["start"].block_until_ready()
    x = result["start"].reshape(-1)
    lp = result["lp"].reshape(-1)
    t1 = time()
    print(f"Finished in {t1-t0:.3f}s")
    
    plt.plot(start_linspace, gt_cdf)
    # plt.show()
    
    weights = jax.lax.exp(lp - jax.scipy.special.logsumexp(lp))
    print(weights.sum())
    plt.hist(x, weights=weights, density=True, bins=100)
    plt.grid(True)
    plt.yticks(jnp.arange(0.,1.2,0.05))
    # plt.show()

    x = x[weights > 1e-9]
    weights = weights[weights > 1e-9]
    print(x.shape)

    kde = jax.scipy.stats.gaussian_kde(x, weights=weights)
    kde_pdf = kde(start_linspace)
    plt.plot(start_linspace, kde_pdf, color="tab:blue")
    plt.plot(start_linspace, gt_pdf, color="tab:orange")
    plt.show()