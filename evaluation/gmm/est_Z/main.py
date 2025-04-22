import sys
sys.path.insert(0, ".")
import os
# import multiprocessing

# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
#     multiprocessing.cpu_count()
# )

if len(sys.argv) > 1:
    if sys.argv[1] == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"

from dccxjax import *
import jax
import jax.numpy as jnp
import dccxjax.distributions as dist
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from typing import List, NamedTuple
from time import time

from dccxjax.infer.mcmc import MCMCState, InferenceInfos, get_inference_regime_mcmc_step_for_slp, add_progress_bar
from dccxjax.infer.dcc import DCC_Result
from dccxjax.core.branching_tracer import retrace_branching

from data import *
from lw import *
from importance import *
from chibs import *

import logging
setup_logging(logging.WARN)

compilation_time_tracker = CompilationTimeTracker()
jax.monitoring.register_event_duration_secs_listener(compilation_time_tracker)


do_lw(N=100_000)
# do_lw N=10,000,000,000 batch_method=1 ys.shape=(100,)
# log_Z_path=Array([-438.79694, -412.90247, -378.02847, -372.42322, -370.9722 ,
#        -370.99368, -371.79416], dtype=float32)
# ESS=Array([1.1252690e+06, 2.0516587e+03, 1.1735829e+01, 3.7160053e+01,
#        3.2372032e+01, 2.7773899e+01, 2.7465406e+01], dtype=float32)
# 0 5.1019314e-31
# 1 2.6957338e-19
# 2 0.0005654061
# 3 0.15370637
# 4 0.49194857
# 5 0.2888929
# 6 0.06487318

# do_lw N=10,000,000,000 batch_method=1 ys.shape=(25,)
# log_Z_path=Array([-110.42564, -107.22047, -105.35848, -103.65361, -102.03826,
#        -101.3502 , -101.00215], dtype=float32)
# ESS=Array([8409222.   ,  303162.6  ,   49098.805,   44003.824,   56211.375,
#         116899.21 ,  191827.61 ], dtype=float32)
# 0 2.0095133e-05
# 1 0.0014866154
# 2 0.014352876
# 3 0.078950614
# 4 0.29782015
# 5 0.35557216
# 6 0.25179815

do_is(N=100_000)
# do_is N = 1_000_000_000 ys.shape=(100,)
# log_Z_path=Array([-438.7951 , -412.87714, -378.07547, -372.03406, -371.0071 ,
#        -371.09848, -371.74484], dtype=float32)
# ESS=Array([9.8786797e+08, 1.5797326e+06, 1.3217342e+05, 1.7380820e+04,
#        4.0514241e+03, 4.1891110e+02, 2.2887285e+02], dtype=float32)
# 0 4.9588643e-31
# 1 2.6824395e-19
# 2 0.0005233622
# 3 0.22006674
# 4 0.46090347
# 5 0.25239247
# 6 0.06612039

do_chibs(n_chains=10, n_samples_per_chain=10_0000)
# do chibs n_chains=10 n_samples_per_chain=1,000,000 ys.shape=(100,)
# log_Z_path=Array([-438.79498, -413.79248, -379.86838, -374.56332, -372.84283,
#        -372.89096, -372.95206], dtype=float32)
# 0 3.242578e-30
# 1 7.021925e-19
# 2 0.0005696494
# 3 0.11469994
# 4 0.4806406
# 5 0.27483195
# 6 0.1292719

# do chibs n_chains=10 n_samples_per_chain=1,000,000 ys.shape=(25,)
# log_Z_path=Array([-110.42581 , -107.036156, -105.42306 , -103.79447 , -101.96445 ,
#        -101.23255 , -101.096924], dtype=float32)
# 0 1.9440942e-05
# 1 0.0017296032
# 2 0.013019579
# 3 0.06635639
# 4 0.3102515
# 5 0.3870108
# 6 0.22161344