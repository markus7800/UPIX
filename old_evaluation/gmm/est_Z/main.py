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
from locally_restriced_is import *
from mixture import *

import logging
setup_logging(logging.WARN)

compilation_time_tracker = CompilationTimeTracker()
jax.monitoring.register_event_duration_secs_listener(compilation_time_tracker)


do_lw(N=100_000_000)
# do_lw N=100,000,000 batch_method=1 ys.shape=(25,)
# log_Z_path=Array([-110.42311 , -107.23499 , -105.42786 , -103.646774, -102.01914 ,
#        -101.30518 , -101.01882 ], dtype=float32)
# ESS=Array([84174.94   ,  3077.6494 ,  1878.0023 ,   248.86404,   639.2897 ,
#         1133.2928 ,  2559.142  ], dtype=float32)
# log_Z=Array([-112.42311, -108.54185, -106.73471, -105.35909, -104.42461,
#        -104.62694, -105.43919], dtype=float32)
# 0 0.00012498461
# 1 0.006059935
# 2 0.036923036
# 3 0.1461237
# 4 0.37201715
# 5 0.30387256
# 6 0.13487557

# do_is(N=100_000)

# do_chibs(n_chains=10, n_samples_per_chain=10_000)

# lis(3, n_chains=100, n_samples_per_chain=100_000)

# do_mixture_is(N = 1_000_000, n_chains = 10, n_samples_per_chain = 100_000, n_components = 10_000, sigma = 0.5)
