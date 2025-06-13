
import os
import time
import sys

if len(sys.argv) > 1:
    if sys.argv[1].endswith("cpu"):
        print("Force run on CPU.")
        os.environ["JAX_PLATFORMS"] = "cpu"


import jax
import jax.numpy as jnp
# import numpyro.distributions as numpyro_dist
from typing import Generic, NamedTuple, TypeVar, Callable, Tuple
import jax.flatten_util
from dataclasses import dataclass

xs = jnp.array([0.        , 0.00815146, 0.01551407, 0.02366553, 0.03155404,
       0.0397055 , 0.047594  , 0.05574546, 0.06389692, 0.07178544,
       0.07993689, 0.0878254 , 0.09597686, 0.10412832, 0.11149093,
       0.11964238, 0.1275309 , 0.13568236, 0.14357087, 0.15172233,
       0.15987378, 0.1677623 , 0.17591375, 0.18380226, 0.19195372,
       0.20010518, 0.2074678 , 0.21561925, 0.22350776, 0.23165922,
       0.23954773, 0.24769919, 0.25585064, 0.26373914, 0.2718906 ,
       0.27977914, 0.28793058, 0.29608205, 0.3037076 , 0.31185907,
       0.31974757, 0.32789904, 0.33578753, 0.343939  , 0.35209045,
       0.35997897, 0.36813042, 0.37601894, 0.38417038, 0.39232185,
       0.39968446, 0.40783593, 0.41572443, 0.4238759 , 0.4317644 ,
       0.43991587, 0.4480673 , 0.45595583, 0.46410728, 0.4719958 ,
       0.48014724, 0.4882987 , 0.49566132, 0.5038128 , 0.5117013 ,
       0.51985276, 0.52774125, 0.5358927 , 0.5440442 , 0.5519327 ,
       0.56008416, 0.56797266, 0.57612413, 0.58427554, 0.5916382 ,
       0.5997896 , 0.6076782 , 0.6158296 , 0.62371814, 0.63186955,
       0.640021  , 0.6479095 , 0.656061  , 0.6639495 , 0.67210096,
       0.68025243, 0.687878  , 0.6960294 , 0.703918  , 0.7120694 ,
       0.71995795, 0.72810936, 0.73626083, 0.7441493 , 0.7523008 ,
       0.7601893 , 0.76834077, 0.77649224, 0.78385484, 0.7920063 ,
       0.7998948 , 0.8080463 , 0.8159348 , 0.82408625, 0.8322377 ,
       0.8401262 , 0.8482777 , 0.8561662 , 0.86431766, 0.8724691 ,
       0.87983173, 0.88798314, 0.8958717 , 0.9040231 , 0.91191167,
       0.9200631 , 0.92821455, 0.9361031 , 0.9442545 , 0.9521431 ,
       0.9602945 , 0.96844596, 0.97580856, 0.98396003, 0.9918485 ,
       1.        ], dtype=float)
ys = jnp.array([-0.35215533, -0.33719274, -0.30228   , -0.30976132, -0.32971144,
       -0.29479873, -0.26237977, -0.26237977, -0.29230496, -0.33469898,
       -0.37210545, -0.33719274, -0.34467402, -0.31724262, -0.27983612,
       -0.29479873, -0.31973636, -0.259886  , -0.20751692, -0.20751692,
       -0.2374421 , -0.29978624, -0.3471678 , -0.2823299 , -0.26986107,
       -0.25739223, -0.1875668 , -0.22497328, -0.20252939, -0.1875668 ,
       -0.13519771, -0.13519771, -0.1726042 , -0.22746705, -0.2673673 ,
       -0.21749198, -0.20502315, -0.18257926, -0.15016031, -0.1800855 ,
       -0.17509797, -0.08781617, -0.05789099, -0.0279658 , -0.11026006,
       -0.15514785, -0.20252939, -0.14766654, -0.14267902, -0.14267902,
       -0.04292839, -0.04542216, -0.06038475, -0.02547203,  0.02689704,
        0.04684717, -0.04043463, -0.10527253, -0.18257926, -0.13021019,
       -0.12272889, -0.16262914, -0.04542216, -0.06537228, -0.04791592,
        0.02689704,  0.12166014,  0.09921625,  0.01442822, -0.06038475,
       -0.12522265, -0.06038475, -0.0279658 , -0.05040969,  0.03437834,
        0.03936587,  0.04185964,  0.1540791 ,  0.2762736 ,  0.23387958,
        0.14659779,  0.0518347 , -0.04043463,  0.06180976,  0.07677235,
        0.05931599,  0.15906662,  0.14909156,  0.16156039,  0.30121127,
        0.3984681 ,  0.378518  ,  0.25382972,  0.1316352 ,  0.0443534 ,
        0.1316352 ,  0.1540791 ,  0.11916637,  0.2563235 ,  0.23637335,
        0.25382972,  0.420912  ,  0.52814394,  0.5331315 ,  0.37602422,
        0.23387958,  0.12914144,  0.20644817,  0.21642323,  0.16156039,
        0.27128607,  0.23637335,  0.27377984,  0.45333096,  0.5929818 ,
        0.6278945 ,  0.37602422,  0.26380476,  0.14161026,  0.20894194,
        0.26629853,  0.22141077,  0.38101175,  0.3560741 ,  0.41592446,
        0.5456003 ], dtype=float)


def transform_log_normal(z, mu, sigma):
    return jax.lax.exp(mu + sigma*z)
def transform_logit_normal(z, scale, mu, sigma):
    return scale * 1 / (1 + jax.lax.exp(-(mu + sigma*z)))
def transform_param(field: str, z):
    if field == "gamma":
        return transform_logit_normal(z, 2, 0, 1)
    else:
        return transform_log_normal(z, -1.5, 1.)
    
@dataclass
class SquaredExponential():
    lengthscale: jax.Array
    amplitude: jax.Array
    def eval_cov_vec(self, ts: jax.Array) -> jax.Array:
        dt = ts.reshape(-1,1) - ts.reshape(1,-1)
        c = jax.lax.exp(-0.5 * dt * dt / jax.lax.square(self.lengthscale))
        return self.amplitude * c
    
# def logdensity1(x):
#     lengthscale = transform_param("lengthscale", x[0])
#     scale_mixture = transform_param("scale_mixture", x[1])
#     noise = transform_param("noise", x[2]) + 1e-5
#     lp = jax.scipy.stats.norm.logpdf(lengthscale) + jax.scipy.stats.norm.logpdf(scale_mixture) + jax.scipy.stats.norm.logpdf(noise)
#     k = UnitRationalQuadratic(lengthscale, scale_mixture)
#     cov_matrix = k.eval_cov_vec(xs) + noise * jnp.eye(xs.size)
#     return lp + jax.scipy.stats.multivariate_normal.logpdf(ys, jax.lax.zeros_like_array(xs), cov_matrix)

def logdensity(x):
    lengthscale = transform_param("lengthscale", x[0])
    amplitude = transform_param("amplitude", x[1])
    noise = transform_param("noise", x[2]) + 1e-5
    lp = jax.scipy.stats.norm.logpdf(lengthscale) + jax.scipy.stats.norm.logpdf(amplitude) + jax.scipy.stats.norm.logpdf(noise)
    k = SquaredExponential(lengthscale, amplitude)
    cov_matrix = k.eval_cov_vec(xs) + noise * jnp.eye(xs.size)
    return lp + jax.scipy.stats.multivariate_normal.logpdf(ys, jax.lax.zeros_like_array(xs), cov_matrix)


# we do not need pytrees / we operate on arrays
OPTIMIZER_STATE = TypeVar("OPTIMIZER_STATE")
OPTIMIZER_PARAMS = jax.Array
OPTIMIZER_UPDATES = jax.Array


class Optimizer(NamedTuple, Generic[OPTIMIZER_STATE]):
    init_fn: Callable[[OPTIMIZER_PARAMS], OPTIMIZER_STATE]
    update_fn: Callable[[int, OPTIMIZER_UPDATES,
                         OPTIMIZER_STATE], OPTIMIZER_STATE]
    get_params_fn: Callable[[OPTIMIZER_STATE], OPTIMIZER_PARAMS]

def Adam(step_size: float, b1=0.9, b2=0.999, eps=1e-8) -> Optimizer:

    def init(x0: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        m0 = jnp.zeros_like(x0)
        v0 = jnp.zeros_like(x0)
        return x0, m0, v0

    def update(i: int, g: jax.Array, state: Tuple[jax.Array, jax.Array, jax.Array]):
        x, m, v = state
        m = (1 - b1) * g + b1 * m  # First  moment estimate.
        v = (1 - b2) * jnp.square(g) + b2 * v  # Second moment estimate.
        # Bias correction.
        mhat = m / (1 - jnp.asarray(b1, m.dtype) ** (i + 1))
        vhat = v / (1 - jnp.asarray(b2, m.dtype) ** (i + 1))
        x = x - step_size * mhat / (jnp.sqrt(vhat) + eps)
        return x, m, v

    def get_params(state: Tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
        x, _, _ = state
        return x
    return Optimizer(init, update, get_params)


class Meanfield:
    def __init__(self, X, init_sigma: float = 1.) -> None:
        flat_X, unravel_fn = jax.flatten_util.ravel_pytree(X)
        self.n_latents = flat_X.shape[0]
        self.mu = jax.lax.zeros_like_array(flat_X)
        self.omega = jax.lax.full_like(flat_X, jnp.log(init_sigma))
        self.unravel_fn = unravel_fn
    def get_params(self) -> jax.Array:
       return jax.lax.concatenate((self.mu, self.omega), 0)
    def update_params(self, params: jax.Array):
       self.mu = params[:self.n_latents]
       self.omega = params[self.n_latents:]
    def sample_and_log_prob(self, rng_key, shape = ()) -> Tuple[jax.Array, jax.Array]:
        z = jax.random.normal(rng_key, self.mu.shape)
        x = self.mu + jax.lax.exp(self.omega) * z
        return self.unravel_fn(x), jax.scipy.stats.norm.logpdf(z).sum()
    
        d = numpyro_dist.Normal(self.mu, jax.lax.exp(self.omega))
        x = d.rsample(rng_key, shape)
        lp = d.log_prob(x).sum(axis=-1)
        return self.unravel_fn(x), lp

class ADVIState(NamedTuple, Generic[OPTIMIZER_STATE]):
    iteration: int
    optimizer_state: OPTIMIZER_STATE

def make_advi_step(logdensity, guide: Meanfield, optimizer: Optimizer[OPTIMIZER_STATE], L: int):
    def elbo_fn(params: jax.Array, rng_key) -> jax.Array:
        guide.update_params(params)
        if L == 1:
            X, lq = guide.sample_and_log_prob(rng_key)
            lp = logdensity(X)
            elbo = lp - lq
        else:
            def _elbo_step(elbo: jax.Array, sample_key) -> Tuple[jax.Array, None]:
                X, lq = guide.sample_and_log_prob(sample_key)
                lp = logdensity(X)
                return elbo + (lp - lq), None
            elbo, _ = jax.lax.scan(_elbo_step, jnp.array(0., float), jax.random.split(rng_key, L))
            elbo = elbo / L
        return elbo
    
    def advi_step(advi_state: ADVIState[OPTIMIZER_STATE], rng_key) -> Tuple[ADVIState[OPTIMIZER_STATE], jax.Array]:
        iteration, optimizer_state = advi_state
        params = optimizer.get_params_fn(optimizer_state)
        elbo, elbo_grad = jax.value_and_grad(elbo_fn, argnums=0)(params, rng_key)
        new_optimizer_state = optimizer.update_fn(iteration, -elbo_grad, optimizer_state)
        return ADVIState(iteration + 1, new_optimizer_state), elbo
    
    return advi_step


@jax.jit
def step(carry, any):
  val, key = carry
  key1, key2 = jax.random.split(key)
  val = val + jax.random.normal(key1)
  return (val,key2), None
  
@jax.jit
def f(seed):
  return jax.lax.scan(step, (jnp.array(0.,float), seed), length=10**7)[0][0]


optimizer = Adam(0.005)
X = (jnp.array(0.,float),jnp.array(0.,float),jnp.array(0.,float))
g = Meanfield(X, 0.1)
advi_step = make_advi_step(logdensity, g, optimizer, 1)
keys = jax.random.split(jax.random.PRNGKey(0), 25_000)
start_wall = time.perf_counter()
start_cpu = time.process_time()
# result = f(jax.random.PRNGKey(0))
_, result = jax.lax.scan(advi_step, ADVIState(0, optimizer.init_fn(g.get_params())), keys)
print(result)
end_wall = time.perf_counter()
end_cpu = time.process_time()
cpu_count = os.cpu_count()

wall_time = end_wall - start_wall
cpu_time = end_cpu - start_cpu
print(f"cpu usage {cpu_time/wall_time:.1f}/{cpu_count} wall_time:{wall_time:.1f}s")

# jax[cuda] GPU
# [-25.38362  -27.829231 -26.376095 ... 110.65305  112.43516  110.49699 ]
# cpu usage 1.0/32 wall_time:8.1s

# jax[cuda] CPU
# [-25.38362  -27.829231 -26.376095 ... 110.65344  112.4355   110.49655 ]
# cpu usage 30.7/32 wall_time:13.3s

# jax[cpu]
# [-25.38362  -27.829231 -26.376095 ... 110.65344  112.4355   110.49655 ]
# cpu usage 30.5/32 wall_time:14.5s

# jax[cpu] taskset -c 0
# [-25.38362  -27.829231 -26.376095 ... 110.65363  112.43542  110.49643 ]
# cpu usage 1.0/32 wall_time:5.3s

# jax[cpu] taskset -c 0-31
# [-25.38362  -27.829231 -26.376095 ... 110.65344  112.4355   110.49655 ]
# cpu usage 30.6/32 wall_time:15.1s

# jax[cpu] taskset -c 0-15
# [-25.38362  -27.829231 -26.376095 ... 110.65334  112.43528  110.49664 ]
# cpu usage 15.3/32 wall_time:8.8s

# jax[cpu] taskset -c 0-3
# [-25.38362  -27.829231 -26.376095 ... 110.65319  112.43548  110.496475]
# cpu usage 3.9/32 wall_time:6.6s

# m2 pro
# [-25.383612 -27.829231 -26.376095 ... 110.65322  112.43553  110.496506]
# cpu usage 1.1/10 wall_time:5.9s


# evaluation/gp/gp.py, exit after first phase

# jax[cpu]
# cpu usage 26.5/32 wall_time:335.4s
# Total compilation time: 70.421s (20.99%)

# jax[cpu] taskset -c 0
# cpu usage 1.0/32 wall_time:233.1s
# Total compilation time: 113.935s (48.89%)

# jax[cuda] GPU
# cpu usage 1.1/32 wall_time:188.4s
# Total compilation time: 114.726s (60.88%)

# jax[cuda] CPU
# cpu usage 26.6/32 wall_time:342.1s
# Total compilation time: 71.183s (20.81%)

# m2 pro
# cpu usage 1.6/10 wall_time:173.9s
# Total compilation time: 65.693s (37.77%)

# f

# jax[cuda] GPU
# cpu usage 1.0/32 wall_time:46.8s

# jax[cuda] CPU
# cpu usage 1.0/32 wall_time:4.1s

# jax[cpu]
# cpu usage 1.0/32 wall_time:3.8s

# jax[cpu] taskset -c 0
# cpu usage 1.0/32 wall_time:4.1s

# m2 pro
# cpu usage 1.0/10 wall_time:4.4s