# import jax
# import jax.numpy as np

import scipy
import numpy as np

obs = np.array([0,1] * 5)
biased = True

def log_Z(N: int):
    if not biased and N == 1:
        return -np.inf
    n_obs0 = np.sum(obs == 0)
    n_obs1 = np.sum(obs == 1)
    lp = -np.inf
    for n_black in range(0,N+1):
        lp_n_black = scipy.stats.binom.logpmf(n_black, N, 0.5)
        r = n_black / N # probability of picking a black ball
        if biased:
            # picked black, observed white
            l01 = scipy.stats.bernoulli.logpmf(0, 0.8) + scipy.stats.bernoulli.logpmf(1, r)
            # picked white, observed white
            l00 = scipy.stats.bernoulli.logpmf(0, 0.2) + scipy.stats.bernoulli.logpmf(0, r)
            # observed white
            l0 = np.logaddexp(l01, l00)
            # picked black, observed black
            l11 = scipy.stats.bernoulli.logpmf(1, 0.8) + scipy.stats.bernoulli.logpmf(1, r)
            # picked white, observed black
            l10 = scipy.stats.bernoulli.logpmf(1, 0.2) + scipy.stats.bernoulli.logpmf(0, r)
            # observed black
            l1 = np.logaddexp(l11, l10)
            # observed n_obs0 white, n_obs1 black
            lp_n_black += n_obs0 * l0 + n_obs1 * l1
        else:
            # picked white = observed white
            l0 = scipy.stats.bernoulli.logpmf(0, r)
            # picked black = observed black
            l1 = scipy.stats.bernoulli.logpmf(1, r)
            lp_n_black += n_obs0 * l0 + n_obs1 * l1
        lp = np.logaddexp(lp, lp_n_black)
    lp += scipy.stats.poisson.logpmf(N, 6)
    return lp

log_Zs = np.hstack([log_Z(N) for N in range(1,256)])

# print(log_Zs)

ps = np.exp(log_Zs - scipy.special.logsumexp(log_Zs))
print(ps[:20])

np.save("evaluation/urn/gt_ps.npy", ps)