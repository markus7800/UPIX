import jax
import jax.numpy as jnp
from dccxjax.distributions import Distribution, DIST_SUPPORT, DIST_SUPPORT_LIKE
import numpyro.distributions as numpyro_dists
from ..core.model_slp import Model
from ..core.samplecontext import SampleContext
from typing import Optional, Dict, cast, Tuple, Set, List
from ..types import FloatArrayLike, FloatArray, Trace, PRNGKey
from .variable_selector import VariableSelector, SingleVariable
from abc import ABC, abstractmethod

__all__ = [
    "lmh"
]

class SingleAddressProposal(ABC):
    @abstractmethod
    def propose(self, rng_key: PRNGKey, current: jax.Array) -> Tuple[jax.Array, FloatArray]:
        raise NotImplementedError
    
    @abstractmethod
    def assess(self, current: jax.Array, proposed: jax.Array) -> FloatArray:
        raise NotImplementedError

class StaticSingleAddressProposal(SingleAddressProposal):
    def __init__(self, proposal_dist: Distribution) -> None:
        self.proposal_dist = proposal_dist

    def propose(self, rng_key: PRNGKey, current: jax.Array) -> Tuple[jax.Array, FloatArray]:
        proposed = self.proposal_dist.sample(rng_key)
        return proposed, self.proposal_dist.log_prob(proposed) 
    
    def assess(self, current: jax.Array, proposed: jax.Array) -> FloatArray:
        return self.proposal_dist.log_prob(proposed) 
    
class LMHCtx(SampleContext):
    def __init__(self, rng_key: PRNGKey, X_current: Trace, resample_address: str, resample_proposal: Optional[SingleAddressProposal]) -> None:
        self.rng_key = rng_key
        self.X_current = X_current
        self.resample_address = resample_address
        self.resample_proposal = resample_proposal
        self.X_proposed: Trace = dict()
        self.log_priors_proposed: Dict[str, FloatArray] = dict()
        self.log_likelihood_proposed = jnp.array(0., float)
        self.log_resample_factor = jnp.array(0., float)

    def sample(self, address: str, distribution: Distribution[DIST_SUPPORT, DIST_SUPPORT_LIKE], observed: Optional[DIST_SUPPORT_LIKE] = None) -> DIST_SUPPORT:
        if observed is not None:
            self.log_likelihood_proposed += distribution.log_prob(observed).sum()
            return cast(DIST_SUPPORT, observed)
        
        if address == self.resample_address:
            current_value = self.X_current[address]
            self.rng_key, proposal_key = jax.random.split(self.rng_key)
            resample_proposal = self.resample_proposal if self.resample_proposal is not None else StaticSingleAddressProposal(distribution)
            proposed_value, forward_log_prob = resample_proposal.propose(proposal_key, current_value)
            backward_log_prob = resample_proposal.assess(proposed_value, current_value)
            self.log_resample_factor = backward_log_prob - forward_log_prob
            value: DIST_SUPPORT = cast(DIST_SUPPORT, proposed_value)
        elif address in self.X_current:
            value: DIST_SUPPORT = cast(DIST_SUPPORT, self.X_current[address])
        else:
            self.rng_key, proposal_key = jax.random.split(self.rng_key)
            value: DIST_SUPPORT = distribution.sample(proposal_key)
            
        self.X_proposed[address] = value
        self.log_priors_proposed[address] = distribution.log_prob(value)
        return value

    def logfactor(self, lf: FloatArrayLike, address: str) -> None:
        self.log_likelihood_proposed += lf
        

def lmh(m: Model, resample_selector: VariableSelector, X: Trace, rng_key: PRNGKey):
    log_priors_current: Dict[str, FloatArray] = dict()
    log_likelihood_current: FloatArray = jnp.array(0., float)
    resample_addresses: List[str] = []
    for addr, (log_prob, is_observed) in m.log_prob_trace(X).items():
        if is_observed:
            log_likelihood_current += log_prob
        else:
            log_priors_current[addr] = log_prob
            if resample_selector.contains(addr):
                resample_addresses.append(addr)
    
    select_key, lmh_key = jax.random.split(rng_key)
    resample_address = resample_addresses[jax.random.randint(select_key, (), 0, len(resample_addresses))]
    with LMHCtx(lmh_key, X, resample_address, None) as ctx:
        m()
    
    acceptance_log_prob = jnp.log(len(resample_addresses) / sum(resample_selector.contains(addr) for addr in ctx.X_proposed.keys())) + ctx.log_resample_factor
    acceptance_log_prob += ctx.log_likelihood_proposed - log_likelihood_current
    acceptance_log_prob += sum((ctx.log_priors_proposed[addr] - log_priors_current[addr] for addr in log_priors_current.keys() & ctx.log_priors_proposed.keys()), start=jnp.array(0,float))
    acceptance_log_prob = jax.lax.min(acceptance_log_prob, 0.)

    return ctx.X_proposed, acceptance_log_prob