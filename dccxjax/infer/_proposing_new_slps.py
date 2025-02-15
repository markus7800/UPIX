



class DecisionRepresentativeCtx(SampleContext):
    def __init__(self, partial_X: Trace, rng_key: PRNGKey) -> None:
        self.rng_key = rng_key
        self.partial_X = partial_X
        self.X: Trace = dict()
    def sample(self, address: str, distribution: dist.Distribution, observed: Optional[jax.Array] = None) -> jax.Array:
        if observed is not None:
            return observed
        if address in self.partial_X:
            value = self.partial_X[address]
        else:
            self.rng_key, sample_key = jax.random.split(self.rng_key)
            value = distribution.sample(sample_key)
        self.X[address] = value
        return value
    
def decision_representative_from_partial_trace(model: Model, partial_X: Trace, rng_key: PRNGKey):
    with DecisionRepresentativeCtx(partial_X, rng_key) as ctx:
        model()
        return ctx.X


def propose_new_slps_from_last_positions(slp: SLP, last_positions: Trace, active_slps: List[SLP], proposed_slps: Dict[SLP,int], n_chains: int, rng_key: PRNGKey, scale: float):
    if len(slp.branching_variables) == 0:
        return
    
    # we want to find a jittered_position such that decision_representative_from_partial_trace(jittered_position) is not in support of any SLP
    # in decision_representative_from_partial_trace we run the model and take values from jittered_position where possible else sample from prior
    # in this process we must encounter at least one decision (provided that len(slp.branching_variables) > 0)
    # furthermore all SLPs share at least one (this first encountered) decision
    # thus SLPs share at least one branching variable
    # so we randomly perturb the values in last_positions for all branching_variables (= jittered_position)
    # the addresses of dr := decision_representative_from_partial_trace(jittered_position) may be different from the addresses of jittered_position / last_positions
    # for each other_slp how can we test if dr is in the support by probing with jittered_position?
    # dr and jittered_position have the same path in the model until they disagree on one decision
    # up to this decision they sample the same addresses (a subset of the addresses of jittered_position)
    # It may happen that due to the random pertubation, dr and jittered_position have the same path until a sample statement is encountered where its address is missing in jittered_position
    # in this case, dr may or may not be in the support of other_slp


    # shape of values is (n_chains, val(dims))
    jittered_positions: Trace = dict()
    for addr, values in last_positions.items():
        if addr in slp.branching_variables:
            rng_key, sample_key = jax.random.split(rng_key)
            jittered_positions[addr] = values + scale * jax.random.normal(sample_key, values.shape)
        else:
            jittered_positions[addr] = values

    for i in range(n_chains):
        partial_X: Trace = {addr: values[i,:] if len(values.shape) == 2 else values[i] for addr, values in jittered_positions.items()}
        rng_key, sample_key = jax.random.split(rng_key)
        decision_representative = decision_representative_from_partial_trace(slp.model, partial_X, sample_key)

        in_support_of_any_slp = False
        for other_slp in active_slps:
            if other_slp.path_indicator(decision_representative):
                in_support_of_any_slp = True
                break
        for other_slp, count in proposed_slps.items():
            if other_slp.path_indicator(decision_representative):
                in_support_of_any_slp = True
                proposed_slps[other_slp] = count + 1
        
        if not in_support_of_any_slp:
            new_slp = slp_from_decision_representative(slp.model, decision_representative)
            proposed_slps[new_slp] = 1
            print(f"Proposed new slp {new_slp.short_repr()}", new_slp.formatted())

