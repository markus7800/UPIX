# adapted from https://github.com/mugamma/gmm.git
using Logging: global_logger
using TerminalLoggers: TerminalLogger
using Gen
using ProgressLogging: @progress
using UUIDs
using Dates
using Hwloc
using Printf
using Trapz
global_logger(TerminalLogger(right_justify=120))

include("dirichlet.jl")
include("gaussian.jl")
include("gibbs.jl")
include("rjmcmc.jl")

const λ = 3
const δ = 5.0
const ξ = 0.0
const κ = 0.01
const α = 2.0
const β = 10.0

@dist positive_poisson(λ) = poisson(λ - 1) + 1

@gen function gmm(n)
    k ~ positive_poisson(λ)
    w ~ dirichlet(δ * ones(k))
    means, vars = zeros(k), zeros(k)
    for j=1:k
        means[j] = ({:μ => j} ~ gaussian(ξ, 1/κ))
        vars[j] = ({:σ² => j} ~ inv_gamma(α, β))
    end
    for i=1:n
        z = ({:z => i} ~ categorical(w))
        {:y => i} ~ gaussian(means[z], vars[z])
    end
end

function mcmc_kernel(tr)
    tr, acc = mh(tr, update_w, ())
    @assert acc "update_w not gibbs"
    tr, acc = mh(tr, update_means, ())
    @assert acc "update_means not gibbs"
    tr, acc = mh(tr, update_vars, ())
    @assert acc "update_vars not gibbs"
    tr, acc = mh(tr, update_allocations, ())
    @assert acc "update_allocations not gibbs"
    tr, = split_merge(tr)
    tr
end

ys = [-7.87951290075215, -23.251364738213493, -5.34679518882793, -3.163770449770572,
10.524424782864525, 5.911987013277482, -19.228378698266436, 0.3898087330050574,
8.576922415766697, 7.727416085566447, -18.043123523482492, 9.108136117789305,
29.398734347901787, 2.8578485031858003, -20.716691460295685, -18.5075008084623,
-21.52338318392563, 10.062657028986715, -18.900545157827718, 3.339430437507262,
3.688098690412526, 4.209808727262307, 3.371091291010914, 30.376814419984456,
12.778653273596902, 28.063124205174137, 10.70527515161964, -18.99693615834304,
8.135342537554163, 29.720363913218446, 29.426043027354385, 28.40516772785764,
31.975585225366686, -20.642437143912638, 30.84807631345935, -21.46602061526647,
12.854676808303978, 30.685416799345685, 5.833520737134923, 7.602680172973942,
10.045516408942117, 28.62342173081479, -20.120184774438087, -18.80125468061715,
12.849708921404385, 31.342270731653656, 4.02761078481315, -19.953549865339976,
-2.574052170014683, -21.551814470820258, -2.8751904316333268,
13.159719198798443, 8.060416669497197, 12.933573330915458, 0.3325664001681059,
11.10817217269102, 28.12989207125211, 11.631846911966806, -15.90042467317705,
-0.8270272159702201, 11.535190070081708, 4.023136673956579,
-22.589713328053048, 28.378124912868305, -22.57083855780972,
29.373356677376297, 31.87675796607244, 2.14864533495531, 12.332798078071061,
8.434664672995181, 30.47732238916884, 11.199950328766784, 11.072188217008367,
29.536932243938097, 8.128833670186253, -16.33296115562885, 31.103677511944685,
-20.96644212192335, -20.280485886015406, 30.37107537844197, 10.581901339669418,
-4.6722903116912375, -20.320978011296315, 9.141857987635252, -18.6727012563551,
7.067728508554964, 5.664227155828871, 30.751158861494442, -20.198961378110013,
-4.689645356611053, 30.09552608716476, -19.31787364001907, -22.432589846769154,
-0.9580412415863696, 14.180597007125487, 4.052110659466889,
-18.978055134755582, 13.441194891615718, 7.983890038551439, 7.759003567480592]

inference_constraints = choicemap([(:y => i, y) for (i, y) in enumerate(ys)]...);

function run_inference(seed::Int, N::Int)
    Gen.Random.seed!(seed)
    mcmc_tr, = generate(gmm, (length(ys),), inference_constraints)
    Ks = zeros(Int, N)

    for i=1:N
        k = mcmc_tr[:k]
        Ks[i] = k
        mcmc_tr = mcmc_kernel(mcmc_tr)
    end
    return Int[sum(Ks .== i) for i in 1:maximum(Ks)]
end


function main()
    t0 = time_ns()
    Base.cumulative_compile_timing(true)
    t0_comp = Base.cumulative_compile_time_ns()
    run_inference(0, 100) # run short chain to compile on main thread

    n_chains = parse(Int, ARGS[1])
    n_samples_per_chain = parse(Int, ARGS[2])
    nthreads = Threads.nthreads()
    println("n_chains=$n_chains, n_samples_per_chain=$n_samples_per_chain, nthreads=$nthreads")
    N = n_samples_per_chain
    result = Vector{Vector{Int}}(undef, n_chains)

    Threads.@threads for i in 1:n_chains
        result[i] = run_inference(i, N)
    end
    max_k = maximum(length(r) for r in result)
    cumulative_result = zeros(Int, max_k)
    for r in result
        for (k, k_sum) in enumerate(r)
            cumulative_result[k] += k_sum
        end
    end
    display(cumulative_result)
    weights = cumulative_result ./ sum(cumulative_result)
    display(weights)

    Base.cumulative_compile_timing(false);
    t1_comp = Base.cumulative_compile_time_ns()
    t1 = time_ns()

    comp_time = (t1_comp[1] - t0_comp[1]) / 10^9 # second is re-compile time
    wall_time = (t1 - t0) / 10^9


    gt_cluster_visits = [687, 574, 119783, 33258676, 46000324, 16768787, 3302321, 485045, 57502, 5806, 457, 38]
    gt_ps = gt_cluster_visits / sum(gt_cluster_visits)
    gt_cdf = cumsum(gt_ps)
    weights = vcat(weights, zeros(length(gt_ps) - length(weights)))
    cdf_est = cumsum(weights)

    W1_distance = trapz(1:length(cdf_est), abs.(cdf_est .- gt_cdf))
    infty_distance = maximum(abs.(cdf_est .- gt_cdf))
    println("W1_distance=$W1_distance infty_distance=$infty_distance comp_time=$comp_time wall_time=$wall_time")

    id = string(uuid4())
    date = string(Dates.format(now(), "yyyy-mm-dd_HH-MM"))
    json = """
{
  "id": "$id",
  "workload": {
    "n_chains": $n_chains,
    "n_samples_per_chain": $n_samples_per_chain
  },
  "timings": {
    "inference_time": $(wall_time - comp_time),
    "compilation_time": $(comp_time),
    "wall_time": $(wall_time)
  },
  "environment_info": {
    "platform": "cpu",
    "cpu-brand": "$(Sys.cpu_info()[1].model)",
    "cpu_count": $(Hwloc.num_virtual_cores()),
    "threads": $nthreads
  },
  "result": {
    "counts": $(repr(cumulative_result))
  }
}
"""
    mkpath(@sprintf("experiments/data/gmm/rjmcmc/cpu_%02d", nthreads))
    open(@sprintf("experiments/data/gmm/rjmcmc/cpu_%02d/nchains_%07d_niter_%d_cpu_%02d_date_%s_%s.json", nthreads, n_chains, n_samples_per_chain, nthreads, date, id[1:8]), "w") do f
        write(f, json)
    end
end
main()

