
import AutoGP
import CSV
import Dates
import DataFrames
using PyPlot: plt, gcf
using UUIDs
import Statistics: mean, std, quantile
import Hwloc
using Printf

function main()
    n_particles = parse(Int, ARGS[1])
    show_plots = parse(Bool, ARGS[2])
    n_threads = Threads.nthreads()
    println("#Threads: ", n_threads)
    println("#Particles: ", n_particles)

    data = CSV.File("evaluation/gp/tsdl.161.csv"; header=[:ds, :y], types=Dict(:ds=>Dates.Date, :y=>Float64));
    df = DataFrames.DataFrame(data)

    n_test = 18
    n_train = DataFrames.nrow(df) - n_test
    df_train = df[1:end-n_test, :]
    df_test = df[end-n_test+1:end, :]

    # node_dist_leaf = AutoGP.GP.normalize([0., 0, 0, 0, 1,])
    # noise = AutoGP.Model.transform_param(:noise,-3.)
    # noise = nothing
    # config = AutoGP.GP.GPConfig(node_dist_leaf=node_dist_leaf, max_depth=1, noise=noise)

    # config = AutoGP.GP.GPConfig(max_depth=1)

    config = AutoGP.GP.GPConfig()

    println(config)


    model = AutoGP.GPModel(df_train.ds, df_train.y; n_particles=n_particles, config=config);

    AutoGP.seed!(0)

    t0 = time_ns()
    Base.cumulative_compile_timing(true)
    t0_comp = Base.cumulative_compile_time_ns()

    AutoGP.fit_smc!(model; schedule=AutoGP.Schedule.linear_schedule(n_train, .10), n_mcmc=75, n_hmc=10, verbose=true);
    
    Base.cumulative_compile_timing(false);
    t1_comp = Base.cumulative_compile_time_ns()
    t1 = time_ns()

    comp_time = (t1_comp[1] - t0_comp[1]) / 10^9 # second is re-compile time
    wall_time = (t1 - t0) / 10^9

    weights = AutoGP.particle_weights(model)
    kernels = AutoGP.covariance_kernels(model)

    kernels_sorted = sort(collect(zip(weights, 1:length(weights), kernels)), rev=true)
    function print_top_10()
        count = 0
        for (w, i, k) in kernels_sorted
            println("Model $(i), Weight $(w), Noise $(model.pf_state.traces[i][:noise])")
            count += 1
            if count <= 10
                display(k)
            end
        end
    end
    print_top_10()
    top_10_ix = [i for (w, i, k) in kernels_sorted[1:min(length(kernels_sorted),10)]]

    if show_plots
        plt.figure()
        plt.scatter(1:length(weights), weights)
        plt.xlabel("Particle")
        plt.ylabel("Weight")
        # plt.show()
        # println("Noises:")
        # println([t[:noise] for t in model.pf_state.traces])

        println("log_marginal_likelihood_estimate: ", AutoGP.log_marginal_likelihood_estimate(model))
        println()

        ds_future = range(start=df_test.ds[end]+Dates.Month(1), step=Dates.Month(1), length=12*3)
        ds_query = vcat(df_train.ds, df_test.ds, ds_future)
        forecasts = AutoGP.predict(model, ds_query; quantiles=[0.025, 0.975]);

        fig, ax = plt.subplots(figsize=(10,5))
        ax.scatter(df_train.ds, df_train.y, marker=".", color="k", label="Observed Data")
        ax.scatter(df_test.ds, df_test.y, marker=".", color="r", label="Test Data")
        for i=1:AutoGP.num_particles(model)
            if i in top_10_ix
                subdf = forecasts[forecasts.particle.==i,:]
                ax.plot(subdf[!,"ds"], subdf[!,"y_mean"], color="k", linewidth=.5)
                ax.fill_between(
                    subdf.ds, subdf[!,"y_0.025"], subdf[!,"y_0.975"];
                    color="tab:blue", alpha=0.05)
            end
        end
        plt.title("Forecast from top 10 particles")

        fig, ax = plt.subplots(figsize=(10,5))
        ax.scatter(df_train.ds, df_train.y, marker=".", color="k", label="Observed Data")
        ax.scatter(df_test.ds, df_test.y, marker=".", color="r", label="Test Data")

        d = AutoGP.predict_mvn(model, ds_query)
        n_sample = 1_000
        x = rand(d, n_sample)
        m = mean(x, dims=2)
        s = std(x, dims=2)
        q975 = [quantile(x[i,:],0.975) for i in 1:size(x)[1]]
        q025 = [quantile(x[i,:],0.025) for i in 1:size(x)[1]]
        ax.plot(ds_query, m, color="k", linewidth=.5)
        ax.fill_between(
            ds_query, q025, q975;
            color="tab:blue", alpha=0.5)
        plt.title("Forecast from mixture of all particles")
        plt.show()
    end

    id = string(uuid4())
    date = string(Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM"))
    json = """
{
  "id": "$id",
  "workload": {
    "n_particles": $n_particles
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
    "threads": $n_threads
  }
}
"""
    mkpath(@sprintf("experiments/data/gp/autogp/cpu_%02d", n_threads))
    open(@sprintf("experiments/data/gp/autogp/cpu_%02d/nparticles_%07d_cpu_%02d_date_%s_%s.json", n_threads, n_particles, n_threads, date, id[1:8]), "w") do f
        write(f, json)
    end
end

main()