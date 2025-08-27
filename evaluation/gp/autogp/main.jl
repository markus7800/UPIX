
import AutoGP
import CSV
import Dates
import DataFrames
using PyPlot: plt, gcf

println("#Threads: ", Threads.nthreads())

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

n_particles = 256
println("#Particles: ", n_particles)

model = AutoGP.GPModel(df_train.ds, df_train.y; n_particles=n_particles, config=config);

AutoGP.seed!(0)

@time AutoGP.fit_smc!(model; schedule=AutoGP.Schedule.linear_schedule(n_train, .10), n_mcmc=75, n_hmc=10, verbose=true);

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
top_10_ix = [i for (w, i, k) in kernels_sorted[1:10]]

plt.scatter(1:length(weights), weights)
plt.show()
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
plt.show()

import Statistics: mean, std, quantile

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
plt.show()


