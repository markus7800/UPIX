
@gen function update_w(tr)
    n, k = get_args(tr)..., tr[:k]
    counts = zeros(k)
    for i=1:n
        counts[tr[:z => i]] += 1
    end
    w ~ dirichlet(δ * ones(k) + counts)
end

@gen function update_means(tr)
    n, k = get_args(tr)..., tr[:k]
    for j=1:k
        y_js = [tr[:y => i] for i=1:n if tr[:z => i] == j]
        n_j, μ_j, σ²_j = length(y_js), tr[:μ => j], tr[:σ² => j]
        if !isempty(y_js)
            {:μ => j} ~ gaussian((sum(y_js)/σ²_j + κ * ξ)/(n_j/σ²_j + κ),
                                 1/(n_j/σ²_j + κ))
        end
    end
end

@gen function update_vars(tr)
    n, k = get_args(tr)..., tr[:k]
    for j=1:k
        y_js = [tr[:y => i] for i=1:n if tr[:z => i] == j]
        n_j, μ_j, σ²_j = length(y_js), tr[:μ => j], tr[:σ² => j]
        if !isempty(y_js)
            {:σ² => j} ~ inv_gamma(α + n_j/2, β + sum((y_js .- μ_j).^2)/2)
        end
    end
end

@gen function update_allocations(tr)
    n, k, w = get_args(tr)..., tr[:k], tr[:w]
    μs = [tr[:μ => j] for j=1:k]
    σ²s = [tr[:σ² => j] for j=1:k]
    for i=1:n
        y_i = tr[:y => i]
        p = [exp(logpdf(gaussian, y_i, μ, σ²)) for (μ, σ²) in zip(μs, σ²s)] .* w
        {:z => i} ~ categorical(p ./ sum(p))
    end
end