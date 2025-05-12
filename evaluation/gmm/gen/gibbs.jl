
@gen function update_w(tr)
    n::Int, k::Int = get_args(tr)..., tr[:k]
    counts = zeros(k)
    for i=1:n
        counts[tr[:z => i]::Int] += 1
    end
    w ~ dirichlet(δ .+ counts)
end

@gen function update_means(tr)
    n::Int, k::Int = get_args(tr)..., tr[:k]
    for j=1:k
        y_js_sum::Float64 = 0.
        y_js_length::Int = 0
        for i in 1:n
            if tr[:z => i] == j
                y_js_length += 1
                y_js_sum += tr[:y => i]::Float64
            end
        end
        n_j::Int, σ²_j::Float64 = y_js_length, tr[:σ² => j]
        if y_js_length > 0
            {:μ => j} ~ gaussian((y_js_sum/σ²_j + κ * ξ)/(n_j/σ²_j + κ), 1/(n_j/σ²_j + κ))
        end
    end
end

@gen function update_vars(tr)
    n::Int, k::Int = get_args(tr)..., tr[:k]
    for j=1:k
        μ_j::Float64 = tr[:μ => j]

        y_js_sum_minus_μ_j_sq::Float64 = 0.
        y_js_length::Int = 0
        for i in 1:n
            if tr[:z => i] == j
                y_js_length += 1
                y_js_sum_minus_μ_j_sq += (tr[:y => i]::Float64 - μ_j)^2
            end
        end
        n_j::Int = y_js_length

        if y_js_length > 0
            {:σ² => j} ~ inv_gamma(α + n_j/2, β + y_js_sum_minus_μ_j_sq/2)
        end
    end
end

@gen function update_allocations(tr)
    n::Int, k::Int, w::Vector{Float64} = get_args(tr)..., tr[:k], tr[:w]
    μs = Float64[tr[:μ => j]::Float64 for j=1:k]
    σ²s = Float64[tr[:σ² => j]::Float64 for j=1:k]
    for i=1:n
        y_i::Float64 = tr[:y => i]
        p = Float64[exp(logpdf(gaussian, y_i, μ, σ²)) for (μ, σ²) in zip(μs, σ²s)] .* w
        {:z => i} ~ categorical(p ./ sum(p))
    end
end