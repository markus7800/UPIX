###################
# Proper Gaussian #
###################

struct Gaussian <: Distribution{Float64} end

"""
    gaussian(mu::Real, var::Real)
Samples a `Float64` value from a normal distribution.
"""
const gaussian = Gaussian()

function Gen.logpdf(::Gaussian, x::Real, mean::Real, var::Real)
    @assert var > 0
    diff = x - mean
    -(diff * diff)/ (2.0 * var) - 0.5 * log(2.0 * pi * var)
end

Gen.random(::Gaussian, mu::Real, var::Real) = mu + sqrt(var) * randn()
Gen.is_discrete(::Gaussian) = false

(::Gaussian)(mu, var) = random(gaussian, mu, var)

Gen.has_output_grad(::Gaussian) = false
Gen.has_argument_grads(::Gaussian) = (false, false)