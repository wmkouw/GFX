"Utility functions"

import SpecialFunctions: digamma

function mvdigamma(x, order)
    "Multivariate digamma function (see https://en.wikipedia.org/wiki/Multivariate_gamma_function)"
    result = 0.
    for i = 1:order
        result += digamma(x + (1 - i)/2.)
    end
    return result
end

function noisecov(Δt; dims=2)
    "Construct noise covariance matrix"

    if dims == 1 
        Q = Δt
    elseif dims == 2
        Q = [Δt^3/3   Δt^2/2;
             Δt^2/2       Δt]
    else
        error("Not implemented yet.")
    end

    return Q
end