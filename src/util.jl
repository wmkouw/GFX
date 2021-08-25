"Utility functions"

import LinearAlgebra: inv
import ForneyLab: cholinv
import SpecialFunctions: digamma
export noisecov

function cholinv(M::Matrix{Float64})
    return LinearAlgebra.inv(M)
end

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
    elseif dims >= 3
        G = [Δt^n /factorial(n) for n in dims:-1:1] 
        g = [Δt^(2n) /factorial(n-1) for n in dims+1:-1:2]
        Q = (G*G' + diagm(g))
    end
    return Q
end