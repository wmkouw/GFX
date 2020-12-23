"Utility functions"

import ForneyLab: cholinv
import SpecialFunctions: digamma

function wMatrix(γ, order; Δt=1.)
    mW = 1e8*Matrix{Float64}(I, order, order)
    mW[end, end] = γ / Δt
    return mW
end

# function cholinv(M::AbstractMatrix)
#     return LinearAlgebra.inv(M)
# end

function mvdigamma(x, order)
    "Multivariate digamma function (see https://en.wikipedia.org/wiki/Multivariate_gamma_function)"
    result = 0.
    for i = 1:order
        result += digamma(x + (1 - i)/2.)
    end
    return result
end