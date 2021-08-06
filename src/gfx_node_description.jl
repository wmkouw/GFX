using ForneyLab
using LinearAlgebra
import ForneyLab: SoftFactor, @ensureVariables, generateId, addNode!, associate!,
                  averageEnergy, differentialEntropy, Interface, Variable, slug, ProbabilityDistribution,
                  unsafeLogMean, unsafeMean, unsafeCov, unsafePrecision, unsafeMeanCov, cholinv
import SpecialFunctions: digamma                  
export GeneralisedFilterX, GFX

"""
Description:

    A Gaussian node for transitions between states in generalised coordinates:

    𝒩(y | A(θ)x + B(η)u, V(τ)),

    where for instance for an order-2 differential 
        
        A(x) = (S + sθ')x, 

    with S = | 1 Δt | and s = |  0  | ,
             | 0  1 |         | -Δt | 

        B(η) = b η ,

    with b = |  0 | ,
             | Δt |
    and
        V(τ) = τ^-1 *[Δt^3/3   Δt^2/2;
                      Δt^2/2       Δt];

    Interfaces:
        1. y (output vector)
        2. θ (state coefficients)
        3. x (generalised coordinates)
        4. η (control coefficients)
        5. u (exogenous input)
        6. τ (precision)

    Construction:
        GeneralisedFilterX(y, θ, x, η, u, τ, Δt=1.0, id=:some_id)

"""

mutable struct GeneralisedFilterX <: SoftFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    # Sampling time
    Δt::Float64

    function GeneralisedFilterX(y, θ, x, η, u, τ; Δt::Float64=1.0, id=generateId(GeneralisedFilterX))
        @ensureVariables(y, x, θ, η, u, τ)
        self = new(id, Array{Interface}(undef, 6), Dict{Symbol,Interface}(), Δt)
        addNode!(currentGraph(), self)
        self.i[:y] = self.interfaces[1] = associate!(Interface(self), y)
        self.i[:θ] = self.interfaces[2] = associate!(Interface(self), θ)
        self.i[:x] = self.interfaces[3] = associate!(Interface(self), x)
        self.i[:η] = self.interfaces[4] = associate!(Interface(self), η)
        self.i[:u] = self.interfaces[5] = associate!(Interface(self), u)
        self.i[:τ] = self.interfaces[6] = associate!(Interface(self), τ)
        return self
    end
end

slug(::Type{GeneralisedFilterX}) = "GFX"

function averageEnergy(::Type{GeneralisedFilterX},
                       marg_y::ProbabilityDistribution{Multivariate},
                       marg_x::ProbabilityDistribution{Multivariate},
                       marg_θ::ProbabilityDistribution{Multivariate},
                       marg_η::ProbabilityDistribution{Univariate},
                       marg_u::ProbabilityDistribution{Univariate},
                       marg_τ::ProbabilityDistribution{Univariate})

    #TODO
    error("not implemented yet")
end
