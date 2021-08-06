import LinearAlgebra: I, Bidiagonal, tr
import ForneyLab: unsafeCov, unsafeMean, unsafePrecision, VariateType,
				  collectNaiveVariationalNodeInbounds, assembleClamp!, ultimatePartner
include("util.jl")

export ruleVariationalGFXOutNPPPPP,
       ruleVariationalGFXIn1PNPPPP,
       ruleVariationalGFXIn2PPNPPP,
       ruleVariationalGFXIn3PPPNPP,
	   ruleVariationalGFXIn4PPPPNP,
	   ruleVariationalGFXIn5PPPPPN


function ruleVariationalGFXOutNPPPPP(Δt :: Float64,
									 marg_y :: Nothing,
									 marg_θ :: ProbabilityDistribution{Multivariate},
                                     marg_x :: ProbabilityDistribution{Multivariate},
                                     marg_η :: ProbabilityDistribution{Univariate},
                                     marg_u :: ProbabilityDistribution{Univariate},
                                     marg_τ :: ProbabilityDistribution{Univariate})
	
	# Extract moments of beliefs
	mθ = unsafeMean(marg_θ)
	mx = unsafeMean(marg_x)
	mη = unsafeMean(marg_η)
	mu = unsafeMean(marg_u)
	mτ = unsafeMean(marg_τ)

	# Set order of system
	order = dims(marg_θ)

	# Define helper matrices
	S = Bidiagonal(ones(order,), Δt.*ones(order-1,), :U)
	s = zeros(order,); s[end] = Δt

	# Generate covariance matrix
	EVγ = inv(mτ)*noisecov(Δt, dims=order)

	# Compute transition matrices
	EAθ = S + s*mθ'
	EBη = s*mη	

	# Set outgoing message
	return Message(Multivariate, GaussianMeanVariance, m=EAθ*mx + EBη*mu, v=EVγ)
end

function ruleVariationalGFXIn1PNPPPP(Δt :: Float64,
									 marg_y :: ProbabilityDistribution{Multivariate},
                                     marg_θ :: Nothing,
									 marg_x :: ProbabilityDistribution{Multivariate},
                                     marg_η :: ProbabilityDistribution{Univariate},
                                     marg_u :: ProbabilityDistribution{Univariate},
                                     marg_τ :: ProbabilityDistribution{Univariate})

	# Extract moments of beliefs
	my,Vy = unsafeMeanCov(marg_y)
	mx,Vx = unsafeMeanCov(marg_x)
	mη,Vη = unsafeMeanCov(marg_η)
	mu,Vu = unsafeMeanCov(marg_u)
	mτ = unsafeMean(marg_τ)

	# Set order of system
	order = dims(marg_x)

	# Define helper matrices
	S = Bidiagonal(ones(order,), Δt.*ones(order-1,), :U)
	s = zeros(order,); s[end] = Δt

	# Cast precision to matrix
	EVγ = inv(mτ)*noisecov(Δt, dims=order)

	# Compute transition matrix
	EBη = s*mη

	# Set parameters
	ϕ = mx*s'*EVγ*(my - EBη*mu) - EVγ*S'*(mx*mx' + Vx)*s
	# ϕ = mx*s'*EVγ*(my - EBη*mu) - (mx*mx' + Vx)*S'*EVγ*s

	Φ = EVγ*(s'*(mx*mx' + Vx)*s)
	# Φ = (s'*s)*EVγ*(mx*mx' + Vx)	

	# Set outgoing message
	return Message(Multivariate, GaussianWeightedMeanPrecision, xi=ϕ, w=Φ)
end

function ruleVariationalGFXIn2PPNPPP(Δt :: Float64,
									 marg_y :: ProbabilityDistribution{Multivariate},
									 marg_θ :: ProbabilityDistribution{Multivariate},
									 marg_x :: Nothing,
									 marg_η :: ProbabilityDistribution{Univariate},
								     marg_u :: ProbabilityDistribution{Univariate},
                                     marg_τ :: ProbabilityDistribution{Univariate})

   	# Extract moments of beliefs
	my,Vy = unsafeMeanCov(marg_y)
	mθ,Vθ = unsafeMeanCov(marg_θ)
	mη,Vη = unsafeMeanCov(marg_η)
	mu,Vu = unsafeMeanCov(marg_u)
	mτ = unsafeMean(marg_τ)

	# Set order of system
	order = dims(marg_θ)

	# Define helper matrices
	S = Bidiagonal(ones(order,), Δt.*ones(order-1,), :U)
	s = zeros(order,); s[end] = Δt

	# Cast precision to matrix
	mV = inv(mτ)*noisecov(Δt, dims=order)
	mW = mτ*inv(noisecov(Δt, dims=order))

	# Compute transition matrices
	EA = S + s*mθ'
	EB = s*mη

	# Set parameters
	ϕ = EA'*mW*(my - EB*mu)
	Φ = S'*mW*S + S'*mW*s*mθ' + mθ*s'*mW*S + s'*mW*s*(Vθ + mθ*mθ')

	# Set outgoing message
	return Message(Multivariate, GaussianWeightedMeanPrecision, xi=ϕ, w=Φ)
end

function ruleVariationalGFXIn3PPPNPP(Δt :: Float64,
									 marg_y :: ProbabilityDistribution{Multivariate},
									 marg_θ :: ProbabilityDistribution{Multivariate},
									 marg_x :: ProbabilityDistribution{Multivariate},
							  	     marg_η :: Nothing,
								     marg_u :: ProbabilityDistribution{Univariate},
                                     marg_τ :: ProbabilityDistribution{Univariate})

	# Extract moments of beliefs
	my = unsafeMean(marg_y)
	mθ = unsafeMean(marg_θ)
	mx = unsafeMean(marg_x)
	mu,vu = unsafeMeanCov(marg_u)
	mτ = unsafeMean(marg_τ)

	# Set order
	order = dims(marg_θ)

	# Define helper matrices
	S = Bidiagonal(ones(order,), Δt.*ones(order-1,), :U)
	s = zeros(order,); s[end] = Δt

	# Compute expected values
	EA = S + s*mθ'

	# Cast precision to matrix
	mW = wMatrix(mτ, order, Δt=Δt)

	# Set parameters
	ϕ = mu'*s'*mW*(my - EA*mx)
	Φ = mτ*(mu*mu' + vu)

	# Set outgoing message
	return Message(Univariate, GaussianWeightedMeanPrecision, xi=ϕ, w=Φ)
end

function ruleVariationalGFXIn4PPPPNP(Δt :: Float64,
									 marg_y :: ProbabilityDistribution{Multivariate},
									 marg_θ :: ProbabilityDistribution{Multivariate},
									 marg_x :: ProbabilityDistribution{Multivariate},
									 marg_η :: ProbabilityDistribution{Univariate},
								     marg_u :: Nothing,
                                     marg_τ :: ProbabilityDistribution{Univariate})

   
	# Extract moments of beliefs
	my,Vy = unsafeMeanCov(marg_y)
	mθ,Vθ = unsafeMeanCov(marg_θ)
	mx,Vx = unsafeMeanCov(marg_x)
	mη,Vη = unsafeMeanCov(marg_η)
	mτ = unsafeMean(marg_τ)

	# Set order
	order = dims(marg_θ)

	# Define helper matrices
	S = Bidiagonal(ones(order,), Δt.*ones(order-1,), :U)
	s = zeros(order,); s[end] = -Δt

	# Cast precision to matrix
	mW = wMatrix(mτ, order, Δt=Δt)
	
	# Compute expected values
	EA = S + s*mθ'

	# Set parameters
	ϕ = mη'*s'*mW*(my - EA*mx)
	Φ = mτ*(mη*mη' + Vη)

	# Set outgoing message
	return Message(Univariate, GaussianWeightedMeanPrecision, xi=ϕ, w=Φ)
end

function ruleVariationalGFXIn5PPPPPN(Δt :: Float64,
									 marg_y :: ProbabilityDistribution{Multivariate},
									 marg_θ :: ProbabilityDistribution{Multivariate},
									 marg_x :: ProbabilityDistribution{Multivariate},
									 marg_η :: ProbabilityDistribution{Univariate},
								     marg_u :: ProbabilityDistribution{Univariate},
                                     marg_τ :: Nothing)

   
	# Extract moments of beliefs
	my,Vy = unsafeMeanCov(marg_y)
	mθ,Vθ = unsafeMeanCov(marg_θ)
	mx,Vx = unsafeMeanCov(marg_x)
	mη,Vη = unsafeMeanCov(marg_η)
	mu,Vu = unsafeMeanCov(marg_u)
	
	# Set order
	order = dims(marg_θ)

	# Define helper matrices
	S = Bidiagonal(ones(order,), Δt.*ones(order-1,), :U)
	s = zeros(order,); s[end] = -Δt
	
	# Compute expected values
	EA = S + s*mθ'
	EB = s*mη

	t1 = my[order]^2 + Vy[order,order]
	t2 = -my[order]*(EA*mx)[order]
	t3 = -my[order]*(EB*mu)[order]
	t4 = -(EA*mx)[order]*my[order]
	t5 = mx'*(EA'*EA + Vθ)*mx + tr((EA'*EA + Vθ)*Vx)
	t6 = (EA*mx)[order]*(EB*mu)[order]
	t7 = -(EB*mu)[order]*my[order]
	t8 = (EB*mu)[order]*(EA*mx)[order]
	t9 = mη'*s'*(mu*mu' + Vu)*s*mη + s'*(mu*mu' + Vu)*s*Vη

	# Set parameters
	a = 3/2
	b = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9

	# Set outgoing message
	return Message(Univariate, Gamma, a=a, b=b)
end

function collectNaiveVariationalNodeInbounds(node::GeneralisedFilterX, entry::ScheduleEntry)
	inbounds = Any[]

	# Push function to calling signature (Δt needs to be defined in user scope)
	push!(inbounds, Dict{Symbol, Any}(:Δt => node.Δt, :keyword => false))

    target_to_marginal_entry = currentInferenceAlgorithm().target_to_marginal_entry

    for node_interface in entry.interface.node.interfaces
        inbound_interface = ultimatePartner(node_interface)
        if node_interface === entry.interface
            # Ignore marginal of outbound edge
            push!(inbounds, nothing)
        elseif (inbound_interface != nothing) && isa(inbound_interface.node, Clamp)
            # Hard-code marginal of constant node in schedule
            push!(inbounds, assembleClamp!(inbound_interface.node, ProbabilityDistribution))
        else
            # Collect entry from marginal schedule
            push!(inbounds, target_to_marginal_entry[node_interface.edge.variable])
        end
    end

    return inbounds
end
