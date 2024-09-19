using DomainSets
using ForwardDiff: jacobian
using TupleTools: deleteat
using ReactiveMP: FunctionalDependencies, messagein, setmessage!, getlocalclusters, clusterindex, getmarginals
using Base.Broadcast: BroadcastFunction

import ReactiveMP: functional_dependencies

include("distributions.jl")


struct GoalObservation end

@node GoalObservation Stochastic [c, z, A]


#----------
# Modifiers
#----------

# Metas
struct BetheMeta{P} # Meta parameterized by x type for rule overloading
    x::P # Pointmass value for observation
end
BetheMeta() = BetheMeta(missing) # Absent observation

struct GeneralizedMeta{P}
    x::P # Pointmass value for observation
    newton_iterations::Int64
end
GeneralizedMeta() = GeneralizedMeta(missing, 20)
GeneralizedMeta(point) = GeneralizedMeta(point, 20)

# Pipelines
struct BethePipeline <: FunctionalDependencies end
struct GeneralizedPipeline <: FunctionalDependencies
    init_message::Union{Bernoulli, Categorical}

    GeneralizedPipeline() = new() # If state is clamped, then no inital message is required
    GeneralizedPipeline(init_message::Union{Bernoulli, Categorical}) = new(init_message)
end

function functional_dependencies(::BethePipeline, factornode, interface, iindex)
    message_dependencies = ()
    
    clusters = getlocalclusters(factornode)
    marginal_dependencies = getmarginals(clusters) # Include all node-local marginals

    return message_dependencies, marginal_dependencies
end

function functional_dependencies(pipeline::GeneralizedPipeline, factornode, interface, iindex)
    clusters = getlocalclusters(factornode)
    cindex = clusterindex(clusters, iindex) # Find the index of the cluster for the current interface

    # Message dependencies
    if (iindex === 2) # Message towards state
        output = messagein(interface)
        setmessage!(output, pipeline.init_message)
        message_dependencies = (interface, )
    else
        message_dependencies = ()
    end

    # Marginal dependencies
    if (iindex === 2) || (iindex === 3) # Message towards state or parameter
        marginal_dependencies = getmarginals(clusters) # Include all marginals
    else
        marginal_dependencies = skipindex(getmarginals(clusters), cindex) # Skip current cluster
    end

    return message_dependencies, marginal_dependencies
end


#------------------------------
# Unobserved Bethe Update Rules
#------------------------------

@rule GoalObservation(:c, Marginalisation) (q_c::Union{Dirichlet, PointMass},
                                            q_z::Union{Bernoulli, Categorical, PointMass}, 
                                            q_A::Union{SampleList, MatrixDirichlet, PointMass}, 
                                            meta::BetheMeta{Missing}) = begin
    log_c = mean(BroadcastFunction(log), q_c)
    z = probvec(q_z)
    log_A = mean(BroadcastFunction(log), q_A)

    # Compute internal marginal
    x = softmax(log_A*z + log_c)

    return Dirichlet(x .+ 1)
end

@rule GoalObservation(:z, Marginalisation) (q_c::Union{Dirichlet, PointMass},
                                            q_z::Union{Bernoulli, Categorical}, 
                                            q_A::Union{SampleList, MatrixDirichlet, PointMass}, 
                                            meta::BetheMeta{Missing}) = begin
    log_c = mean(BroadcastFunction(log), q_c)
    z = probvec(q_z)
    log_A = mean(BroadcastFunction(log), q_A)

    # Compute internal marginal
    x = softmax(log_A*z + log_c)

    return Categorical(softmax(log_A'*x))
end

@rule GoalObservation(:A, Marginalisation) (q_c::Union{Dirichlet, PointMass},
                                            q_z::Union{Bernoulli, Categorical, PointMass}, 
                                            q_A::Union{SampleList, MatrixDirichlet, PointMass}, 
                                            meta::BetheMeta{Missing}) = begin
    log_c = mean(BroadcastFunction(log), q_c)
    z = probvec(q_z)
    log_A = mean(BroadcastFunction(log), q_A)

    # Compute internal marginal
    x = softmax(log_A*z + log_c)

    return MatrixDirichlet(x*z' .+ 1)
end

@average_energy GoalObservation (q_c::Union{Dirichlet, PointMass}, 
                                 q_z::Union{Bernoulli, Categorical, PointMass}, 
                                 q_A::Union{SampleList, MatrixDirichlet, PointMass}, 
                                 meta::BetheMeta{Missing}) = begin
    log_c = mean(BroadcastFunction(log), q_c)
    z = probvec(q_z)
    log_A = mean(BroadcastFunction(log), q_A)

    # Compute internal marginal
    x = softmax(log_A*z + log_c)

    return -x'*(log_A*z + log_c - safelog.(x))
end


#----------------------------
# Observed Bethe Update Rules
#----------------------------

@rule GoalObservation(:c, Marginalisation) (q_c::Union{Dirichlet, PointMass}, # Unused
                                            q_z::Union{Bernoulli, Categorical, PointMass}, 
                                            q_A::Union{SampleList, MatrixDirichlet, PointMass}, 
                                            meta::BetheMeta{<:AbstractVector}) = begin
    return Dirichlet(meta.x .+ 1)
end

@rule GoalObservation(:z, Marginalisation) (q_c::Union{Dirichlet, PointMass},
                                            q_z::Union{Bernoulli, Categorical}, # Unused
                                            q_A::Union{SampleList, MatrixDirichlet, PointMass}, 
                                            meta::BetheMeta{<:AbstractVector}) = begin
    log_A = mean(BroadcastFunction(log), q_A)

    return Categorical(softmax(log_A'*meta.x))
end

@rule GoalObservation(:A, Marginalisation) (q_c::Union{Dirichlet, PointMass},
                                            q_z::Union{Bernoulli, Categorical, PointMass}, 
                                            q_A::Union{SampleList, MatrixDirichlet, PointMass}, # Unused
                                            meta::BetheMeta{<:AbstractVector}) = begin
    z = probvec(q_z)

    return MatrixDirichlet(meta.x*z' .+ 1)
end

@average_energy GoalObservation (q_c::Union{Dirichlet, PointMass}, 
                                 q_z::Union{Bernoulli, Categorical, PointMass}, 
                                 q_A::Union{SampleList, MatrixDirichlet, PointMass}, 
                                 meta::BetheMeta{<:AbstractVector}) = begin
    log_c = mean(BroadcastFunction(log), q_c)
    z = probvec(q_z)
    log_A = mean(BroadcastFunction(log), q_A)

    return -meta.x'*(log_A*z + log_c)
end


#------------------------------------
# Unobserved Generalized Update Rules
#------------------------------------

@rule GoalObservation(:c, Marginalisation) (q_z::Union{Bernoulli, Categorical, PointMass}, 
                                            q_A::Union{SampleList, MatrixDirichlet, PointMass}, 
                                            meta::GeneralizedMeta{Missing}) = begin
    z = probvec(q_z)
    A = mean(q_A)

    return Dirichlet(A*z .+ 1)
end

@rule GoalObservation(:z, Marginalisation) (m_z::Union{Bernoulli, Categorical},
                                            q_c::Union{Dirichlet, PointMass},
                                            q_z::Union{Bernoulli, Categorical},
                                            q_A::Union{SampleList, MatrixDirichlet, PointMass}, 
                                            meta::GeneralizedMeta{Missing}) = begin
    d = probvec(m_z)
    log_c = mean(BroadcastFunction(log), q_c)
    z_0 = probvec(q_z)
    (A, h_A) = mean_h(q_A)

    # Root-finding problem for marginal statistics
    g(z) = z - softmax(-h_A + A'*log_c - A'*safelog.(A*z) + safelog.(d))

    z_k = deepcopy(z_0)
    for k=1:meta.newton_iterations
        z_k = z_k - inv(jacobian(g, z_k))*g(z_k) # Newton step for multivariate root finding
    end

    # Compute outbound message statistics
    rho = softmax(safelog.(z_k) - log.(d .+ 1e-6))

    return Categorical(rho)
end

@rule GoalObservation(:A, Marginalisation) (q_c::Union{Dirichlet, PointMass},
                                            q_z::Union{Bernoulli, Categorical, PointMass}, 
                                            q_A::Union{SampleList, MatrixDirichlet, PointMass},
                                            meta::GeneralizedMeta{Missing}) = begin
    log_c = mean(BroadcastFunction(log), q_c)
    z = probvec(q_z)
    A_bar = mean(q_A)
    M, N = size(A_bar)

    log_mu(A) = (A*z)'*(log_c - safelog.(A_bar*z)) - z'*h(A)

    return ContinuousMatrixvariateLogPdf((RealNumbers()^M, RealNumbers()^N), log_mu)
end

@average_energy GoalObservation (q_c::Union{Dirichlet, PointMass}, 
                                 q_z::Union{Bernoulli, Categorical, PointMass}, 
                                 q_A::Union{SampleList, MatrixDirichlet, PointMass}, 
                                 meta::GeneralizedMeta{Missing}) = begin
    log_c = mean(BroadcastFunction(log), q_c)
    z = probvec(q_z)
    (A, h_A) = mean_h(q_A)

    return z'*h_A - (A*z)'*(log_c - safelog.(A*z))
end


#----------------------------------
# Observed Generalized Update Rules
#----------------------------------

@rule GoalObservation(:c, Marginalisation) (q_z::Union{Bernoulli, Categorical, PointMass}, # Unused
                                            q_A::Union{SampleList, MatrixDirichlet, PointMass}, # Unused
                                            meta::GeneralizedMeta{<:AbstractVector}) = begin
    return Dirichlet(meta.x .+ 1)
end

@rule GoalObservation(:z, Marginalisation) (m_z::Union{Bernoulli, Categorical}, # Unused
                                            q_c::Union{Dirichlet, PointMass}, # Unused
                                            q_z::Union{Bernoulli, Categorical}, # Unused
                                            q_A::Union{SampleList, MatrixDirichlet, PointMass}, 
                                            meta::GeneralizedMeta{<:AbstractVector}) = begin
    log_A = clamp.(mean(BroadcastFunction(log), q_A), -12, 12)
    return Categorical(softmax(log_A'*meta.x))
end

@rule GoalObservation(:A, Marginalisation) (q_c::Union{Dirichlet, PointMass}, # Unused
                                            q_z::Union{Bernoulli, Categorical, PointMass}, 
                                            q_A::Union{SampleList, MatrixDirichlet, PointMass}, # Unused
                                            meta::GeneralizedMeta{<:AbstractVector}) = begin
    z = probvec(q_z)

    return MatrixDirichlet(meta.x*z' .+ 1)
end

@average_energy GoalObservation (q_c::Union{Dirichlet, PointMass}, 
                                 q_z::Union{Bernoulli, Categorical, PointMass}, 
                                 q_A::Union{SampleList, MatrixDirichlet, PointMass}, 
                                 meta::GeneralizedMeta{<:AbstractVector}) = begin
    log_c = mean(BroadcastFunction(log), q_c)
    z = probvec(q_z)
    log_A = clamp.(mean(BroadcastFunction(log), q_A), -12, 12)

    return -meta.x'*(log_A*z + log_c)
end