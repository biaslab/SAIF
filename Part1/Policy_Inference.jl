# Activate the project environment
using Pkg
Pkg.activate("..")  # Uncomment if you need to instantiate packages
# Pkg.instantiate()

# Import necessary packages
using RxInfer, LinearAlgebra, Distributions, Random

# Set random seed for reproducibility
Random.seed!(666)

# Load all helper modules
include("transition_mixture/transition_mixture.jl")
include("transition_mixture/marginals.jl")
include("transition_mixture/in.jl")
include("transition_mixture/out.jl")
include("transition_mixture/switch.jl")
include("../goal_observation.jl")
include("helpers.jl")

# Define PointMassFormConstraint optimizer
import RxInfer.default_point_mass_form_constraint_optimizer
import RxInfer.PointMassFormConstraint

function default_point_mass_form_constraint_optimizer(::Type{Univariate}, ::Type{Discrete}, constraint::PointMassFormConstraint, distribution)
    out = zeros(length(probvec(distribution)))
    out[argmax(probvec(distribution))] = 1.0
    PointMass(out)
end

# Define the probabilistic model using the @model macro
@model function t_maze(A, D, B1, B2, B3, B4, T, c)
    z_0 ~ Categorical(D)
    z_prev = z_0
    for t in 1:T
        switch[t] ~ Categorical(fill(1.0 / 4.0, 4))
        z[t] ~ TransitionMixture(z_prev, switch[t], B1, B2, B3, B4)
        c[t] ~ GoalObservation(z[t], A) where {dependencies = GeneralizedPipeline(vague(Categorical, 8))}
        z_prev = z[t]
    end
end

# Define pointmass constraints
@constraints function pointmass_q()
    q(switch) :: PointMassFormConstraint()
end

# Define meta information for the model
@meta function t_maze_meta()
    GoalObservation(c, z) -> GeneralizedMeta()
end

# Initialization function for marginals
@initialization function init_marginals()
    q(z) = Categorical(fill(1.0 / 8.0, 8))
end

# Configure experiment parameters
T = 2  # Planning horizon
its = 10  # Number of inference iterations to run

# Generate necessary matrices using helper function
A, B, C, D = constructABCD(0.9, [2.0 for _ in 1:T], T)  # Generate the matrices we need

# Run initial inference
result = infer(
    model = t_maze(A = A, D = D, B1 = B[1], B2 = B[2], B3 = B[3], B4 = B[4], T = T),
    data = (c = C,),
    initialization = init_marginals(),
    meta = t_maze_meta(),
    iterations = its,
)

# Inspect results
println("Posterior controls as T=1, ", probvec.(result.posteriors[:switch][end][1]), "\n")
println("Posterior controls as T=2, ", probvec.(result.posteriors[:switch][end][2]))

# Repeat experiments with pointmass constraints
result = infer(
    model = t_maze(A = A, D = D, B1 = B[1], B2 = B[2], B3 = B[3], B4 = B[4], T = T),
    data = (c = C,),
    initialization = init_marginals(),
    meta = t_maze_meta(),
    constraints = pointmass_q(),
    iterations = its,
)

println("Posterior controls as T=1, ", result.posteriors[:switch][end][1].point, "\n")
println("Posterior controls as T=2, ", result.posteriors[:switch][end][2].point, "\n")
