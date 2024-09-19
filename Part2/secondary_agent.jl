function constructSecondaryPriors()
    eps = 0.1
    A_prime_0 = eps*ones(2, L) # Maps offer (index of αs) to (CV, NV)
end

function initializeSecondaryAgent(A_prime_0)
    iterations = 50 # Iterations of variational algorithm

    A_prime_s = deepcopy(A_prime_0) # Initialize prior
    function inference_prime(t::Int64, a_prime::Union{Int64, Missing}, o_prime::Union{Vector, Missing})
        # Define possible policies
        G_prime = Vector{Union{Float64, Missing}}(missing, L)
        if t === 1
            pols = 1:L # Enumerate all possible offers
        elseif t === 2
            pols = a_prime # Register made offer
        end
        
        # Define (un)observed data for meta objects
        x_prime = deepcopy(o_prime)
    
        for i in pols
            α = αs[i] # Select offer value

            # Convert offer to one-hot control
            u_prime = zeros(L)
            u_prime[i] = 1.0

            # Define model
            model = t_maze_secondary(A_prime_s=A_prime_s, x_prime=x_prime, alpha_s=u_prime)

            # Utility to secondary agent depends on offer
            C_prime = softmax((1-α)*[c, -c])
            data = (c_prime = C_prime,)

            # Define constraints
            constraints = structured(t<2)
            initialization = init_marginals_secondary(A_prime_s)
    
            res = infer(model          = model,
                        data           = data,
                        constraints    = constraints,
                        initialization = initialization,
                        iterations     = iterations,
                        free_energy    = true)
                        
            G_prime[i] = mean(res.free_energy[10:iterations])./log(2) # Average to smooth fluctuations and convert to bits
            if t === 2 # Return posterior statistics after learning
                A_prime_s = res.posteriors[:A_prime][end].a
            end
        end
    
        return (G_prime, A_prime_s)
    end

    function act_prime(G_prime)
        p = softmax(-10.0*G_prime) # Sharpen for minimum selection (fixed precision)
        pol = rand(Categorical(p)) # Select a policy
        
        return pol # Select from possible actions
    end

    return (inference_prime, act_prime)
end
;