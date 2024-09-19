function constructPriors()
    eps = 0.1
    
    # Position 1 surely does not offer disambiguation
    A_0_1 = [10.0 10.0;
             10.0 10.0;
             eps  eps;
             eps  eps]

    # But the other positions might
    A_0_X = [1.0  eps;
             eps  1.0;
             eps  eps;
             eps  eps]
    
    A_0 = eps*ones(16, 8) # Vague prior on everything else

    A_0[1:4, 1:2] = A_0_1
    A_0[5:8, 3:4] = A_0_X
    A_0[9:12, 5:6] = A_0_X
    A_0[13:16, 7:8] = A_0_X

    # Agent knows it starts at position 1
    D_0 = zeros(8)
    D_0[1:2] = [0.5, 0.5]

    return (A_0, D_0)
end

function initializeAgent(A_0, B, C, D_0)
    iterations = 50 # Iterations of variational algorithm
    A_s = deepcopy(A_0)
    D_s = deepcopy(D_0)
    function inference(t::Int64, a::Vector, o::Vector)
        # Define possible policies
        G = Matrix{Union{Float64, Missing}}(missing, 4, 4)
        if t === 1
            pols = [(1,1), (1,2), (1,3), (1,4), (2,1), (3,1), (4,1), (4,2), (4,3), (4,4)]
        elseif t === 2
            a1 = a[1] # Register first move
            if a1 in [2, 3]
                pols = [(a1,1)] # Mandatory move to 1
            else
                pols = [(a1,1), (a1,2), (a1,3), (a1,4)]
            end
        elseif t === 3
            a1 = a[1] # Register both moves
            a2 = a[2]
            pols = [(a1, a2)]
        end
    
        # Define (un)observed data for meta objects
        x = Vector{Union{Vector{Float64}, Missing}}(missing, 2)
        for k=1:2
            if isassigned(o, k) # Observed datapoint
                x[k] = o[k]
            end
        end
    
        # Define model
        model = t_maze(A_s=A_s, D_s=D_s, x=x)
    
        # Define constraints
        constraints    = structured(t<3) # Sampling approximation for t<3
        initialization = init_marginals(A_s)

        for (i, j) in pols
            data = (u = [B[i], B[j]],
                    c = [C, C])
    
            res = infer(model          = model,
                        constraints    = constraints, 
                        data           = data,
                        initialization = initialization,
                        iterations     = iterations,
                        free_energy    = true)
            
            G[i, j] = mean(res.free_energy[10:iterations])./log(2) # Average to smooth fluctuations and convert to bits
            if t === 3 # Return posterior statistics after learning
                A_s = res.posteriors[:A][end].a
            end
        end
    
        return (G, A_s)
    end
    
    function act(t, G)
        # We include policy selection in the act function for clearer code; procedurally, policy selection belongs in the plan step
        idx = findall((!).(ismissing.(G))) # Find coordinates of non-missing entries
        Gvec = G[idx] # Convert to vector of valid entries
        p = softmax(-10.0*Gvec) # Sharpen for minimum selection
        pol = rand(Categorical(p)) # Select a policy
        
        return idx[pol][t] # Select current action from policy
    end

    return (inference, act)
end