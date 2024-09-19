function constructGoalPriors()
    eps = 0.1
    
    C_0_k = kron(ones(4), [eps, eps, 10.0, eps])

    return [C_0_k, C_0_k]
end


function initializeGoalAgent(A, B, C_0, D; t_maze_model::Function)
    iterations = 50 # Iterations of variational algorithm
    C_s = deepcopy(C_0)
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
        model = t_maze_model(A=A, C_s=C_s, D=D, x=x)
    
        # Define constraints
        constraints    = structured()
        initialization = init_marginals(C_s)

        for (i, j) in pols
            data = (u = [B[i], B[j]],)
        
            res = infer(model          = model,
                        constraints    = constraints, 
                        data           = data,
                        initialization = initialization,
                        iterations     = iterations,
                        free_energy    = true)
        
            G[i, j] = res.free_energy[end]/log(2) # Convert to bits
            if t === 3 # Return posterior statistics after learning
                C_s = Vector{Vector}(undef, 2)
                C_s[1] = res.posteriors[:c][end][1].alpha
                C_s[2] = res.posteriors[:c][end][2].alpha
            end
        end
    
        return (G, C_s)
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