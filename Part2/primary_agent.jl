function initializePrimaryAgent(B, C, D)
    iterations = 50 # Iterations of variational algorithm
    function inference(t::Int64, a::Vector, o::Vector, a_prime::Int64) # Inference depends on offer by secondary agent
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
        A_s = constructPrimaryA(αs[a_prime]) # Offer is encoded by primary observation matrix
        model = t_maze_primary(A_s=A_s, D=D, x=x)
        
        # Define constraints
        initialization = init_marginals_primary()

        for (i, j) in pols
            data = (u = [B[i], B[j]],
                    c = [C, C])
        
            res = infer(model          = model,
                        data           = data,
                        initialization = initialization,
                        iterations     = iterations,
                        free_energy    = true)
            
            G[i, j] = res.free_energy[end]/log(2) # Convert to bits
        end
    
        return G
    end
    
    function act(t::Int64, G::Matrix)
        idx = findall((!).(ismissing.(G))) # Find coordinates of non-missing entries
        Gvec = G[idx] # Convert to vector of valid entries
        p = softmax(-10.0*Gvec) # Sharpen for minimum selection
        pol = rand(Categorical(p)) # Select a policy
        
        return idx[pol][t] # Select current action from policy
    end

    return (inference, act)
end
;