# The secondary World defines the Markov blanket between the secondary and the primary agent 
function initializeSecondaryWorld()
    # The secondary execute simulates the interaction of primary agent with its environment
    function execute_prime(s::Int64, a_prime::Int64)
        as = Vector{Int64}(undef, 2) # Primary actions per time
        os = Vector{Vector}(undef, 2) # Primary observations (one-hot) per time
        reset(s, a_prime) # Reset primary agent for next trial
        
        # Simulate interaction of primary agent with T-maze
        for t=1:2
              G_t = inference(t, as, os, a_prime)
            as[t] = act(t, G_t)
                    execute(as[t])
            os[t] = observe()
        end

        a_s = deepcopy(as) # Register primary action sequence
    end

    a_s = Vector{Int64}(undef, 2) # Predefine primary action sequence
    function observe_prime() # Accepts action sequence of primary agent
        # Secondary agent observes CV if primary agent has tried to visit the cue position (NC otherwise)
        if 4 in a_s # Offer is made when action to cue is proposed
                  # CV   NC
            return [1.0, 0.0]
        else
            return [0.0, 1.0]
        end
    end

    return (execute_prime, observe_prime)
end
;