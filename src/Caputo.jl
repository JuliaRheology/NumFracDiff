"""
    generate_weights!(::Caputo, order::NumDiffFloat, prob::NumDiffProblem, ws::NumDiffWorkspace)

    Compute the convolution weights for the Caputo fractional derivative
    and store them **in-place** in the workspace `ws`.

    # Arguments

    - `::Caputo` : Dispatch type indicating the Caputo method.

    - `prob::NumDiffProblem` : Numerical differentiation problem definition.

    - `ws::NumDiffWorkspace` : Workspace containing preallocated `weights` field.
    Must have at least `n` elements.

"""

function generate_weights!(::Union{Caputo, CaputoThreads}, prob::NumDiffProblem, ws::NumDiffWorkspace;L::Int64=prob.n)
    α = 1.0 - prob.order
    ws.weights[1] = 1.0  # first weight
    prev_pow = 1.0       # start with 1^α
    
    for k in 2:L
        curr_pow = k^α
        ws.weights[k] = curr_pow - prev_pow
        prev_pow = curr_pow
    end
end

"""
    compute!(method, ws::NumDiffWorkspace, data::Vector{NumDiffFloat}, prob::NumDiffProblem)

    Compute the Caputo fractional derivative of the input `data` and store the result in `ws.deriv`.

    This function has two method variants:

    1. **Serial version** (`::Caputo`): computes the derivative in a simple serial loop.  
    2. **Parallel version** (`::CaputoThreads`): computes the derivative using `Threads.@threads` over the output indices.  

    # Arguments
    - `method::DiffMethod`: method used for the numerical differentiation
    - `ws::NumDiffWorkspace` : workspace containing precomputed weights (`ws.weights`) and preallocated derivative buffer (`ws.deriv`).
    - `data::Vector{NumDiffFloat}` : input signal or time series to differentiate.
    - `prob::NumDiffProblem` : problem definition, containing `order`, `dt`, and `L`.
"""

function compute!(::Caputo, ws::NumDiffWorkspace, data::Vector{Float64}, prob::NumDiffProblem)
    n = prob.n
    α = prob.order
    dt = prob.dt
    weights = ws.weights
    d = deepcopy(data)
    C = dt^(-α) / gamma(2.0 - α)

    # First element at t=0
    ws.deriv[1] = 0.0

    # Loop over all other points
    @inbounds for i in 2:n
        tmp = 0.0
        # sum over past increments
        for j in 1:(i-1)
            tmp += weights[j] * (d[i-j+1] - d[i-j])
        end
        # Add the first data point contribution
        tmp += weights[i] * d[1]
        ws.deriv[i] = C * tmp
    end
end


function compute!(::CaputoThreads, ws::NumDiffWorkspace, data::Vector{NumDiffFloat}, prob::NumDiffProblem)
    n = prob.n
    α = prob.order
    dt = prob.dt
    d = deepcopy(data)
    weights = ws.weights
    C = dt^(-α) / gamma(2.0 - α)

    # First element is simple
    @inbounds ws.deriv[1] = 0.0

    # Parallel loop over indices 2..n
    @threads :dynamic for i in 2:n
        tmp = 0.0
        @simd for j in 1:(i-1)
            tmp += weights[j] * (d[i-j+1] - d[i-j])
        end
        # Add the first data point contribution
        tmp += weights[i] * d[1]
        @inbounds ws.deriv[i] = C * tmp
    end

end