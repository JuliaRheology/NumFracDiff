"""
    init_workspace(prob::NumDiffProblem) -> NumDiffWorkspace

    Create a reusable workspace for numerical fractional differentiation.

    Precomputes convolution weights based on the given problem `prob` and
    allocates a buffer to store the derivative of a signal of length `prob.n`.

    The returned workspace should be reused across calls to `compute!`
    to **avoid repeated allocations** and improve performance, especially
    in loops or optimization routines.

    # Arguments
    - `prob::NumDiffProblem` : Numerical differentiation problem definition.

    # Returns
    A `NumDiffWorkspace` containing:
    - `weights` : Precomputed convolution weights.
    - `deriv` : Preallocated vector for storing the derivative.
"""
function init_workspace(prob::NumDiffProblem; L::Int64 = prob.n)

    # Preallocate workspace buffers
    weights = zeros(NumDiffFloat, L)
    deriv   = zeros(NumDiffFloat, L)

    # Create workspace structure
    ws = NumDiffWorkspace(weights, deriv)

    # Fill weights in-place
    generate_weights!(prob.method, prob, ws, L=L)

    return ws
end


"""
    optimal_L(data, prob; tol=1e-2)

    Compute optimal memory length L (in samples) for the Short-Memory method.
    
    # Arguments

    - data : input signal
    - prob : NumDiffProblem (must contain dt and order)
    - tol  : tolerated tail error
    - safety : safety factor (>1 increases robustness)

    Returns
    -------
    Integer memory length (number of time steps).
"""
function optimal_L!(data::Vector{NumDiffFloat}, prob::NumDiffProblem; tol::Float64 = 1e-1)
                   
    method = prob.method

    if !(method isa Union{
            GLShortMem,
            GLShortMemThreads,
            GLShortMemCorr,
            GLShortMemCorrThreads,
            RLShortMem,
            RLShortMemThreads,
            RLShortMemCorr,
            RLShortMemCorrThreads
        })
        @warn "Short-memory length not valid for $(typeof(method))"
        return nothing
    end

    α  = prob.order
    dt = prob.dt

    # maximum magnitude of the signal
    M = maximum(abs, data)

    # if M == 0
    #     return 1
    # end
    try
        # continuous memory length
        L_cont = (M / (tol * gamma(1 - α)))^(1/α)

        # convert to number of time steps
        L = Int(ceil(L_cont / dt))

        # clamp to reasonable bounds
        L = clamp(L, Int(length(data) ÷ 10), length(data))

        prob._L = L
    catch e
        prob._L = length(data)
    end

end

"""
    update_order!(prob::NumDiffProblem, ws::NumDiffWorkspace, order_new::Float64)

    Update the fractional derivative order of a numerical differentiation problem and recomputes the weights for the current method.

    # Arguments
    - `prob::NumDiffProblem` : The problem struct containing `dt`, `order`, `n`, and `method`.
    - `ws::NumDiffWorkspace` : Workspace containing precomputed weights and derivative storage.
    - `order_new::Float64`  : The new fractional order to apply.
"""

function update_order!(prob, ws, order_new)
    # If the order is the same, keep existing weights and continue
    if order_new == prob.order
        return
    end

    # Otherwise update order and regenerate weights
    prob.order = order_new
    generate_weights!(prob.method, prob, ws)
end