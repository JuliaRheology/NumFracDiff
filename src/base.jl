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
function init_workspace(prob::NumDiffProblem)

    # Preallocate workspace buffers
    weights = zeros(NumDiffFloat, prob.n)
    deriv   = zeros(NumDiffFloat, prob.n)

    # Create workspace structure
    ws = NumDiffWorkspace(weights, deriv)

    # Fill weights in-place
    generate_weights!(prob.method, prob, ws)

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
function optimal_L!(data::Vector{NumDiffFloat}, prob::NumDiffProblem; tol::Float64 = 1e-2)
                   
    method = prob.method

    if !(method isa Union{
            RL,
            RLThreads,
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

    # continuous memory length
    L_cont = (M / (tol * gamma(1 - α)))^(1/α)

    # convert to number of time steps
    L = Int(ceil(L_cont / dt))

    # clamp to reasonable bounds
    L = clamp(L, Int(length(data) ÷ 10), length(data))

    method.L = L

end

