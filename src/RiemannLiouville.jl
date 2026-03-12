"""
    generate_weights!(::RL, prob::NumDiffProblem, ws::NumDiffWorkspace)

    Compute the convolution weights for the Riemann–Liouville (RL) fractional derivative
    and store them **in-place** in `ws.weights`.

    # Arguments
    - `::RL` : dispatch type for RL methods Union{RL, RLThreads, RLShortMemCorr, RLShortMemCorrThreads}
    - `prob::NumDiffProblem` : numerical differentiation problem definition
    - `ws::NumDiffWorkspace` : workspace containing preallocated `weights` vector (length ≥ `prob.n`)
"""
function generate_weights!(::Union{RL, RLThreads,RLShortMem,RLShortMemThreads, RLShortMemCorr, RLShortMemCorrThreads}, prob::NumDiffProblem, ws::NumDiffWorkspace)
    
    alpha = 1.0 - prob.order
    n = prob.n

    ws.weights[1] = 1.0

    p1 = 0.0   # (0.0^alpha)
    p2 = 1.0   # (1.0^alpha)

    for j in 2:n
        p3 = j^alpha
        ws.weights[j] = p3 - 2.0*p2 + p1
        
        p1 = p2
        p2 = p3
    end
end


"""
    compute!(method, ws::NumDiffWorkspace, data::Vector{NumDiffFloat}, prob::NumDiffProblem)

    Compute the Riemann-Liouville fractional derivative of the input signal `data` and store the result
    directly in `ws.deriv`. This function has two variants depending on `method`:

    1. **Serial version** (`method::RL`) 
    2. **Parallel version** (`method::RLThreads`) 

    # Arguments

    - `method::RL` or `RLThreads`: Method descriptor. Must contain field `L::NumDiffInt` specifying 
    the short-memory length.  
    - `ws::NumDiffWorkspace`: Workspace containing precomputed `weights` and the derivative vector `deriv`.  
    The derivative will be written **in-place** to avoid additional memory allocations.  
    - `data::Vector{NumDiffFloat}`: Input signal or time series to differentiate.  
    - `prob::NumDiffProblem`: Problem definition containing the derivative order `order`, time step `dt`, 
    and signal length `n`.

    # Returns

    - `ws.deriv` updated **in-place** with the computed fractional derivative.
"""
function compute!(method::RL, ws::NumDiffWorkspace, data::Vector{NumDiffFloat}, prob::NumDiffProblem)

    n  = prob.n
    α  = prob.order
    dt = prob.dt

    weights = ws.weights

    C = dt^(-α) / gamma(2.0 - α)

    @inbounds ws.deriv[1] = C * weights[1] * data[1]

    @inbounds for i in 2:n
        acc = zero(NumDiffFloat)

        for j in 1:i
            acc += weights[j] * data[i-j+1]
        end

        ws.deriv[i] = C * acc
    end


end

function compute!(method::RLThreads, ws::NumDiffWorkspace, data::Vector{NumDiffFloat}, prob::NumDiffProblem)

    n  = prob.n
    α  = prob.order
    dt = prob.dt

    weights = ws.weights

    C = dt^(-α) / gamma(2.0 - α)

    # first element is scalar, no threading needed
    @inbounds ws.deriv[1] = C * weights[1] * data[1]

    @threads :dynamic for i in 2:n
        acc = zero(NumDiffFloat)
        @simd for j in 1:i
            @inbounds acc += weights[j] * data[i-j+1]
        end
        @inbounds ws.deriv[i] = C * acc
    end
end

"""
# Short-memory option

Both serial and threaded versions support the short-memory approximation: if `prob._L < prob.n`,  
the computation uses only the last `L` points of `data` for indices `i > L` (fixed memory).  
For `i <= L`, the algorithm uses all available past points (`i` points, growing memory).  
If `prob._L` is set to `typemax(Int)` or a value ≥ `prob.n`, the full memory algorithm is used.

"""

function compute!(method::RLShortMem, ws::NumDiffWorkspace, data::Vector{NumDiffFloat}, prob::NumDiffProblem)

    n  = prob.n
    α  = prob.order
    dt = prob.dt
    optimal_L!(data,prob)
    L=prob._L

    weights = ws.weights

    C = dt^(-α) / gamma(2.0 - α)

    @inbounds ws.deriv[1] = C * weights[1] * data[1]

    # ---------- Region 1: growing memory ----------
    grow_end = min(n, L)

    @inbounds for i in 2:grow_end
        acc = zero(NumDiffFloat)

        for j in 1:i
            acc += weights[j] * data[i-j+1]
        end

        ws.deriv[i] = C * acc
    end

    # ---------- Region 2: fixed memory ----------
    if L < n
        @inbounds for i in (L+1):n
            acc = zero(NumDiffFloat)

            for j in 1:L
                acc += weights[j] * data[i-j+1]
            end

            ws.deriv[i] = C * acc
        end
    end

end

function compute!(method::RLShortMemThreads, ws::NumDiffWorkspace, data::Vector{NumDiffFloat}, prob::NumDiffProblem)

    n  = prob.n
    α  = prob.order
    dt = prob.dt
    optimal_L!(data,prob)
    L=prob._L
    weights = ws.weights

    C = dt^(-α) / gamma(2.0 - α)

    # first element is scalar, no threading needed
    @inbounds ws.deriv[1] = C * weights[1] * data[1]

    # ---------- Region 1: growing memory ----------
    grow_end = min(n, L)
    @threads :dynamic for i in 2:grow_end
        acc = zero(NumDiffFloat)
        @simd for j in 1:i
            @inbounds acc += weights[j] * data[i-j+1]
        end
        @inbounds ws.deriv[i] = C * acc
    end

    # ---------- Region 2: fixed memory ----------
    if L < n
        @threads :dynamic for i in (L+1):n
            acc = zero(NumDiffFloat)
            @simd for j in 1:L
                @inbounds acc += weights[j] * data[i-j+1]
            end
            @inbounds ws.deriv[i] = C * acc
        end
    end

end



"""
    compute!(method, ws::NumDiffWorkspace, data::Vector{NumDiffFloat}, prob::NumDiffProblem)

    Compute the **Riemann–Liouville fractional derivative** using the **short-memory approximation with correction** 
    and store the result directly in `ws.deriv`. This function has two variants depending on `method`:

    1. **Serial version** (`RLShortMemCorr`)  
    2. **Parallel version** (`RLShortMemCorrThreads`)  

    # Description

    The algorithm splits the computation into two regions:

    - **Region 1 (growing memory)**: For indices `i ≤ L`, the derivative is computed using all available past data points (`i` points).  
    - **Region 2 (fixed memory with correction)**: For indices `i > L`, only the last `L` points are used, and a correction term is applied to reduce truncation error.  

    # Arguments

    - `method::RLShortMemCorr` or `RLShortMemCorrThreads` : Method descriptor specifying the short-memory length `L`.  
    - `ws::NumDiffWorkspace` : Workspace containing precomputed convolution weights (`ws.weights`) and preallocated derivative vector (`ws.deriv`).  
    - `data::Vector{NumDiffFloat}` : Input signal or time series to differentiate.  
    - `prob::NumDiffProblem` : Problem definition containing the derivative order `order`, time step `dt`, and signal length `n`.

    # Returns

    - `ws.deriv` updated **in-place** with the computed fractional derivative.
"""


function compute!(method::RLShortMemCorr, ws::NumDiffWorkspace, data::Vector{NumDiffFloat}, prob::NumDiffProblem)
    n  = prob.n
    order  = prob.order
    dt = prob.dt
    weights = ws.weights
    optimal_L!(data,prob)
    L=prob._L

    # Precompute constants
    C = dt^(-order) / gamma(2.0 - order)
    one_minus_order = 1.0 - order
    k_0 = 1.0 / gamma(one_minus_order)
    k_1 = -order / gamma(2.0 - order)

    # Precompute offsets at L
    offset_0 = k_0 * L^(-order)
    offset_1 = k_1 * L^(one_minus_order)

    # Precompute powers for correction region (i > L)
    pow_neg_order = [(i)^(-order) for i in (L+1):n]
    pow_one_minus_order = [(i)^(one_minus_order) for i in (L+1):n]

    # ---------- Region 1: growing memory ----------
    for i in 1:min(L, n)
        acc = zero(NumDiffFloat)
        @simd for j in 1:i
            acc += weights[j] * data[i-j+1]
        end
        ws.deriv[i] = C * acc
    end

    # ---------- Region 2: fixed memory with correction ----------
    if L < n
        for idx in 1:(n-L)
            i = L + idx
            acc = zero(NumDiffFloat)

            # convolution over last L points
            @simd for j in 1:L
                acc += weights[j] * data[i-j+1]
            end

            # compute correction term
            m_0 = k_0 * pow_neg_order[idx] - offset_0
            m_1 = k_1 * pow_one_minus_order[idx] - offset_1

            y_0 = data[i-L]
            y_1 = i != L+1 ? y_0 - data[i-L-1] : 0.0

            correction = y_0 * m_0 + y_1 * (L * m_0 - m_1)

            ws.deriv[i] = C * (acc + correction)
        end
    end

end

function compute!(method::RLShortMemCorrThreads, ws::NumDiffWorkspace, data::Vector{NumDiffFloat}, prob::NumDiffProblem)
    n  = prob.n
    order  = prob.order
    dt = prob.dt
    weights = ws.weights
    optimal_L!(data,prob)
    L  = prob._L

    # Precompute constants
    C = dt^(-order) / gamma(2.0 - order)
    one_minus_order = 1.0 - order
    k_0 = 1.0 / gamma(one_minus_order)
    k_1 = -order / gamma(2.0 - order)

    # Precompute offsets at L
    offset_0 = k_0 * L^(-order)
    offset_1 = k_1 * L^(one_minus_order)

    # Precompute powers for correction region (i > L) - TO BE CHECKED
    pow_neg_order = [(i)^(-order) for i in (L+1):n]
    pow_one_minus_order = [(i)^(one_minus_order) for i in (L+1):n]

    # ---------- Region 1: growing memory (i ≤ L) ----------
    @threads :dynamic for i in 1:min(L, n)
        acc = zero(NumDiffFloat)
        @simd for j in 1:i
            @inbounds acc += weights[j] * data[i-j+1]
        end
        @inbounds ws.deriv[i] = C * acc
    end

    # ---------- Region 2: fixed memory with correction (i > L) ----------
    if L < n
        @threads :dynamic for idx in 1:(n-L)
            i = L + idx
            acc = zero(NumDiffFloat)

            # convolution over last L points
            @simd for j in 1:L
                @inbounds acc += weights[j] * data[i-j+1]
            end

            # compute correction term
            m_0 = k_0 * pow_neg_order[idx] - offset_0
            m_1 = k_1 * pow_one_minus_order[idx] - offset_1

            @inbounds y_0 = data[i-L]
            @inbounds y_1 = i != L+1 ? y_0 - data[i-L-1] : 0.0

            correction = y_0 * m_0 + y_1 * (L * m_0 - m_1)

            @inbounds ws.deriv[i] = C * (acc + correction)
        end
    end

end