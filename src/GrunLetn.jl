
"""
    generate_weights!(::GL, order::NumDiffFloat, L::Int, ws::NumDiffWorkspace)

    Compute the convolution weights for the Grunwald-Letnikov fractional derivative
    and store them **in-place** in the workspace `ws`.

    # Arguments

    - `::GL` : Dispatch type indicating the GL method.

    - `prob::NumDiffProblem` : Numerical differentiation problem definition.

    - `ws::NumDiffWorkspace` : Workspace containing preallocated `weights` field.
    Must have at least `n` elements.

"""
function generate_weights!(::Union{GL,GLFFT,GLThreads,GLShortMem,GLShortMemThreads,GLShortMemCorr,GLShortMemCorrThreads}, prob::NumDiffProblem, ws::NumDiffWorkspace; L::Int64 =prob.n)
    
    ws.weights[1] = 1.0

    alpha_plus_one = prob.order + 1.0
    
    for k in 2:L
        ws.weights[k] = ws.weights[k-1] * (1.0 - alpha_plus_one/(k - 1.0)) 
    end

end

"""
    compute!(method::GL, ws::NumDiffWorkspace, data::Vector{NumDiffFloat}, prob::NumDiffProblem)

    Compute the Grünwald-Letnikov fractional derivative of the input signal `data` and store the result
    directly in `ws.deriv`. This function has two variants depending on `method`:

    1. **Serial version** (`method::GL`) 
    2. **Parallel version** (`method::GLThreads`) 

    # Arguments

    - `method::GL` or `GLThreads`: Method descriptor. Must contain field `L::NumDiffInt` specifying 
    the short-memory length.  
    - `ws::NumDiffWorkspace`: Workspace containing precomputed `weights` and the derivative vector `deriv`.  
    The derivative will be written **in-place** to avoid additional memory allocations.  
    - `data::Vector{NumDiffFloat}`: Input signal or time series to differentiate.  
    - `prob::NumDiffProblem`: Problem definition containing the derivative order `order`, time step `dt`, 
    and signal length `n`.

    # Returns

    - `ws.deriv` updated **in-place** with the computed fractional derivative.
"""

function compute!(method::GL, ws::NumDiffWorkspace, data::Vector{NumDiffFloat}, prob::NumDiffProblem)
    n  = prob.n
    α  = prob.order
    dt = prob.dt
    weights = ws.weights
    d = deepcopy(data)
    inv_dt_pow = 1.0 / (dt^α)
    @inbounds for i in 1:n
        acc = 0.0
        for j in 1:i
            acc += weights[j] * d[i-j+1]
        end
        ws.deriv[i] = acc * inv_dt_pow
    end
end

function compute!(method::GLThreads, ws::NumDiffWorkspace, data::Vector{NumDiffFloat}, prob::NumDiffProblem)
    n  = prob.n
    α  = prob.order
    dt = prob.dt
    weights = ws.weights
    d = deepcopy(data)

    inv_dt_pow = 1.0 / (dt^prob.order)

    @threads :dynamic for i in 1:n
    acc = 0.0
        @simd for j in 1:i
            @inbounds acc += weights[j] * d[i-j+1]
        end
        @inbounds ws.deriv[i] = acc * inv_dt_pow
    end
end

"""
    compute!(method::GLShortMem, ws::NumDiffWorkspace, data::Vector{NumDiffFloat}, prob::NumDiffProblem)

    # Short-memory option

    Both serial and threaded versions support the short-memory approximation: if `prob._L < prob.n`,  
    the computation uses only the last `L` points of `data` for indices `i > L` (fixed memory).  
    For `i <= L`, the algorithm uses all available past points (`i` points, growing memory).  
    If `prob._L` is set to `typemax(Int)` or a value ≥ `prob.n`, the full memory algorithm is used.

    # Arguments

    - `method::GL` or `GLThreads`: Method descriptor.
    - `ws::NumDiffWorkspace`: Workspace containing precomputed `weights` and the derivative vector `deriv`.  
    The derivative will be written **in-place** to avoid additional memory allocations.  
    - `data::Vector{NumDiffFloat}`: Input signal or time series to differentiate.  
    - `prob::NumDiffProblem`: Problem definition containing the derivative order `order`, time step `dt`, 
    and signal length `n`.

    # Returns

    - `ws.deriv` updated **in-place** with the computed fractional derivative.

"""

function compute!(method::GLShortMem, ws::NumDiffWorkspace, data::Vector{NumDiffFloat}, prob::NumDiffProblem)
    n  = prob.n
    α  = prob.order
    dt = prob.dt
    weights = ws.weights
    d = deepcopy(data)
    optimal_L!(d,prob)
    L=prob._L

    inv_dt_pow = 1.0 / (dt^α)
    # ---------- Region 1: growing memory ----------
    grow_end = min(n, L)
    @inbounds for i in 1:grow_end
        acc = 0.0
        for j in 1:i
            acc += weights[j] * d[i-j+1]
        end
        ws.deriv[i] = acc * inv_dt_pow
    end
    if L < n
        @inbounds for i in (L+1):n
            acc = 0.0
                for j in 1:L
                    acc += weights[j] * d[i-j+1]
                end
                
                ws.deriv[i] = acc * inv_dt_pow
        end
    end
end

function compute!(method::GLShortMemThreads, ws::NumDiffWorkspace, data::Vector{NumDiffFloat}, prob::NumDiffProblem)
    n  = prob.n
    α  = prob.order
    dt = prob.dt
    weights = ws.weights
    d = deepcopy(data)
    optimal_L!(d,prob)
    L=prob._L

    inv_dt_pow = 1.0 / (dt^prob.order)

    # ---------- Region 1: growing memory ----------
    grow_end = min(n, L)
    @threads :dynamic for i in 1:grow_end
    acc = 0.0
        @simd for j in 1:i
            @inbounds acc += weights[j] * d[i-j+1]
        end
        @inbounds ws.deriv[i] = acc * inv_dt_pow
    end
    if L < n
        @threads :dynamic for i in (L+1):n
            acc = 0.0
                @simd for j in 1:L
                    @inbounds acc += weights[j] * d[i-j+1]
                end
                
                @inbounds ws.deriv[i] = acc * inv_dt_pow
        end
    end
end


"""
    compute!(method::GLShortMemCorr, ws::NumDiffWorkspace, data::Vector{NumDiffFloat}, prob::NumDiffProblem)

    Compute the **Grünwald-Letnikov fractional derivative** using the **short-memory approximation with correction** 
    and store the result directly in `ws.deriv`. This function has two variants depending on `method`:

    1. **Serial version** (`GLShortMemCorr`)  
    2. **Parallel version** (`GLShortMemCorrThreads`)  

    # Description

    The algorithm splits the computation into two regions:

    - **Region 1 (growing memory)**: For indices `i ≤ L`, the derivative is computed using all available past data points (`i` points).  
    - **Region 2 (fixed memory with correction)**: For indices `i > L`, only the last `L` points are used, and a correction term is applied to reduce truncation error.  

    # Arguments

    - `method::GLShortMemCorr` or `GLShortMemCorrThreads` : Method descriptor  
    - `ws::NumDiffWorkspace` : Workspace containing precomputed convolution weights (`ws.weights`) and preallocated derivative vector (`ws.deriv`).  
    - `data::Vector{NumDiffFloat}` : Input signal or time series to differentiate.  
    - `prob::NumDiffProblem` : Problem definition containing the derivative order `order`, time step `dt`, and signal length `n`.

    # Returns

    - `ws.deriv` updated **in-place** with the computed fractional derivative.
"""

function compute!(method::GLShortMemCorr, ws::NumDiffWorkspace, data::Vector{NumDiffFloat}, prob::NumDiffProblem)
    n  = prob.n
    α  = prob.order
    dt = prob.dt
    weights = ws.weights
    d = deepcopy(data)
    optimal_L!(d,prob)
    L=prob._L

    inv_dt_pow = 1.0 / (dt^α)
    one_minus_order = 1.0 - α
    # Precompute constants for the correction term
    k_0 = 1.0 / gamma(one_minus_order) 
    k_1 = -α / gamma(2.0 - α)

    term_L_alpha = L^(-α)
    term_L_1_alpha = L^(one_minus_order)

    offset_0 = k_0 * term_L_alpha
    offset_1 = k_1 * term_L_1_alpha
    # ---------- Region 1: growing memory ----------
    grow_end = min(n, L)
    @inbounds for i in 1:grow_end
        acc = 0.0
        for j in 1:i
            acc += weights[j] * d[i-j+1]
        end
        ws.deriv[i] = acc * inv_dt_pow
    end
    if L < n
        @inbounds for i in (L+1):n
            acc = 0.0
            for j in 1:L
                acc += weights[j] * d[i-j+1]
            end
            
            # Compute useful terms for the correction, i.e. m_0, m_1, y_0 and y_1
            term_i_alpha = i^(-α)
            m_0 = k_0 * term_i_alpha - offset_0
            
            term_i_1_alpha = i^(one_minus_order)
            m_1 = k_1 * term_i_1_alpha - offset_1

            @inbounds y_0 = d[i-L]

            # We consider the case when i = L+1 separately to avoid accessing d[0]
            if i != L + 1
                @inbounds y_1 = y_0 - d[i-L-1]
            else
                y_1 = 0.0
            end
            
            correction = y_0 * m_0 + y_1 * (L * m_0 - m_1)

            ws.deriv[i] = (acc + correction) * inv_dt_pow
        end
    end
end

function compute!(method::GLShortMemCorrThreads, ws::NumDiffWorkspace, data::Vector{NumDiffFloat}, prob::NumDiffProblem)
    n  = prob.n
    α  = prob.order
    dt = prob.dt
    weights = ws.weights
    d = deepcopy(data)
    optimal_L!(d,prob)
    L=prob._L

    inv_dt_pow = 1.0 / (dt^α)
    one_minus_order = 1.0 - α
    # Precompute constants for the correction term
    k_0 = 1.0 / gamma(one_minus_order) 
    k_1 = -α / gamma(2.0 - α)

    term_L_alpha = L^(-α)
    term_L_1_alpha = L^(one_minus_order)

    offset_0 = k_0 * term_L_alpha
    offset_1 = k_1 * term_L_1_alpha
    # ---------- Region 1: growing memory ----------
    grow_end = min(n, L)
    @threads :dynamic  for i in 1:grow_end
        acc = 0.0
        @simd for j in 1:i
            @inbounds acc += weights[j] * d[i-j+1]
        end
        ws.deriv[i] = acc * inv_dt_pow
    end
    if L < n
        @threads :dynamic  for i in (L+1):n
            acc = 0.0
            @simd for j in 1:L
                @inbounds acc += weights[j] * d[i-j+1]
            end
            
            # Compute useful terms for the correction, i.e. m_0, m_1, y_0 and y_1
            term_i_alpha = i^(-α)
            m_0 = k_0 * term_i_alpha - offset_0
            
            term_i_1_alpha = i^(one_minus_order)
            m_1 = k_1 * term_i_1_alpha - offset_1

            @inbounds y_0 = d[i-L]

            # We consider the case when i = L+1 separately to avoid accessing d[0]
            if i != L + 1
                @inbounds y_1 = y_0 - d[i-L-1]
            else
                y_1 = 0.0
            end
            
            correction = y_0 * m_0 + y_1 * (L * m_0 - m_1)
            
            ws.deriv[i] = (acc + correction) * inv_dt_pow
        end
    end
end

"""
    compute!(method::GLFFT, ws::NumDiffWorkspace, data::Vector{NumDiffFloat}, prob::NumDiffProblem)

    Compute variant using Fourier Transform, more suited for serial, cannot run in parallel.

    # Arguments

    - `method::GLFFt`: Method descriptor.
    - `ws::NumDiffWorkspace`: Workspace containing precomputed `weights` and the derivative vector `deriv`.  
    The derivative will be written **in-place** to avoid additional memory allocations.  
    - `data::Vector{NumDiffFloat}`: Input signal or time series to differentiate.  
    - `prob::NumDiffProblem`: Problem definition containing the derivative order `order`, time step `dt`, 
    and signal length `n`.

    # Returns

    - `ws.deriv` updated **in-place** with the computed fractional derivative.
"""

function compute!(method::GLFFT, ws::NumDiffWorkspace, data::Vector{NumDiffFloat}, prob::NumDiffProblem)
    d = deepcopy(data)
    deriv = fftfilt(ws.weights, d)
    
    inv_dt_pow = 1.0 / (prob.dt^prob.order)
    for i in eachindex(ws.deriv)
        ws.deriv[i] = deriv[i] * inv_dt_pow
    end
    
    
end