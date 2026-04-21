
"""
    generate_weights!(::GL, order::NumDiffFloat, L::Int, ws::NumDiffWorkspace)

    Compute the convolution weights for the Grunwald-Letnikov fractional derivative
    and store them **in-place** in the workspace `ws`.

    # Arguments

    - `::GL` : Dispatch type indicating the Caputo method.

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

function compute!(method::GL, ws::NumDiffWorkspace, data::Vector{NumDiffFloat}, prob::NumDiffProblem)
    n  = prob.n
    α  = prob.order
    dt = prob.dt
    weights = ws.weights

    inv_dt_pow = 1.0 / (dt^α)
    @inbounds for i in 1:n
        acc = 0.0
        for j in 1:i
            acc += weights[j] * data[i-j+1]
        end
        ws.deriv[i] = acc * inv_dt_pow
    end
end

function compute!(method::GLThreads, ws::NumDiffWorkspace, data::Vector{NumDiffFloat}, prob::NumDiffProblem)
    n  = prob.n
    α  = prob.order
    dt = prob.dt
    weights = ws.weights

    inv_dt_pow = 1.0 / (dt^prob.order)

    @threads :dynamic for i in 1:n
    acc = 0.0
        @simd for j in 1:i
            @inbounds acc += weights[j] * data[i-j+1]
        end
        @inbounds ws.deriv[i] = acc * inv_dt_pow
    end
end


function compute!(method::GLShortMem, ws::NumDiffWorkspace, data::Vector{NumDiffFloat}, prob::NumDiffProblem)
    n  = prob.n
    α  = prob.order
    dt = prob.dt
    weights = ws.weights
    optimal_L!(data,prob)
    L=prob._L

    inv_dt_pow = 1.0 / (dt^α)
    # ---------- Region 1: growing memory ----------
    grow_end = min(n, L)
    @inbounds for i in 1:grow_end
        acc = 0.0
        for j in 1:i
            acc += weights[j] * data[i-j+1]
        end
        ws.deriv[i] = acc * inv_dt_pow
    end
    if L < n
        @inbounds for i in (L+1):n
            acc = 0.0
                for j in 1:L
                    acc += weights[j] * data[i-j+1]
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
    optimal_L!(data,prob)
    L=prob._L

    inv_dt_pow = 1.0 / (dt^prob.order)

    # ---------- Region 1: growing memory ----------
    grow_end = min(n, L)
    @threads :dynamic for i in 1:grow_end
    acc = 0.0
        @simd for j in 1:i
            @inbounds acc += weights[j] * data[i-j+1]
        end
        @inbounds ws.deriv[i] = acc * inv_dt_pow
    end
    if L < n
        @threads :dynamic for i in (L+1):n
            acc = 0.0
                @simd for j in 1:L
                    @inbounds acc += weights[j] * data[i-j+1]
                end
                
                @inbounds ws.deriv[i] = acc * inv_dt_pow
        end
    end
end

function compute!(method::GLShortMemCorr, ws::NumDiffWorkspace, data::Vector{NumDiffFloat}, prob::NumDiffProblem)
    n  = prob.n
    α  = prob.order
    dt = prob.dt
    weights = ws.weights
    optimal_L!(data,prob)
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
            acc += weights[j] * data[i-j+1]
        end
        ws.deriv[i] = acc * inv_dt_pow
    end
    if L < n
        @inbounds for i in (L+1):n
            acc = 0.0
            for j in 1:L
                acc += weights[j] * data[i-j+1]
            end
            
            # Compute useful terms for the correction, i.e. m_0, m_1, y_0 and y_1
            term_i_alpha = i^(-α)
            m_0 = k_0 * term_i_alpha - offset_0
            
            term_i_1_alpha = i^(one_minus_order)
            m_1 = k_1 * term_i_1_alpha - offset_1

            @inbounds y_0 = data[i-L]

            # We consider the case when i = L+1 separately to avoid accessing data[0]
            if i != L + 1
                @inbounds y_1 = y_0 - data[i-L-1]
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
    optimal_L!(data,prob)
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
            @inbounds acc += weights[j] * data[i-j+1]
        end
        ws.deriv[i] = acc * inv_dt_pow
    end
    if L < n
        @threads :dynamic  for i in (L+1):n
            acc = 0.0
            @simd for j in 1:L
                @inbounds acc += weights[j] * data[i-j+1]
            end
            
            # Compute useful terms for the correction, i.e. m_0, m_1, y_0 and y_1
            term_i_alpha = i^(-α)
            m_0 = k_0 * term_i_alpha - offset_0
            
            term_i_1_alpha = i^(one_minus_order)
            m_1 = k_1 * term_i_1_alpha - offset_1

            @inbounds y_0 = data[i-L]

            # We consider the case when i = L+1 separately to avoid accessing data[0]
            if i != L + 1
                @inbounds y_1 = y_0 - data[i-L-1]
            else
                y_1 = 0.0
            end
            
            correction = y_0 * m_0 + y_1 * (L * m_0 - m_1)
            
            ws.deriv[i] = (acc + correction) * inv_dt_pow
        end
    end
end


function compute!(method::GLFFT, ws::NumDiffWorkspace, data::Vector{NumDiffFloat}, prob::NumDiffProblem)
    deriv = fftfilt(ws.weights, data)
    
    inv_dt_pow = 1.0 / (prob.dt^prob.order)
    for i in eachindex(ws.deriv)
        ws.deriv[i] = deriv[i] * inv_dt_pow
    end
    
    
end