
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

function generate_weights!(::Union{GL}, prob::NumDiffProblem, ws::NumDiffWorkspace)
    
    ws.weights[1] = 1.0

    alpha_plus_one = order + 1.0
    
    for k in 2:prob.n
        ws.weights[k] = ws.weights[k-1] * (1.0 - alpha_plus_one/(k - 1.0)) 
    end

end
