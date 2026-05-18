"""
    DiffMethod

Abstract supertype for specifying numerical differentiation methods 
(e.g., Grünwald-Letnikov, Caputo, Riemann-Liouville).
"""
abstract type DiffMethod end

struct Caputo <: DiffMethod end
struct CaputoThreads <: DiffMethod end


struct RL <: DiffMethod end
struct RLThreads <: DiffMethod end
struct RLShortMem <: DiffMethod end
struct RLShortMemThreads <: DiffMethod end
struct RLShortMemCorr <: DiffMethod end
struct RLShortMemCorrThreads <: DiffMethod end


struct GL <: DiffMethod end
struct GLThreads <: DiffMethod end
struct GLShortMem <: DiffMethod end
struct GLShortMemThreads <: DiffMethod end
struct GLShortMemCorr <: DiffMethod  end
struct GLShortMemCorrThreads <: DiffMethod end
struct GLFFT <: DiffMethod end


"""
    NumDiffProblem{NumDiffFloat}

    Definition of a numerical differentiation problem.

    This structure stores the parameters that define the derivative
    to be computed, independent of the input data. It can be reused
    with multiple datasets without reallocation.

    `NumDiffFloat` and `NumDiffInt` denote the numeric types used internally
    by NumFracDiff.jl (typically `Float64` and `Int`).

    # Fields

    - `dt::NumDiffFloat`: Sampling interval (time step).

    - `order::NumDiffFloat`: Derivative order. May be integer or fractional.

    - `n::NumDiffInt`: Length of the data vector to be processed.

    - `method::Symbol`: Numerical differentiation method. Currently supported methods include:  `:caputo`, `:rl`, `:gl`.
"""

mutable struct NumDiffProblem{NumDiffFloat}
    dt::NumDiffFloat
    order::NumDiffFloat
    n::NumDiffInt             # Length of the vector to be differentiated
    method::DiffMethod
    _L::NumDiffInt

    # Keyword constructor
    function NumDiffProblem(; dt::NumDiffFloat, order::NumDiffFloat, n::NumDiffInt, method::DiffMethod)
        new{NumDiffFloat}(dt, order, n, method, n)
    end

end

"""
    NumDiffWorkspace{NumDiffFloat}

Workspace container for numerical differentiation routines.

This structure stores precomputed weights and preallocated buffers
used during derivative evaluation. Reusing a workspace avoids
memory allocations and improves performance when processing
multiple signals.

`NumDiffFloat` denotes the numeric type used internally by NumFracDiff.jl.

# Fields

- `weights::Vector{NumDiffFloat}`: Precomputed convolution weights used by the differentiation algorithm.

- `deriv::Vector{NumDiffFloat}`: Preallocated vector where the computed derivative is stored.
"""

struct NumDiffWorkspace{NumDiffFloat}
    weights::Vector{NumDiffFloat}
    deriv::Vector{NumDiffFloat}
end


