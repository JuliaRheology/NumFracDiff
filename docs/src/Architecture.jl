# # Architecture

# Here is an explanation on the data structers usen by NumFracDiff.
# The library needs a set of data on how to perform the derivative of an input vector of size wrt of time.

# ## DiffMethod
# The library offers 3 main types of derivative methods: Grünwald-Letnikov, Caputo and Riemann-Liovuille.
# Method choice is handled by defining all the types as sub-structs of an abstract struct DiffMethod, so we can dispatch the compute functions based on DiffMethod typing.
# A more in-depth section for the methods is reported later in the Documentation
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

# ## Problem
# Mutable data structure representing the defition of the numerical differentiation problem we want to solve.
# The structure contains the parameters needed:
# -dt::NumDiffFloat: Sampling interval (Must be constant!)
# -order::NumDiffFloat: Derivative order. Can be negative as well for fraction integrals
# -n::NumDiffInt: Expected length of the input data to derive.
# -method::DiffMethod:  Method for differentiation
# -L::NumDiffInt: parameter set automatically, usually same as n or shorter for "Short Memory" methods
mutable struct NumDiffProblem{NumDiffFloat}
    dt::NumDiffFloat
    order::NumDiffFloat
    n::NumDiffInt
    method::DiffMethod
    _L::NumDiffInt

    # Keyword constructor
    function NumDiffProblem(; dt::NumDiffFloat, order::NumDiffFloat, n::NumDiffInt, method::DiffMethod)
        new{NumDiffFloat}(dt, order, n, method, n)
    end

end


# ## Workspace
# Data structures containing data that doesn't require the struct to be mutable for better performance.
# Weights vector is needed when differentiating, while deriv vector contains the output of the library.
struct NumDiffWorkspace{NumDiffFloat}
    weights::Vector{NumDiffFloat}
    deriv::Vector{NumDiffFloat}
end





