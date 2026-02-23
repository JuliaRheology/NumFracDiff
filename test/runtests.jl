using NumFracDiff
using Test
using SpecialFunctions: gamma
using Plots

@testset "NumFracDiff.jl" begin

    print("----------------------------------------\n")
    print("              Test base.jl              \n")
    print("----------------------------------------\n")
    include("base.jl")

    print("----------------------------------------\n")
    print("      Test Analytical solutions         \n")
    print("----------------------------------------\n")
    include("analyticalSolutions.jl")

    print("----------------------------------------\n")
    print("      Test Serial vs parallel           \n")
    print("----------------------------------------\n")
    include("serialVsParallel.jl")

end

