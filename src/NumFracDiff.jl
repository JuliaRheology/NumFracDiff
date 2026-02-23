module NumFracDiff

    using SpecialFunctions: gamma
    using FFTW
    using DSP
    using Base.Threads

    ######################################################

    # This defines the data type for all arrays, parameters and processing
    # it is defined as a const to avoid performance penalties.
    # See julia docs for more info on this.
    const NumDiffFloat = Float64
    const NumDiffInt = Int64

    ######################################################

    export generate_weights!, compute!

    # workspace.jl
    export NumDiffProblem, NumDiffWorkspace, DiffMethod
    export Caputo, CaputoThreads
    export GL, GLThreads, GLShortMem, GLShortMemThreads, GLShortMemCorr, GLShortMemCorrThreads
    export RL, RLThreads, RLShortMemCorr, RLShortMemCorrThreads

    # base.jl
    export init_workspace, optimal_L!

    export NumDiffFloat, NumDiffInt

    ######################################################

    include("workspace.jl")
    include("base.jl")
    include("Caputo.jl")
    include("RiemannLiouville.jl")
    include("GrunLetn.jl")

end
