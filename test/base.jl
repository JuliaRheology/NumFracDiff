
@testset "optimal_L tests" begin

    dt = convert(NumDiffFloat, 1e-3)
    α  = convert(NumDiffFloat, 0.5)
    n = 1000

    # Sample signals
    ramp = convert(Vector{NumDiffFloat}, 1e-2 .* collect(0:dt:(n-1)*dt))
    constant  = ones(NumDiffFloat, n)
    zerosig   = zeros(NumDiffFloat, n)
    bigsignal = 10 .* ramp

    L = convert(NumDiffInt, 100)

    rl_methods = [
        RL(L),
        RLThreads(L),
        RLShortMemCorr(L),
        RLShortMemCorrThreads(L)
    ]

    non_rl_methods = [
        Caputo(),
        CaputoThreads()
    ]

     # -----------------------------
    # RL methods should set L
    # -----------------------------
    for method in rl_methods
        prob = NumDiffProblem(dt=dt, order=α, n=n, method=method)

        optimal_L!(ramp, prob; tol=1e-2)

        @test hasfield(typeof(prob.method), :L)
        @test prob.method.L isa NumDiffInt
        @test 1 ≤ prob.method.L ≤ n
    end

    # -----------------------------
    # Non-RL methods should not set L
    # -----------------------------
    for method in non_rl_methods
        prob = NumDiffProblem(dt=dt, order=α, n=n, method=method)

        optimal_L!(ramp, prob; tol=1e-2)

        # Since optimal_L! returns early with a warning, L should not exist or stay undefined
        @test !hasfield(typeof(prob.method), :L) || prob.method.L === nothing
    end

    # -----------------------------
    # Larger amplitude → larger L
    # -----------------------------
    prob1 = NumDiffProblem(dt=dt, order=α, n=n, method=RL())
    optimal_L!(ramp, prob1; tol=1e-2)
    L1 = prob1.method.L

    prob2 = NumDiffProblem(dt=dt, order=α, n=n, method=RL())
    optimal_L!(bigsignal, prob2; tol=1e-2)
    L2 = prob2.method.L

    @test L2 > L1

    # -----------------------------
    # Tighter tolerance → larger L
    # -----------------------------
    prob_loose = NumDiffProblem(dt=dt, order=α, n=n, method=RL())
    optimal_L!(ramp, prob_loose; tol=1e-1)
    L_loose = prob_loose.method.L

    prob_tight = NumDiffProblem(dt=dt, order=α, n=n, method=RL())
    optimal_L!(ramp, prob_tight; tol=1e-3)
    L_tight = prob_tight.method.L

    @test L_tight > L_loose


end