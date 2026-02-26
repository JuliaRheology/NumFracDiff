
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



function test_update_order_methods()
    dt = 1e-3
    n  = 500
    α  = 0.5
    new_order = 0.8

    # List of methods to test
    methods = [
        Caputo(),
        CaputoThreads(),
        RL(),
        RLThreads(),
        RLShortMemCorr(1000),
        RLShortMemCorrThreads(1000)
    ]

    for method in methods
        @testset "Testing update_order! with $(typeof(method))" begin
            prob = NumDiffProblem(dt=dt, order=α, n=n, method=method)
            ws = NumDiffWorkspace(zeros(n), zeros(n))

            # Save original weights for comparison
            original_weights = copy(ws.weights)

            # Update order
            update_order!(prob, ws, new_order)

            # Test that order is updated
            @test prob.order == new_order

            # Test that weights are regenerated
            @test !all(ws.weights .== original_weights)

            # Test that assertion triggers if same order is passed again
            err = try
                update_order!(prob, ws, new_order)
                false
            catch e
                e isa AssertionError
            end
            @test err == true
        end
    end
end

# Run the test
@testset "update_order! tests" begin
    test_update_order_methods()
end