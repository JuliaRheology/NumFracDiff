
@testset "optimal_L tests" begin

    dt = convert(NumDiffFloat, 1e-3)
    α  = convert(NumDiffFloat, 0.5)
    n = 1000

    # Sample signals
    ramp = convert(Vector{NumDiffFloat}, 1e-2 .* collect(0:dt:(n-1)*dt))
    constant  = ones(NumDiffFloat, n)
    zerosig   = zeros(NumDiffFloat, n)
    bigsignal = 10 .* ramp    

    sm_methods = [
        GLShortMem(),
        GLShortMemThreads(),
        GLShortMemCorr(),
        GLShortMemCorrThreads(),
        RLShortMem(),
        RLShortMemThreads(),
        RLShortMemCorr(),
        RLShortMemCorrThreads()
    ]

     # -----------------------------
    # RL methods should set L
    # -----------------------------
    for method in sm_methods
        prob = NumDiffProblem(dt=dt, order=α, n=n, method=method)

        optimal_L!(ramp, prob; tol=1e-2)

        @test hasfield(typeof(prob), :_L)
        @test prob._L isa NumDiffInt
        @test 1 ≤ prob._L ≤ n
    end

    # -----------------------------
    # Larger amplitude → larger L
    # -----------------------------
    prob1 = NumDiffProblem(dt=dt, order=α, n=n, method=RLShortMem())
    optimal_L!(ramp, prob1; tol=1e-2)
    L1 = prob1._L

    prob2 = NumDiffProblem(dt=dt, order=α, n=n, method=RLShortMem())
    optimal_L!(bigsignal, prob2; tol=1e-2)
    L2 = prob2._L

    @test L2 > L1

    # -----------------------------
    # Tighter tolerance → larger L
    # -----------------------------
    prob_loose = NumDiffProblem(dt=dt, order=α, n=n, method=RLShortMem())
    optimal_L!(ramp, prob_loose; tol=1e-1)
    L_loose = prob_loose._L

    prob_tight = NumDiffProblem(dt=dt, order=α, n=n, method=RLShortMem())
    optimal_L!(ramp, prob_tight; tol=1e-3)
    L_tight = prob_tight._L

    @test L_tight > L_loose


end



function test_update_order_methods()
    dt = 1e-3
    n  = 500
    α  = 0.5
    new_order = 0.8

    # List of methods to test
    methods = [
        GL(),
        GLThreads(),
        GLShortMem(),
        GLShortMemThreads(),
        GLShortMemCorr(),
        GLShortMemCorrThreads(),
        Caputo(),
        CaputoThreads(),
        RL(),
        RLThreads(),
        RLShortMem(),
        RLShortMemThreads(),
        RLShortMemCorr(),
        RLShortMemCorrThreads()
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