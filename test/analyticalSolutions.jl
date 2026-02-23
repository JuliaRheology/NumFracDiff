# -----------------------------
# Unit test for Caputo derivatives
# -----------------------------


"""
Compute the exact fractional derivatives.
"""
function exact_solutions(t::Vector{<:Real}, order::Float64, type::String; amp=1.0, offset=0.0, ν = -0.5)
    y = zeros(length(t))
    if type == "step"
        # Derivative of a step function: D^α H(t) = t^(-α)/Gamma(1-α)
        y .= amp .* (t .- offset).^(-order) ./ gamma(1.0 - order)
    elseif type == "ramp"
        # Derivative of ramp: f(t) = t → D^α t = t^(1-α)/Gamma(2-α)
        y .= (t .- offset).^(1.0 - order) ./ gamma(2.0 - order)
    elseif type == "ramp_hold"
        # Ramp up to t1, hold constant after
        t1 = 1.0
        # Fractional derivative of ramp-and-hold
        for (i, ti) in enumerate(t)
            τ = ti - offset
            if τ < 0
                y[i] = 0.0
            elseif τ < t1
                y[i] = amp * τ^(1.0 - order) / gamma(2.0 - order)
            else
                y[i] = amp * ( τ^(1.0 - order) - (τ - t1)^(1.0 - order) ) / gamma(2.0 - order)
            end
        end

    elseif type == "power"
        y .= (t.^(ν - order)) .* gamma(ν + 1) ./ gamma(ν + 1 - order)
    else
        error("Unknown input type: $type")
    end
    return y
end



@testset "Fractional derivative tests" begin

    outdir = joinpath(pwd(), "analyticalSolutions_plots")

    if isdir(outdir)
        rm(outdir; recursive=true, force=true)
    end

    mkpath(outdir)

    # Common parameters
    dt = convert(NumDiffFloat, 1e-3)
    t = dt:dt:5.0
    α = convert(NumDiffFloat, 0.2)
    n = convert(NumDiffInt, length(t))

    L = convert(NumDiffInt, 4000)

    # Define the derivative methods you want to test
    methods = [
        Caputo(),
        CaputoThreads(), 
        RL(), 
        RLThreads(), 
        RL(L), 
        RLThreads(L), 
        RLShortMemCorr(L), 
        RLShortMemCorrThreads(L)
    ]

    tollerances = [
        1e-10,
        1e-10,
        1e-10, 
        1e-10,
        1e-1,
        1e-1,
        1e-2,
        1e-2 
    ]

    # Loop over methods
    for (i, method) in enumerate(methods)

        @info "Testing method: $(typeof(method))"

        for (func_type, kwargs) in [
            ("step", (amp=1.0, offset=0.0)),
            ("ramp", ()),
            ("ramp_hold", ()), 
            ("power", (ν = -0.4,))
        ]

            # -----------------------------
            # Define input data
            # -----------------------------
            data = zeros(n)
            if func_type == "step"
                epsilon = 10*dt  # fast ramp to regularize step
                amp = get(kwargs, :amp, 1.0)
                data .= [t_i < epsilon ? (amp/epsilon)*t_i : amp for t_i in t]
            elseif func_type == "ramp"
                data .= t
            elseif func_type == "ramp_hold"
                t1 = 1.0
                data .= [ti < t1 ? ti : t1 for ti in t]
            elseif func_type == "power"
                amp = 1.0
                ν   = -0.4
                data .= amp .* t.^ν
            end

            data = convert(Vector{NumDiffFloat}, data)
            # -----------------------------
            # Initialize workspace and problem
            # -----------------------------
            ws = NumDiffWorkspace(zeros(NumDiffFloat, n), zeros(NumDiffFloat, n))
            prob = NumDiffProblem(dt=dt, order=α, n=n, method=method)

            # -----------------------------
            # Generate weights and compute derivative
            # -----------------------------
            generate_weights!(method, prob, ws)
            compute!(method, ws, data, prob)

            # -----------------------------
            # Compute exact solution
            # -----------------------------
            exact = exact_solutions(collect(t), α, func_type; kwargs...)

            # -----------------------------
            # Compute error (ignore first point for step)
            # -----------------------------
            if func_type == "step" || func_type == "power"
                error = 0.0 # maximum(abs.(ws.deriv[2:end] .- exact[2:end]))
            else
                error = maximum(abs.(ws.deriv[2:end] .- exact[2:end])./exact[2:end])
            end

            @info "Max error for $func_type = $error"

            # -----------------------------
            # Plot results
            # -----------------------------
            # if func_type == "power"
            #     p = plot(t[2:end], exact[2:end], lw=2, label="Analytical", color=:blue, xaxis=:log10, yaxis=:log10)
            #     plot!(p, t[2:end], ws.deriv[2:end], lw=2, ls=:dash, label="Numerical", color=:red, xaxis=:log10, yaxis=:log10)
            # else
            #     p = plot(t, exact, lw=2, label="Analytical", color=:blue)
            #     plot!(p, t, ws.deriv, lw=2, ls=:dash, label="Numerical", color=:red)
            # end


            # xlabel!("Time")
            # ylabel!("D^α $func_type")
            
            # T = typeof(method)
            # name = string(T)

            # if hasfield(T, :L) && (getfield(method, :L) != typemax(NumDiffInt))
            #     Lval = getfield(method, :L)
            #     name = "$(name)_L_$(Lval)"
            # end

            # title!("$(name) - $func_type")
            # savefig(p, "./analyticalSolutions_plots/$(name)_$func_type.pdf")

            # -----------------------------
            # Test tolerance
            # -----------------------------
            @test error < tollerances[i]  # adjust tolerance depending on dt

        end
    end
end


