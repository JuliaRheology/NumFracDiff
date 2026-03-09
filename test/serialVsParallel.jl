
using Random
using LinearAlgebra
using BenchmarkTools

# -----------------------------
# Setup
# -----------------------------
const N = 1000
const dt = 0.01
const α = 0.5
Random.seed!(1234)

# Generate test data
data = rand(N)

# -----------------------------
# Helper: maximum absolute difference
# -----------------------------
absdiff(a, b) = maximum(abs.(a .- b))

# -----------------------------
# Test GL derivative
# -----------------------------
@testset "GL derivative" begin
    L_full = typemax(Int)  # full memory
    prob_gl = NumDiffProblem(dt=dt, order=α, n=N, method=GL(L_full))
    ws_gl   = init_workspace(prob_gl)

    prob_rl_threads = NumDiffProblem(dt=dt, order=α, n=N, method=GLThreads(L_full))
    ws_gl_threads   = init_workspace(prob_rl_threads)

    compute!(prob_gl.method, ws_gl, data, prob_gl)
    compute!(prob_rl_threads.method, ws_gl_threads, data, prob_rl_threads)

    method_rl = prob_gl.method
    method_rl_threads = prob_rl_threads.method

    println("Benchmarking GL serial vs parallel...")
    @btime compute!($method_rl, $ws_gl, $data, $prob_gl)
    @btime compute!($method_rl_threads, $ws_gl_threads, $data, $prob_rl_threads)

    @test absdiff(ws_gl.deriv, ws_gl_threads.deriv) < 1e-12
end

# -----------------------------
# Test GL short-memory
# -----------------------------
@testset "GL Short-Memory" begin
    L_short = 50
    prob_gl_short = NumDiffProblem(dt=dt, order=α, n=N, method=GL(L_short))
    ws_gl_short   = init_workspace(prob_gl_short)

    prob_gl_short_threads = NumDiffProblem(dt=dt, order=α, n=N, method=GLThreads(L_short))
    ws_gl_short_threads   = init_workspace(prob_gl_short_threads)

    compute!(prob_gl_short.method, ws_gl_short, data, prob_gl_short)
    compute!(prob_gl_short_threads.method, ws_gl_short_threads, data, prob_gl_short_threads)

    method_gl_short = prob_gl_short.method
    method_gl_short_threads = prob_gl_short_threads.method

    println("Benchmarking GL short-memory serial vs parallel...")
    @btime compute!($method_gl_short, $ws_gl_short, $data, $prob_gl_short)
    @btime compute!($method_gl_short_threads, $ws_gl_short_threads, $data, $prob_gl_short_threads)

    @test absdiff(ws_gl_short.deriv, ws_gl_short_threads.deriv) < 1e-12
end

# -----------------------------
# Test GL Short-Memory with Correction
# -----------------------------
@testset "GL Short-Memory with Correction" begin
    L_corr = 50  # short-memory length
    prob_gl_corr = NumDiffProblem(dt=dt, order=α, n=N, method=GLShortMemCorr(L_corr))
    ws_gl_corr   = init_workspace(prob_gl_corr)

    prob_gl_corr_threads = NumDiffProblem(dt=dt, order=α, n=N, method=GLShortMemCorrThreads(L_corr))
    ws_gl_corr_threads   = init_workspace(prob_gl_corr_threads)

    # Compute serial and parallel derivatives
    compute!(prob_gl_corr.method, ws_gl_corr, data, prob_gl_corr)
    compute!(prob_gl_corr_threads.method, ws_gl_corr_threads, data, prob_gl_corr_threads)

    method_gl_corr = prob_gl_corr.method
    method_gl_corr_threads = prob_gl_corr_threads.method

    println("Benchmarking GL short-memory with correction serial vs parallel...")
    @btime compute!($method_gl_corr, $ws_gl_corr, $data, $prob_gl_corr)
    @btime compute!($method_gl_corr_threads, $ws_gl_corr_threads, $data, $prob_gl_corr_threads)

    # Validate that serial and parallel give the same result
    @test absdiff(ws_gl_corr.deriv, ws_gl_corr_threads.deriv) < 1e-12
end

# -----------------------------
# Test Caputo
# -----------------------------
@testset "Caputo derivative" begin
    prob_serial = NumDiffProblem(dt=dt, order=α, n=N, method=Caputo())
    ws_serial   = init_workspace(prob_serial)

    prob_parallel = NumDiffProblem(dt=dt, order=α, n=N, method=CaputoThreads())
    ws_parallel   = init_workspace(prob_parallel)

    compute!(prob_serial.method, ws_serial, data, prob_serial)
    compute!(prob_parallel.method, ws_parallel, data, prob_parallel)

    method_serial = prob_serial.method
    method_parallel = prob_parallel.method

    println("Benchmarking Caputo serial vs parallel...")
    @btime compute!($method_serial, $ws_serial, $data, $prob_serial)
    @btime compute!($method_parallel, $ws_parallel, $data, $prob_parallel)

    @test absdiff(ws_serial.deriv, ws_parallel.deriv) < 1e-12
end

# -----------------------------
# Test RL derivative
# -----------------------------
@testset "RL derivative" begin
    L_full = typemax(Int)  # full memory
    prob_rl = NumDiffProblem(dt=dt, order=α, n=N, method=RL(L_full))
    ws_rl   = init_workspace(prob_rl)

    prob_rl_threads = NumDiffProblem(dt=dt, order=α, n=N, method=RLThreads(L_full))
    ws_rl_threads   = init_workspace(prob_rl_threads)

    compute!(prob_rl.method, ws_rl, data, prob_rl)
    compute!(prob_rl_threads.method, ws_rl_threads, data, prob_rl_threads)

    method_rl = prob_rl.method
    method_rl_threads = prob_rl_threads.method

    println("Benchmarking RL serial vs parallel...")
    @btime compute!($method_rl, $ws_rl, $data, $prob_rl)
    @btime compute!($method_rl_threads, $ws_rl_threads, $data, $prob_rl_threads)

    @test absdiff(ws_rl.deriv, ws_rl_threads.deriv) < 1e-12
end

# -----------------------------
# Test RL short-memory
# -----------------------------
@testset "RL Short-Memory" begin
    L_short = 50
    prob_rl_short = NumDiffProblem(dt=dt, order=α, n=N, method=RL(L_short))
    ws_rl_short   = init_workspace(prob_rl_short)

    prob_rl_short_threads = NumDiffProblem(dt=dt, order=α, n=N, method=RLThreads(L_short))
    ws_rl_short_threads   = init_workspace(prob_rl_short_threads)

    compute!(prob_rl_short.method, ws_rl_short, data, prob_rl_short)
    compute!(prob_rl_short_threads.method, ws_rl_short_threads, data, prob_rl_short_threads)

    method_rl_short = prob_rl_short.method
    method_rl_short_threads = prob_rl_short_threads.method

    println("Benchmarking RL short-memory serial vs parallel...")
    @btime compute!($method_rl_short, $ws_rl_short, $data, $prob_rl_short)
    @btime compute!($method_rl_short_threads, $ws_rl_short_threads, $data, $prob_rl_short_threads)

    @test absdiff(ws_rl_short.deriv, ws_rl_short_threads.deriv) < 1e-12
end

# -----------------------------
# Test RL Short-Memory with Correction
# -----------------------------
@testset "RL Short-Memory with Correction" begin
    L_corr = 50  # short-memory length
    prob_rl_corr = NumDiffProblem(dt=dt, order=α, n=N, method=RLShortMemCorr(L_corr))
    ws_rl_corr   = init_workspace(prob_rl_corr)

    prob_rl_corr_threads = NumDiffProblem(dt=dt, order=α, n=N, method=RLShortMemCorrThreads(L_corr))
    ws_rl_corr_threads   = init_workspace(prob_rl_corr_threads)

    # Compute serial and parallel derivatives
    compute!(prob_rl_corr.method, ws_rl_corr, data, prob_rl_corr)
    compute!(prob_rl_corr_threads.method, ws_rl_corr_threads, data, prob_rl_corr_threads)

    method_rl_corr = prob_rl_corr.method
    method_rl_corr_threads = prob_rl_corr_threads.method

    println("Benchmarking RL short-memory with correction serial vs parallel...")
    @btime compute!($method_rl_corr, $ws_rl_corr, $data, $prob_rl_corr)
    @btime compute!($method_rl_corr_threads, $ws_rl_corr_threads, $data, $prob_rl_corr_threads)

    # Validate that serial and parallel give the same result
    @test absdiff(ws_rl_corr.deriv, ws_rl_corr_threads.deriv) < 1e-12
end




println("All tests passed.")