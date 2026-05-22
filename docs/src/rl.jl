# # Riemann-Liovuille

using NumFracDiff

# ### Mathematical Definition

# The Riemann-Liovuille fractional derivative of order $α$ of a function $f(t)$ is defined as the limit of the sum:

# $$D^{\alpha} f(t) = \frac{1}{\Gamma(1-\alpha)} \frac{d}{dt} \int_{t_0}^{t} \frac{f(\tau)}{(t-\tau)^\alpha} d\tau, \quad t \ge t_0.$$

dt = 0.01
n_points = 1000
t = [i * dt for i in 0:(n_points-1)]
data = t .^ 2
prob = NumDiffProblem(dt = dt, order = 0.5, n = n_points, method = RL())
ws = init_workspace(prob)
generate_weights!(prob.method, prob, ws, L=1000)

# ### Numerical Implementation

# Given a sampling time $\Delta t$, the derivative at index $i$ is calculated leveraging the precomputed structural weights:

# $$D^\alpha f(t_i) \approx \frac{\Delta t^{-\alpha}}{\Gamma(2-\alpha)} \sum_{j=1}^{i} w_{j-1}^{(\alpha)} f(t_{i-j+1})$$

# In this formulation, the weights $w_k^{(\alpha)}$ are generated via the difference of consecutive powers:
# - $w_0^{(\alpha)} = 1$
# - $w_k^{(\alpha)} = (k+1)^{1-\alpha} - 2k^{1-\alpha} + (k-1)^{1-\alpha} \quad \text{for } k \ge 1.$

# These coefficients are computed `in-place` using the function [`generate_weights!`](@ref) and represent the structural memory of the operator.

# ## RL

# The standard Riemann-Liouville method computes the fractional derivative by carrying out a full-history convolution. While providing maximum precision, it features an asymptotic computational complexity of $\mathcal{O}(n^2)$, where $n$ is the total number of data points. The library provides two dispatch types for this full-memory approach:
# - `RL()`: The standard, single-threaded sequential implementation.
prob = NumDiffProblem(dt = dt, order = 0.5, n = n_points, method = RL())
ws = init_workspace(prob)
compute!(prob.method, ws, data, prob)
# - `RLThreads()`: A multi-threaded implementation that parallelizes the outer loop using Julia's native task migration (`@threads :dynamic`) and leverages multi-core processors via `@simd` vectorization.
prob = NumDiffProblem(dt = dt, order = 0.5, n = n_points, method = RLThreads())
ws = init_workspace(prob)
compute!(prob.method, ws, data, prob)

# ## RL Short memory

# To avoid the quadratic growth in computational cost ($\mathcal{O}(n^2)$) on long time-series, NumFracDiff implements the Short Memory Principle for the Riemann-Liouville operator. By invoking [`optimal_L!`](@ref), the calculation drops old historical data and focuses exclusively on a fixed window of the recent past of length $L$. This effectively shifts the asymptotic complexity down to $\mathcal{O}(n \cdot L)$.

# When short memory is enabled, the execution automatically transitions across two operational zones:
# 1. `Growing Memory Region` ($i \le L$): The full available history is evaluated because the current index is smaller than the window length $L$.
# 2. `Short Memory Region` ($i > L$): Only the most recent $L$ points are used in the convolution sum, significantly speeding up calculation.

# ### Error Correction

# Dropping the remote history causes a deterministic truncation error that degrades precision. NumFracDiff solves this via Memory Correction variants (`RLShortMemCorr` and `RLShortMemCorrThreads`). 

# These methods approximate the discarded memory by treating the remote history under a first-order local trend, adding a dynamic correction term inside the loop. This restores high accuracy while preserving the computational footprint of the short-memory approach.

# The library provides four distinct dispatch types for short-memory RL differentiation:
# - `RLShortMem()`: Standard short-memory windowing, executed sequentially.
prob = NumDiffProblem(dt = dt, order = 0.5, n = n_points, method = RLShortMem())
ws = init_workspace(prob)
compute!(prob.method, ws, data, prob)
# - `RLShortMemThreads()`: Parallelized version of `RLShortMem` leveraging multi-threaded loop scheduling.
prob = NumDiffProblem(dt = dt, order = 0.5, n = n_points, method = RLShortMemThreads())
ws = init_workspace(prob)
compute!(prob.method, ws, data, prob)
# - `RLShortMemCorr()`: Short-memory calculation paired with the error correction algorithm, executed sequentially.
prob = NumDiffProblem(dt = dt, order = 0.5, n = n_points, method = RLShortMemCorr())
ws = init_workspace(prob)
compute!(prob.method, ws, data, prob)
# - `RLShortMemCorrThreads()`: Highly optimized parallel version that combines the short-memory error correction with dynamic multi-threading.
prob = NumDiffProblem(dt = dt, order = 0.5, n = n_points, method = RLShortMemCorrThreads())
ws = init_workspace(prob)
compute!(prob.method, ws, data, prob)