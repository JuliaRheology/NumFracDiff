# # Caputo

using NumFracDiff

# ### Mathematical Definition

# The Caputo fractional derivative of order $\alpha$ of a function $f(t)$ is defined by anchoring the derivative directly to the derivatives of integer order:

# $$D^{\alpha} f(t) = \frac{1}{\Gamma(1-\alpha)} \int_{t_0}^{t} \frac{f'(\tau)}{(t-\tau)^\alpha} d\tau, \quad t \ge t_0.$$

dt = 0.01
n_points = 1000
t = [i * dt for i in 0:(n_points-1)]
data = t .^ 2
prob = NumDiffProblem(dt = dt, order = 0.5, n = n_points, method = Caputo())
ws = init_workspace(prob)
generate_weights!(prob.method, prob, ws, L=1000)

# ### Numerical Implementation

# Given a sampling interval $\Delta t$, the derivative at the initial index is zero ($D^\alpha f(t_1) = 0$). For any subsequent index $i$, by approximating the first derivative $f'(\tau)$ via a forward finite difference (the classical L1 scheme) over a uniform grid, the continuous integral is discretized into a weighted sum of historical increments:

# $$D^\alpha f(t_i) \approx \frac{\Delta t^{-\alpha}}{\Gamma(2-\alpha)} \left[ \sum_{j=1}^{i-1} w_j^{(\alpha)} \left(f(t_{i-j+1}) - f(t_{i-j})\right) + w_i^{(\alpha)} f(t_1) \right]$$

# The weights $w_k^{(\alpha)}$ are computed `in-place` using [`generate_weights!`](@ref) and are generated as follows:
# - $w_1^{(\alpha)} = 1$
# - $w_k^{(\alpha)} = k^{1-\alpha} - (k-1)^{1-\alpha} \quad \text{for } k \ge 2$

# ## Caputo

# The standard Caputo method performs a full-history evaluation over the dataset. Because the calculation at each index $i$ requires looping through all previous data increments, this approach features an asymptotic computational complexity of $\mathcal{O}(n^2)$, where $n$ is the total number of points in the signal. The library offers two dispatch types for this full-memory calculation:
# - `Caputo()`: The standard, single-threaded sequential implementation.
prob = NumDiffProblem(dt = dt, order = 0.5, n = n_points, method = Caputo())
ws = init_workspace(prob)
compute!(prob.method, ws, data, prob)
# - `CaputoThreads()`: A parallelized implementation that distributes the outer loop across available CPU cores using Julia's native task migration (`@threads :dynamic`) and speeds up inner loops via `@simd` vectorization.
prob = NumDiffProblem(dt = dt, order = 0.5, n = n_points, method = CaputoThreads())
ws = init_workspace(prob)
compute!(prob.method, ws, data, prob)