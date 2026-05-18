# # Grünwald-Letnikov

using NumFracDiff

# ### Mathematical Definition

# The Grünwald-Letnikov fractional derivative of order $α$ of a function $f(t)$ is defined as the limit of the sum:

# $$D^{α} f(t) = \lim_{h \to 0} \frac{1}{h^α} \sum_{k=0}^{n} w_k^{(α)} f(t - kh)$$

# where $h$ is the step size and $n=\lfloor{(t-t_0)/h}\rfloor$. The coefficients $w_k^{(α)}$ are given by the recursive relation:

# $$w_0^{(α)} = 1 , \quad w_k^{(α)} = w_{k-1}^{(α)} \left(1 - \frac{α + 1}{k}\right)$$

# and can be computed efficiently using the function [`generate_weights!`](@ref).

prob = NumDiffProblem(dt = 0.01, order = 0.5, n = 1000, method = GL())
weights = zeros(NumDiffFloat, 1000)
deriv = zeros(NumDiffFloat, 1000)
ws = NumDiffWorkspace(weights, deriv)
generate_weights!(prob.method, prob, ws, L=1000)

# ### Numerical Implementation

# Given a sampling time of $\Delta t$ and a vector of data points, the derivative at index $i$ is calculated as a weighted sum:

# $$D^\alpha f(t_i) \approx \frac{1}{\Delta t^\alpha} \sum_{j=1}^{i} w_{j-1}^{(\alpha)} f(t_{i-j+1})$$

#md # !!! note "Note"
#md #     The order $\alpha$ can be any real number. When $\alpha > 0$, the operator behaves as a fractional derivative; when $\alpha < 0$, it behaves as a fractional integral.

# ## GL

# The basic Grünwald-Letnikov method computes the fractional derivative by taking into account the complete history of the dataset. While highly accurate, this approach scales with an asymptotic computational complexity of $\mathcal{O}(n^2)$, where $n$ is the number of data points.
# The library offers two dispatch types for this full-memory calculation:
# - `GL()`: The standard, single-threaded sequential implementation.
# - `GLThreads()`: A multi-threaded implementation that parallelizes the outer loop using Julia's native task migration (`@threads :dynamic`) and leverages multi-core processors.

compute!(::GL, ::NumDiffWorkspace, ::Vector{NumDiffFloat}, ::NumDiffProblem)
compute!(::GLThreads, ::NumDiffWorkspace, ::Vector{NumDiffFloat}, ::NumDiffProblem)


# ## GL Short memory

# For long time-series or large datasets, the full-history approach of the standard GL method becomes computationally prohibitive. To address this, NumFracDiff implements the Short Memory Principle.

# This principle assumes that the "memory" of the fractional derivative fades over time. Therefore, the calculation at index $i$ is truncated to include only a fixed window of the recent past, defined by an optimal memory length $L$ (determined automatically via `optimal_L!`):

# $$L \ge \left(\left|\frac{M}{\epsilon_0\,\Gamma(1-\alpha)}\right|\right)^\frac{1}{\alpha}$$

# where $\alpha$ is the fractional order, $\epsilon_0$ is the tolerance we set such that $\Delta t<\epsilon_0$ and $M>0$ is a value greater than $|f(t)|$ in the entire domain. This reduces the asymptotic computational complexity from \mathcal{O}(n^2) to \mathcal{O}(n \cdot L).

# When using a short memory method, the computation is split into two regions:
# 1. `Growing Memory Region` ($i \le L$): The full history is used since the number of available data points is smaller than the memory window $L$.
# 2. `Short Memory Region` ($i > L$): Only the most recent $L$ data points are used in the convolution, dropping older history to save time and memory.

# ### Error Correction

# Truncating the history introduces a deterministic truncation error. To compensate for this loss of accuracy, NumFracDiff offers Memory Correction variants (`GLShortMemCorr` and `GLShortMemCorrThreads`). These methods apply a dynamic correction factor which is added to the standard sum in order to improve precision, while maintaining the speed advantages of the short-memory approach.

# The library provides four dispatch types for short memory calculation:
# - `GLShortMem()`: Standard short-memory truncation, executed sequentially.
# - `GLShortMemThreads()`: Parallelized version of `GLShortMem` using multi-threading (`@threads :dynamic`).
# - `GLShortMemCorr()`: Short-memory truncation with the correction term, executed sequentially.
# - `GLShortMemCorrThreads()`: Parallelized version of `GLShortMemCorr` combining error correction with multi-threading.

compute!(::GLShortMem, ::NumDiffWorkspace, ::Vector{NumDiffFloat}, ::NumDiffProblem)
compute!(::GLShortMemThreads, ::NumDiffWorkspace, ::Vector{NumDiffFloat}, ::NumDiffProblem)
compute!(::GLShortMemCorr, ::NumDiffWorkspace, ::Vector{NumDiffFloat}, ::NumDiffProblem)
compute!(::GLShortMemCorrThreads, ::NumDiffWorkspace, ::Vector{NumDiffFloat}, ::NumDiffProblem)

# ## GL FFT

# While the standard full-memory approach has a computational complexity of $\mathcal{O}(n^2)$, the `GLFFT` method accelerates this operation by performing the convolution in the frequency domain.

# By leveraging the Fast Fourier Transform (FFT) via the `fftfilt` function, the asymptotic computational complexity is reduced to $\mathcal{O}(n \log n)$. Unlike the Short Memory Principle, this acceleration does not truncate the past history; it retains `full-memory accuracy` while offering massive performance gains for large datasets.
compute!(::GLFFT, ::NumDiffWorkspace, ::Vector{NumDiffFloat}, ::NumDiffProblem)