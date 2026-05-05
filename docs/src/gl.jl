# # Grünwald-Letnikov

using NumFracDiff

# ### Mathematical Definition

# The Grünwald-Letnikov fractional derivative of order $α$ of a function $f(t)$ is defined as the limit of the sum:

# $$D^{α} f(t) = \lim_{h \to 0} \frac{1}{h^α} \sum_{k=0}^{n} w_k^{(α)} f(t - kh)$$

# where $h$ is the step size and $n=\lfloor{(t-t_0)/h}\rfloor$. The coefficients $w_k^{α}$ are given by the recursive relation:

# $$w_0^{(α)} = 1 , \quad w_k^{(α)} = w_{k-1}^{(α)} \left(1 - \frac{α + 1}{k}\right)$$

# and can be computed efficiently using the function [`generate_weights!`](@ref).

prob = NumDiffProblem(dt = 0.01, order = 0.5, n = 1000, method = GL())
weights = zeros(NumDiffFloat, 1000)
deriv = zeros(NumDiffFloat, 1000)
ws = NumDiffWorkspace(weights, deriv)
generate_weights!(prob.method, prob, ws, L=1000)

# ### Numerical Implementation

# Given a sampling time of $\Delta t$ and a vector of data points,  the derivative at index $i$ is calculated as a weighted sum:

# $$D^\alpha f(t_i) \approx \frac{1}{\Delta t^\alpha} \sum_{j=1}^{i} w_{j-1}^{(\alpha)} f(t_{i-j+1})$$

#md # !!! note "Note"
#md #     The order $\alpha$ can be any real number. When $\alpha > 0$, the operator behaves as a fractional derivative; when $\alpha < 0$, it behaves as a fractional integral.

# ## GL

#md # !!! note "Note"
#md #     Note

# ## GL Short memory

# ## GL FFT