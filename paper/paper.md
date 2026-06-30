---
title:
tags:
  - Fractional Calculus
  - Julia
authors:
  - name: Lorenzo Fonnesu
  - name: Andrea Grassi
  - name: Alessandra Bonfanti 
affiliations:
date:
bibliography: paper.bib
---
# Summary
Fractional-order derivatives generalize the ordinary derivative to non-integer orders and are widely used in various fields of mathematics and engineering: for example, in rheology, they are used to study the behaviour of materials with long-term memory, whilst in hydrogeology and the physics of porous media, they are fundamental to describing the phenomena of anomalous transport and sub-diffusion.

Unlike an integer-order derivative, a fractional derivative cannot be computed from a few nearby elements: its classical definition requires a weighted sum over the entire history of the signal, a cost that compounds quickly when the same signal must be differentiated many times, as is typical during model fitting or batch simulation.

NumFracDiff.jl is a package for computing the fractional derivative of a sampled signal under the three classical definitions (Grünwald-Letnikov, Riemann-Liouville, and Caputo) implemented through several interchangeable numerical schemes. In addition to a direct, multithreaded evaluation of the entire history, the package offers, for the Grünwald-Letnikov and Riemann-Liouville cases, a memory-truncation scheme based on the classic short-memory principle, together with an optional analytical correction that recovers much of the precision lost due to the truncation of the history, and, for the Grünwald-Letnikov case only, an accelerated exact evaluation via FFT. NumFracDiff.jl is already used inside RHEOS, an established Julia package for rheological data analysis.

# Statement of need
Repeated evaluation of a fractional-order derivative is a recurring bottleneck across fields that rely on fractional calculus to model memory-dependent behavior. NumFracDiff.jl addresses this challenge by focusing on the following critical needs:
- **Analysis of Sampled Time-Series Data:** Unlike most existing open-source fractional calculus libraries, which are designed to solve fractional differential equations (FDEs) as initial value problems, NumFracDiff.jl specifically targets users who need to process already-sampled, standalone time-series data.
- **Optimization of the History Bottleneck:** Because classical definitions require summing over a signal's entire history, optimization and fitting routines traditionally suffer massive performance penalties.
- **Memory Management via Reusable Workspaces:** By using a preallocated *NumDiffWorkspace*, the package avoids repeated memory allocations when a signal is differentiated many times (e.g., inside an optimization loop or batch simulation).
- **Interchangeable Numerical Approximations:** Instead of forcing users to rely on a single, fixed algorithm, the library exposes a unified API with interchangeable numerical schemes. Users can easily switch between direct multithreaded evaluation, short-memory approximations with analytical corrections, or exact FFT methods based on their computing infrastructures and precision requirements.

# State of the field
Libraries for fractional derivatives already existed, mainly in SciFracX and SciML organizations:
- **FractionalCalulus.jl**  is designed for evaluating fractional derivatives and integrals of continuous analytical functions $f(x)$ at specific points or over symbolic intervals.
- **FractionalDiffEq** is a comprehensive, high-performance solver suite integrated into the SciML ecosystem. It is optimized for forward-solving initial value problems of the form $D^\alpha y = f(t,y)$.
- **FdeSolver.jl** focuses on solving non-linear systems of fractional ordinary differential equations under the Caputo definition, utilizing FFT-accelerated product-integration rules to handle long-term memory effects efficiently.

**NumFracDiff** was built from the ground instead of using or contributing to other libraries for several reason. First, the library was built to serve as the high-performance numerical engine for the RHEOS.jl framework, where fractional operators must be evaluated recursively tens of thousands of times inside tight optimization loops to fit material parameters. Achieving this required a specialized codebase entirely free from the compilation overhead and external dependencies of generalized packages. Second, while existing libraries assume analytical function inputs or solve forward differential equations, experimental rheology is fundamentally data-driven. NumFracDiff treats fractional differentiation strictly as an isolated, discrete array-to-array transformation, operating directly on raw experimental vectors without the overhead of wrapping data in interpolating functions. Third, because fractional derivatives are non-local operators that require evaluating the entire past history of the material, they become exceptionally heavy over large datasets.

# Software design
**NumFracDiff** is designed around a modular, three-tier architecture comprising Methods, Problems, and Workspaces. This separation of concerns ensures a highly customizable interface while optimizing memory management for repetitive evaluations.
The numerical schemes are implemented as concrete types inheriting from an abstract DiffMethod type, supporting like the Grünwald-Letnikov, Caputo, and Riemann-Liouville definitions. This includes specialized variants for short-memory approximations, correction terms, multithreading, and Fast Fourier Transforms.
The mathematical parameters of the operation are encapsulated in a mutable NumDiffProblem structure. This problem configuration pairs with an immutable NumDiffWorkspace structure, which serves as the pre-allocated computational backend holding the cached weights and the output derivative vectors. Decoupling the problem setup from execution provides a major performance advantage in multiple differentiation contexts, such as parameter-fitting loops within RHEOS.jl. Because the heavy vector allocations live inside the reusable workspace, the library completely avoids memory reallocation and garbage collection overhead across thousands of sequential evaluations when processing data of the same length and time-step.

# Research impact statement

# AI usage disclosure

# References