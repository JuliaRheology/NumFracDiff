---
title: NumFracDiff.jl -- A Julia Package for Fractional Derivatives and Integral Numerical Evaluation
tags:
  - Fractional Calculus
  - High Performance Computing
  - Julia
authors:
  - name: Lorenzo Fonnesu
    affiliation: 1
  - name: Andrea Grassi
    affiliation: 1
  - name: Jonathan Louis Kaplan
    orcid: 0000-0002-2700-5229
    affiliation: 2
  - name: Alexandre J Kabla
    orcid: 0000-0002-0280-3531
    affiliation: 2
  - name: Alessandra Bonfanti
    orcid: 0000-0003-2185-4913
    affiliation: 3
affiliations:
  - name: Department of Electronics Information and Bioengineering, Politecnico di Milano, Piazza Leonardo Da Vinci 32, Milan, 20133, Italy
    index: 1
  - name: Department of Engineering, University of Cambridge, Trumpington St, Cambridge, CB2 1PZ, United Kingdom
    index: 2
  - name: Department of Civil and Environmental Engineering, Politecnico di Milano, Piazza Leonardo Da Vinci 32, Milan, 20133, Italy
    index: 3
date: XX-XX-2026
bibliography: paper.bib
---
# Summary
Fractional-order derivatives generalize the ordinary derivative to non-integer orders and are widely used in various fields of mathematics and engineering [@oldham1974fractional]: for example, in rheology, they are used to study the behaviour of materials with long-term memory [@mainardi2022fractional], whilst in hydrogeology and the physics of porous media, they are fundamental to describing the phenomena of anomalous transport and sub-diffusion [@su2020fractional].

Unlike an integer-order derivative, a fractional derivative cannot be computed from a few nearby elements: its classical definition requires a weighted sum over the entire history of the signal, a cost that compounds quickly when the same signal must be differentiated many times, as is typical during model fitting or batch simulation.

NumFracDiff.jl is a Julia package [@bezanson2017julia] for computing the fractional derivative of a sampled signal under the three classical definitions (Grünwald-Letnikov, Riemann-Liouville, and Caputo [@oldham1974fractional; @diethelm2010analysis; @podlubny1998fractional]) implemented through several interchangeable numerical schemes. In addition to a direct, multithreaded evaluation of the entire history, the package offers, for the Grünwald-Letnikov and Riemann-Liouville cases, a memory-truncation scheme based on the classic short-memory principle [@podlubny1998fractional; @li2015numerical], together with an optional analytical correction that recovers much of the precision lost due to the truncation of the history, and, for the Grünwald-Letnikov case only, an accelerated full-history evaluation via FFT. NumFracDiff.jl is already used inside RHEOS.jl [@kaplan2020rheos], an established Julia package for rheological data analysis.

# Statement of need
Repeated evaluation of a fractional-order derivative is a recurring bottleneck across fields that rely on fractional calculus to model memory-dependent behavior. NumFracDiff.jl addresses this challenge by focusing on the following critical needs:
- **Analysis of Sampled Time-Series Data:** Unlike most existing open-source fractional calculus libraries, which are designed to solve fractional differential equations (FDEs) as initial value problems, NumFracDiff.jl specifically targets users who need to process already-sampled, standalone time-series data.
- **Optimization of the History Bottleneck:** Because classical definitions require summing over a signal's entire history, optimization and fitting routines traditionally suffer massive performance penalties.
- **Memory Management via Reusable Workspaces:** By using a preallocated *NumDiffWorkspace*, the package avoids repeated memory allocations for weights and derivatives when a signal is differentiated many times (e.g., inside an optimization loop or batch simulation).
- **Interchangeable Numerical Approximations:** Instead of forcing users to rely on a single, fixed algorithm, the library exposes a unified API with interchangeable numerical schemes. Users can easily switch between direct multithreaded evaluation, short-memory approximations with analytical corrections, or exact FFT methods based on their computing infrastructures and precision requirements.

# State of the field
Libraries for fractional derivatives already existed, mainly in SciFracX and SciML organizations:
- **FractionalCalculus.jl** [@FractionalCalculus] is designed for evaluating fractional derivatives and integrals of continuous analytical functions $f(x)$ at specific points or over symbolic intervals; it does not accept arbitrary sampled data arrays as direct input. The Short-Memory Principle is discussed in its documentation as a conceptual approximation, but is not exposed as a selectable numerical scheme in its source code - no dedicated struct or function implements it.
- **FractionalDiffEq.jl** [@FractionalDiffEq] is a comprehensive, high-performance solver suite integrated into the SciFracX ecosystem. It is optimized for forward-solving initial value problems of the form $D^\alpha y = f(t,y)$.
- **FdeSolver.jl** [@Benedetti:2022] focuses on solving non-linear systems of fractional ordinary differential equations under the Caputo definition, utilizing FFT-accelerated product-integration rules to handle long-term memory effects efficiently.
- **differint** [@differint] is a Python package that, unlike the three above, targets the direct computation of fractional differintegrals on sampled data arrays rather than the solution of FDEs. It supports the Grünwald-Letnikov, Caputo and Riemann-Liouville definitions and includes an FFT-accelerated GL scheme. However, it recomputes its convolution weights from scratch on every call, offers no reusable workspace, no Short-Memory truncation, and no multithreaded variant.

**NumFracDiff** was built from the ground instead of using or contributing to other libraries for several reason. First, the library was built to serve as the high-performance numerical engine for the RHEOS.jl framework, where fractional operators must be evaluated recursively tens of thousands of times inside tight optimization loops to fit material parameters. Achieving this required a specialized codebase entirely free from the compilation overhead and external dependencies of generalized packages. Second, while existing libraries assume analytical function inputs or solve forward differential equations, experimental rheology is fundamentally data-driven. NumFracDiff treats fractional differentiation strictly as an isolated, discrete array-to-array transformation, operating directly on raw experimental vectors without the overhead of wrapping data in interpolating functions, and separates the one-time cost of generating convolution weights from the per-call computation via its reusable workspace. Third, because fractional derivatives are non-local operators that require evaluating the entire past history of the material, they become exceptionally heavy over large datasets.

# Software design
**NumFracDiff** is designed around a modular, three-tier architecture comprising Methods, Problems, and Workspaces. This separation of concerns ensures a highly customizable interface while optimizing memory management for repetitive evaluations.

The numerical schemes are implemented as concrete types inheriting from an abstract DiffMethod type, supporting like the Grünwald-Letnikov, Caputo, and Riemann-Liouville definitions. This includes specialized variants for short-memory approximations, correction terms, multithreading, and Fast Fourier Transforms.

The mathematical parameters of the operation are encapsulated in a mutable NumDiffProblem structure. This problem configuration pairs with an immutable NumDiffWorkspace structure, which serves as the pre-allocated computational backend holding the cached weights and the output derivative vectors. Decoupling the problem setup from execution provides a major performance advantage in multiple differentiation contexts, such as parameter-fitting loops within RHEOS.jl [Add link to repository?]. Because the heavy vector allocations live inside the reusable workspace, the library completely avoids memory reallocation and garbage collection overhead across thousands of sequential evaluations when processing data of the same length and time-step.

To maximize computational efficiency on large datasets, where the non-local nature of fractional operators introduces inherent complexity, NumFracDiff integrates parallelization strategies at both the loop and hardware levels. The most computationally critical inner loops are optimized through systematic vectorization, instructing the compiler to generate explicit SIMD instructions that fully exploit the registers of modern processors. For methods operating on shared-memory architectures the library leverages Julia's native multithreading parallelism model. This parallelization distributes the calculation of historical convolutions and weight summations across multiple CPU cores, drastically reducing execution times.

# Research impact statement
The computational bottleneck of evaluating non-local fractional operators on large datasets has historically limited the adoption of fractional calculus in routine, real-world data processing. By isolating fractional differentiation as a high-performance, discrete array-to-array operation, NumFracDiff.jl enables the application of fractional calculus to massive experimental datasets without requiring users to develop their own numerical solvers or manage low-level memory allocations.

The package significantly lowers the computational barrier in fields heavily reliant on parameter identification and optimization loops, such as rheology or polymer science. In these fields, fitting a fractional model to experimental data requires calculating a fractional derivative tens of thousands of times in order to minimise an objective function; similarly, calculating a material’s response to deformation involves evaluating several fractional derivatives at thousands of different points. The combination of pre-allocated workspaces, dynamic multithreading and the other optimisation strategies described (including the short memory principle and the FFT) allows optimization routines to execute significantly faster than before.

To demonstrate the practical usefulness of NumFracDiff and the impact that the implemented methods can have, they have been completely integrated and fully tested into a dedicated branch of the RHEOS.jl library [@kaplan2020rheos], directly accelerating core tasks within viscoelastic modeling: for model parameter fitting, both implementations, using the short memory principle and FFT, demonstrated a speedup of around 20 times compared to the current method, whilst for prediction tasks, FFT achieves a speedup of up to 100 times. As well as demonstrating its effectiveness under rigorous, data-driven workflows, this also shows how NumFracDiff.jl can be adopted within complex external frameworks without requiring changes to their public interface.

# AI usage disclosure
Generative AI tools were used during the development of NumFracDiff to assist with the review of the code and documentation, primarily to identify errors and improve syntax and grammar. All AI suggestions were verified by the authors, who designed and developed the architecture and all critical components.

# References
