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

# Software design

# Research impact statement

# AI usage disclosure

# References