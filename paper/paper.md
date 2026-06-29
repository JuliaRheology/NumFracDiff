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

NumFracDiff.jl is a package for computing the fractional derivative of a sampled signal under the three classical definitions (Grünwald-Letnikov, Riemann-Liouville, and Caputo) implemented through several interchangeable numerical schemes. In addition to a direct, multithreaded evaluation of the entire history, the package offers, for the Grünwald-Letnikov and Riemann-Liouville cases, a memory-truncation scheme based on the classic short-memory principle, together with an optional analytical correction that recovers much of the precision lost due to the truncation of the history, and, for the Grünwald-Letnikov case only, an accelerated exact evaluation via FFT.

NumFracDiff.jl is fully documented and offers a comprehensive test coverage. Moreover, the library is written in Julia, a high-level, high-performance language designed specifically for scientific and numerical computing, and is already in use in the RHEOS (RHEology Open-Source) library.

# Statement of need

# State of the field

# Software design

# Research impact statement

# AI usage disclosure

# References