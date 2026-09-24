---
layout: post
title: "SBP-SAT on GPU: closing the Triton–CUDA gap"
description: "SBP-SAT on GPU: closing the Triton–CUDA gap"
date: 2026-09-24
permalink: /blog/sbp-sat-on-gpu/
tags:
  - Triton
  - CUDA
  - SBP-SAT
  - Numerical Methods
---

# SBP-SAT on GPU: closing the Triton–CUDA gap

How a masked, branchy Triton kernel ran at 36% of hand-tuned CUDA — and how
restructuring the algorithm (not tuning the kernel) got it to 68%, at the
STREAM roofline. 2D Poisson, 4th-order SBP + SAT, n=1025 in fp64, Tesla T4.

## The setup

The operator is a 2D Laplacian discretized with 4th-order summation-by-parts
(SBP) finite differences in the interior, one-sided boundary closures, and
simultaneous approximation terms (SAT) imposing Dirichlet boundaries weakly.
Diagonal norm, 1.05M unknowns, fp64. The hot loop is the sparse matvec
`y = A x` — a GMRES solve needs dozens of them.

Three implementations of the same matvec:

- **Triton v1** — one kernel, masked loads/stores, runtime branches selecting
  the stencil per grid point.
- **Triton v2** — nine kernels over disjoint write-sets (below).
- **CUDA** — hand-written: tiled shared-memory and direct global-memory
  variants, CUDA-event timed.

Metric: wall-clock ms per matvec, plus effective GB/s against the STREAM
roofline (244.6 GB/s on this T4).

## v1: the gap

| kernel | Triton | CUDA | Triton/CUDA |
|---|---|---|---|
| copy | 0.088 ms | 0.069 ms | 78% |
| stencil5 | 0.185 ms | 0.110 ms | 60% |
| **sbp_full** | **0.669 ms** | **0.236 ms** | **36%** |

The more complex the kernel, the worse Triton did. That's the signature of a
codegen problem, not a hardware problem.

![v1 benchmark ladder: Triton vs CUDA across copy, stencil5, sbp_full]({{ '/assets/images/blog/sbp-sat-on-gpu/ladder.png' | relative_url }})

## Autopsy: why v1 was slow

I dumped the SASS for the v1 kernel. Out of 2,760 instructions, only 201 were
DFMA — **7.3% of the instruction stream did real arithmetic**:

- 184 of 377 global loads were predicated (masked);
- 354 FSEL plus select scaffolding (27.8%) existed only to emulate masked
  loads;
- 152 registers per thread → 384 threads/SM → **18.75% occupancy**;
- zero vectorized memory accesses.

![SASS breakdown of Triton v1: instruction mix and occupancy]({{ '/assets/images/blog/sbp-sat-on-gpu/codegen.png' | relative_url }})

Root cause: the masked, always-execute-both-paths structure. Every thread
loaded both the interior and boundary stencil paths and selected afterward.
No compiler flag fixes an algorithm shaped like this.

## v2: restructure, don't tune

The fix was to change what each thread *writes*, not how it computes. The grid
is partitioned into 9 disjoint write-sets: one branch-free interior kernel
(`i,j ∈ [4, n−5]`, pure 5-point stencil, zero masked loads) plus 8 boundary
rectangles. The stencil structure in each rectangle is chosen by **compile-time
constexprs** — the generated code for each part is a straight-line sequence
with no runtime branches and no masked loads.

This also settled a real correctness hazard: the SAT penalty touches corner
points from two sides, so a naive split would have two threads racing on the
same grid point. Here every point has exactly one writer thread — the corner's
double SAT term is added locally by its owning thread. Race-free by
construction, not by synchronization.

![The 9-part write-set partition at n=33; rings mark the double-SAT corners]({{ '/assets/images/blog/sbp-sat-on-gpu/partition.png' | relative_url }})

Verification before benchmarking: the split algorithm matches the assembled
sparse matrix to ~3e-16 on CPU, and the compiled PTX shows zero predicated
loads and zero fp32 ops.

## Rematch verdict

Correctness on the T4: **1.639e-16** vs a torch fp64 reference. Then:

| kernel | ms | effective GB/s |
|---|---|---|
| copy | 0.0887 | — |
| sbp_interior (BLOCK=128, 4 warps) | 0.2968 | 278.7 |
| sbp_boundary (8 parts) | 0.1896 | — |
| **sbp_full (v2)** | **0.3499** | **241.3** |

That's **1.91× faster than v1** (0.669 ms), now within **1.48× of hand-tuned
CUDA tiled** (0.236 ms) — and the effective bandwidth sits at the STREAM
roofline. The remaining gap is memory reuse (CUDA's shared-memory tiling),
not instruction overhead.

One war story: the first rematch run *failed* — the fail-fast assert tripped
at 1.7e-08 instead of ~1e-16. The algorithm was proven correct on CPU, so it
wasn't a math bug: Colab's triton 3.6.0 was silently demoting `np.float64`
scalar kernel arguments to fp32 (my clean check had run on 3.8.0 — a
version-specific behavior). The fix: constants now ride in a device fp64
tensor that the kernels load directly. Unambiguous on every triton version.
This is why you assert numerics *before* you benchmark.

## What's left

- Fuse the 8 boundary launches (they're launch-bound); try a single persistent
  kernel with constexpr dispatch.
- Per-part `num_warps` autotuning.
- Re-measure on A100/H100, where occupancy and shared memory matter more.

## Takeaways

1. **Masked always-execute-both-paths is the #1 Triton perf killer.**
   Restructure the algorithm; don't tune the kernel.
2. **Compile-time specialization** (`constexpr`) turns runtime branches into
   separate clean code paths — free performance the compiler can't infer.
3. **Assert numerics before benchmarking.** A fail-fast check caught a
   version-specific fp32 demotion that looked exactly like a math bug.
4. Roofline-effective bandwidth is achievable in Triton. The remaining gap to
   CUDA is memory reuse (shared memory), not instruction overhead.

*All numbers measured on a Tesla T4 via Google Colab, 2026-09-24.*
