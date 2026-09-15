---
layout: post
title: "Porting 2-D SBP-SAT Operators to Triton"
description: "A practical look at translating a variable-coefficient SBP-SAT operator into race-free Triton kernels and measuring its scaling on an NVIDIA Tesla T4."
permalink: /blog/porting-sbp-sat-kernels-to-triton/
tags:
  - Triton
  - SBP-SAT
  - GPU Computing
  - Numerical Methods
math: true
---

High-order PDE solvers often spend most of their time applying structured differential operators. The arithmetic is local, but boundary closures, variable coefficients, and memory layout make the GPU implementation less straightforward than a textbook stencil.

This experiment ports a two-dimensional, variable-coefficient summation-by-parts/simultaneous-approximation-term (SBP-SAT) operator from CUDA.jl-style kernels to [Triton](https://triton-lang.org/). The implementation also includes the diagonal SBP norm and its inverse, along with prolongation and restriction kernels for a multilevel solver.

The complete experiment is available in the [`Triton_SBP_SAT.ipynb`]({{ '/notebooks/Triton_SBP_SAT.ipynb' | relative_url }}) notebook.

## The discrete operator

On a mapped two-dimensional grid, the second-order variable-coefficient operator has the schematic form

$$
\mathcal{L}u =
\partial_r\!\left(c_{rr}\,\partial_r u + c_{rs}\,\partial_s u\right)
+
\partial_s\!\left(c_{rs}\,\partial_r u + c_{ss}\,\partial_s u\right).
$$

The arrays `crr`, `css`, and `crs` hold the metric-dependent coefficients. In the interior, each output uses the center point, its four axial neighbors, and four diagonal neighbors. The implementation groups those contributions as

$$
A_{rr}u + A_{ss}u + A_{rs}u + A_{sr}u,
$$

where the first two terms approximate the pure derivatives and the final two terms approximate the mixed derivatives.

SBP operators are designed so that their discrete derivative and norm matrices reproduce integration by parts. In one dimension this defining relation is

$$
H D + D^T H = B,
$$

where $H$ is a positive-definite discrete quadrature norm and $B$ contains the boundary contribution. SAT terms then impose boundary or interface conditions weakly while preserving an energy estimate. That split is reflected directly in the GPU implementation: one kernel handles regular interior points and four specialized kernels handle the faces and corners.

## Preserving the storage contract

The source arrays use Julia's column-major linearization. If $(i,j)$ denotes a one-based grid coordinate and `Nr1` is the leading dimension, the Julia index is

$$
k = (i-1)N_r + j.
$$

Triton and PyTorch use zero-based offsets, so the corresponding address is

$$
g = (i-1)N_r + (j-1).
$$

Keeping this contract explicit was essential. A kernel can execute successfully while silently transposing its numerical interpretation if the flat storage order is changed during a port.

The interior launch maps a two-dimensional Triton program grid directly to the physical grid:

```python
def launch_x_interior(hr, hs, x, Nr1, Ns1, crr, css, crs, out):
    grid = (max(Ns1 - 2, 0), max(Nr1 - 2, 0))
    x_interior_kernel[grid](
        hr, hs, x, Nr1, Ns1, crr, css, crs, out
    )
    return out
```

Each program computes one interior point. This is deliberately simple: the first goal was a transparent translation whose indexing could be checked against the original formulas.

## Making the boundary update race-free

The original monolithic operator allowed boundary threads to write the boundary value and neighboring output entries. Translating that literally would make ownership difficult to reason about and could introduce write races.

The Triton version instead decomposes the application into five launches:

1. `x_interior_kernel` writes regular interior points.
2. `x_f1_kernel` and `x_f2_kernel` apply the left and right SAT closures.
3. `x_f3_kernel` and `x_f4_kernel` apply the bottom and top closures.
4. Explicit corner branches apply the half-weighted two-dimensional closures.

The face kernels retain the one-sided derivative stencil used by the SAT term. For example, the normal derivative at a boundary uses

$$
\partial_n u \approx \frac{1}{h}
\left(\frac{3}{2}u_0 - 2u_1 + \frac{1}{2}u_2\right).
$$

Separating the faces costs additional launches, but it makes output ownership visible and keeps the irregular boundary logic out of the interior kernel. That is a useful trade while establishing correctness; fusion can be revisited after reference tests are in place.

## The SBP norm and transfer operators

For this second-order tensor-product SBP discretization, the diagonal norm weights are $1$ in the interior, $1/2$ on an edge, and $1/4$ at a corner. The Triton kernel applies $H$ without assembling a matrix:

```python
boundary_x = (i == 1) | (i == ns1)
boundary_y = (j == 1) | (j == nr1)
corner = boundary_x & boundary_y
edge = boundary_x | boundary_y

scale = hr * hs
out = tl.where(
    corner,
    0.25 * scale * value,
    tl.where(edge, 0.5 * scale * value, scale * value),
)
```

The inverse kernel uses the reciprocal weights. The notebook also implements bilinear prolongation and full-weighting restriction, including separate formulas for edges and corners. Together, these kernels cover the main building blocks needed to use the SBP-SAT operator inside a geometric multigrid iteration.

## Benchmark setup

The saved run used an NVIDIA Tesla T4 with single-precision PyTorch tensors. Square grids ranged from $32^2$ to $512^2$. Each measurement used 10 warm-up launches followed by 100 timed launches, with CUDA synchronization before and after the timed region.

## Performance results and analysis

<figure>
    <img src="{{ '/assets/images/triton-sbp-sat-benchmark.png' | relative_url }}" alt="Log-scale benchmark plot of the Triton interior, left-face SAT, and H matrix kernels for square grids from 32 by 32 to 512 by 512." loading="lazy">
    <figcaption>Kernel execution time on an NVIDIA Tesla T4. Lower is better; the vertical axis is logarithmic.</figcaption>
</figure>

Three representative kernels were measured:

| Kernel | Work per launch | Observed behavior |
|:--|:--|:--|
| Interior operator | $O(N^2)$ | Runtime grows strongly once the grid is large enough to dominate launch overhead. |
| Left-face SAT closure | $O(N)$ | Runtime remains nearly flat over the measured range. |
| Diagonal $H$ application | $O(N^2)$ | Runtime grows with the number of grid points, below the more arithmetic-heavy interior operator. |

The interior kernel rises from roughly 35 microseconds at $64^2$ to about 1.25 milliseconds at $512^2$. Over that interval the number of grid points grows by $64\times$, while runtime grows by about $36\times$. The sublinear increase relative to point count is consistent with fixed launch overhead and low GPU occupancy dominating the smaller cases. It should not be interpreted as a better-than-linear algorithm: each interior point still performs a fixed stencil, so the asymptotic work is $O(N^2)$.

The diagonal $H$ kernel follows the same area-dependent trend but remains substantially faster, reaching roughly 450 microseconds at $512^2$. This is expected because each point needs only one load, a boundary-dependent scale, and one store. The interior operator loads several coefficient and neighboring solution values and evaluates pure and mixed derivative terms, so it has much greater memory traffic and arithmetic work per point.

The left-face SAT kernel stays in a narrow range of roughly 30--45 microseconds after the smallest case. Its work grows only with one grid dimension, from 32 to 512 boundary points, and that amount of parallelism is too small to amortize launch overhead effectively on a T4. The slight non-monotonicity is therefore best treated as timing noise and launch-level effects, not as meaningful scaling.

At $512^2$, the interior launch is about $35\times$ slower than one face launch. This confirms that optimizing the full operator should focus first on the two-dimensional interior and on reducing total launch and memory traffic across all five operator kernels. Possible next experiments include fusing compatible face work, processing multiple points per Triton program, and reporting effective memory bandwidth. Those changes should come only after comparison against a numerical reference.

These timings are best read as a scaling sanity check, not a performance comparison. The notebook does not retain the raw timing arrays, and it does not benchmark the original CUDA.jl implementation or a tuned PyTorch baseline.

## What the experiment establishes

The notebook's smoke test successfully launches the interior, all four face closures, $H$, $H^{-1}$, prolongation, and restriction on the Tesla T4. The benchmark also separates the expected $O(N)$ boundary work from the $O(N^2)$ interior work.

There is an important distinction between *runs successfully* and *is numerically verified*. Before this port is used in a solver, the next validation steps should be:

1. Compare every kernel against a CPU or CUDA.jl reference on small deterministic arrays.
2. Verify the discrete SBP identity and an energy estimate, including corner terms.
3. Use a manufactured solution to measure convergence under grid refinement.
4. Check prolongation and restriction for constant preservation and adjoint consistency.
5. Benchmark end-to-end operator applications after correctness is established.

The main implementation lesson is that boundary closures should remain first-class computational regions. Triton makes the interior stencil concise, but the reliability of an SBP-SAT port still depends on explicit indexing, unambiguous write ownership, and tests derived from the numerical method rather than from kernel execution alone.

## Full code

<details markdown="1">
<summary><strong>Show the complete Triton implementation and benchmark</strong></summary>

```python
{% include triton-sbp-sat-full-code.py %}
```

</details>