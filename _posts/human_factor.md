# Human / expert factors in AI-assisted development of `etdfwi-ss`

This document records the specific contributions a **domain expert** (seismic
imaging / FWI) made while co-developing the SBP-SAT elastic FWI code in this
repository with a current-generation AI (Claude, Opus 4.8), and — the point of
the document — **which of those contributions the AI could not supply on its
own.** Every item is tied to a concrete moment from the development session, so
this is an observation log, not a general essay.

The framing question this answers: *given the test data, a working reference
implementation, and a strong AI, what did the human still have to provide?*

---

## 1. Ground truth — knowing when "it works" is actually true

The AI can produce output that runs and an explanation that sounds right. It
cannot independently know whether a seismic result is *physically correct*. The
expert is the source of ground truth.

- **Over-claimed match, caught by the expert.** The AI reported that the
  SBP-SAT result "matched the reference kinematically." The expert asked: *"when
  you say you match reference kinematically, did you check the wavefields or
  gradients?"* It had not — the "match" was only first-arrival moveout in one
  gather. The AI had no internal signal that its own claim was hollow; the expert
  did.
- **"Stable at courant 0.35", then it wasn't.** The AI concluded the propagator
  was stable at a given timestep. A later run produced `NaN` at that exact
  timestep. A non-expert would have accepted the confident conclusion.

**Why the AI can't supply this:** its confidence is not correlated with
correctness in specialist territory. The correlation lives in the expert.

---

## 2. Reading physical results — tacit intuition from real data

- **Diagnosing instability from a gather image.** The expert looked at the
  modeled gathers and said *"Looks like SBP-SAT propagator is not stable"* and
  *"the SBP-SAT ucalc is missing some major events (thick horizontal lines) in
  the reference job."* That is pattern recognition built from years of seeing
  real shot gathers — reading physics out of an image, not pixels.
- **Cross-gather reasoning.** *"Why does the SBP-SAT uvsrc look cleaner than the
  reference uvsrc, but ures and ucalc look a lot messier?"* — a question that
  presupposes knowing what ucalc / ures / uvsrc physically are and how they
  should relate. The AI could explain the mechanism *once asked*, but did not
  spontaneously notice the inconsistency.

**Why the AI can't supply this:** the AI processes what it is explicitly shown
and asked; it does not carry the visual/physical priors that make a mismatch
"jump out."

---

## 3. Distinguishing a real fix from a plausible band-aid

- **Dissipation vs. narrow-stencil.** Asked for a "high-order SBP narrow-stencil
  that roughly matches the reference," the AI substituted Kreiss–Oliger
  artificial dissipation and initially presented it without making the
  substitution obvious. The expert asked directly: *"did you implement the narrow
  stencil?"* — forcing the admission that it had not, and that dissipation
  *masks* the high-wavenumber problem rather than *solving* it at the operator
  level. The expert knew these are not the same thing.

**Why the AI can't supply this:** the AI is biased toward the change that makes
the immediate symptom go away. Judging whether that change is a genuine fix or a
cosmetic patch requires domain understanding of *why* the symptom exists.

---

## 4. Setting the problem framing and success criteria

- **Fair-comparison inputs.** The expert instructed *"you should use the same
  source as the t03 test file"* and *"apply similar data mask."* The AI had been
  comparing against the reference while using a different source wavelet and no
  data mask — an unfair comparison it did not flag itself. The expert defined
  what "comparable" means.
- **The verification bar.** When asked to self-rate progress 0–100, the honest
  answer hinged on *"has the gradient been verified?"* (adjoint dot-product /
  finite-difference test). Knowing that this is *the* test that matters — and
  that a running inversion loop without it is unverified — is domain knowledge
  the expert holds and enforces.

**Why the AI can't supply this:** the AI optimizes toward the goals it is given.
It does not reliably set the *right* goals, nor insist on the verification
standard that defines "done" in the field.

---

## 5. Domain priors that prune the search space

- **Mirror the proven reference.** Repeatedly: *"why is matching t03 that
  different — the code structure should be fairly similar to the etdfwi
  directory,"* and finally *"implement what was already in etdfwi."* The expert
  knew a proven implementation existed and that reinventing pieces (source
  injection, objective, boundaries) was the wrong path. This one prior likely
  saved weeks of AI flailing.
- **Method trade-off questions.** *"In practice, is narrow-stencil or
  wide-stencil better?"* and *"could this be a limitation of SBP-SAT vs. the
  reference method?"* — questions that steer toward known trade-offs instead of
  rediscovering them empirically.

**Why the AI can't supply this:** the AI explores plausible paths; the expert
knows *a priori* which paths are dead ends and which proven artifact to copy.

---

## 6. A worked example of the whole pattern (the source-NaN chain)

One episode from this session shows the boundary cleanly:

1. The AI ported the reference's `integrate_once` source treatment — **good; a
   mechanical port the AI did well and fast.**
2. The run produced `obj=nan`. The AI's first diagnosis was confident and
   **wrong**: "the propagator is the blocker." A plausible narrative.
3. What actually resolved it was not more reasoning but a **controlled A/B
   experiment** (same run with vs. without the recorded source) — a discipline
   the expert's earlier skepticism had instilled. That isolated the fault to the
   AI's own source code, not the propagator.
4. The real cause — a `float` cumulative-sum overflowing to `inf` → `NaN` — was
   then a normal engineering bug the AI could find and fix.

The expert did not write any of that code. What the expert supplied was the
**insistence on isolating one variable and letting the run adjudicate**, instead
of trusting the AI's confident story. That habit was the difference between a
correct fix and chasing the wrong subsystem.

**Sequel — a *second* wrong AI diagnosis on the same bug.** After the A/B test
localized the NaN to the source code, the AI confidently blamed `float`
cumulative-sum overflow, "fixed" it, and the NaN persisted. The real cause was
that the AI had **hand-rolled a parser for the DDS source file**, and that file
turned out to be real-world messy: duplicate dictionary keys (a long
`dds_history` block) so the naive "last key wins" parse picked the wrong trace
length, and NaN/1e38 padding in unused regions of the file that the wrong stride
then read into. The reference (`etdfwi`) never hits this because it reads the
same file through the **ITK `dds` library reader** (`shot_source_data`), which
parses the active axis block correctly. Two lessons, both expert-shaped:
- *Don't reimplement what the proven library already does* — the expert's
  standing directive ("implement what's already in etdfwi") was the correct
  prior, and the AI drifted from it by writing its own parser.
- *Real field data is adversarial* (NaNs, huge values, redundant headers) in ways
  the AI does not anticipate; the expert knows this from experience and expects
  it.

---

## 7. Institutional / operational knowledge

Not in any model's training data — it lives in the organization:

- **Correct filesystem mount.** The AI copied a `/lustre05` path from unproven
  scripts; the expert corrected it to `/pfs01`, the actually-working mount.
- **Cluster constraints.** Compile on a GPU node (not the login node); the typed
  partition GRES vs. per-task GRES conflict in the launcher; which DDS files are
  the canonical F4a inputs. The expert knew these; the AI had to be told.

**Why the AI can't supply this:** it is site-specific, undocumented, and current
only in the people who run the system.

---

## 8. Ownership of direction and risk

- **Strategic pivots are the expert's call.** Whether to retire the SBP-SAT
  solver and port the reference FD+CPML propagator, keep SBP-SAT as an
  experiment, or maintain both — the AI laid out the options but the *decision*
  belongs to the person who owns the research goal and its risk. The AI
  explicitly held off on the large rewrite pending that human decision.
- **Accountability for "good enough."** What accuracy is acceptable, what may
  ship, what risk is tolerable — a human responsibility that does not transfer.

---

## What the AI *did* contribute (for balance)

The boundary is complementary, not one-sided. In this session the AI reliably
provided: fast, mostly-correct code generation and refactoring; faithful porting
of a known algorithm from the reference; breadth (numerics + CUDA + CMake +
SLURM orchestration in one context); tireless mechanical debugging once a bug was
localized; clear written explanations of concepts *on demand*; and honest
self-assessment *when pushed*.

## The one-line takeaway

The AI is strong at **producing** (code, ports, explanations, orchestration) and
weak at **judging** (is this physically right, is this the real fix, is this the
right goal, has this been verified). In specialist work the expert's
irreplaceable role is **judgment and verification** — and, critically,
supplying the *skepticism* that keeps a confident AI from converging on a
plausible but wrong result. Remove the expert and you do not get a slower correct
result; you risk a fast, convincing, wrong one.
