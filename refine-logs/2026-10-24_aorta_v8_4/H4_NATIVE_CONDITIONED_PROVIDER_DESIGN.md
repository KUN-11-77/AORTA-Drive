# H4 native-conditioned provider design constraints

> Status: theory/design constraint after R004H4P2R1; not an implemented or registered
> experiment.

## Why the frozen provider is insufficient

The current provider depends only on current speed and uses one constant acceleration and
one constant curvature over four seconds. After repairing speed semantics it supplies only
82/128 anchors. Among the 46 unsupplied anchors, 23 have a candidate inside the progress
interval but fail lateral or heading compatibility, while 23 fail progress or the native
minimum-displacement definition. Adding a denser acceleration grid alone cannot close the
lateral/heading half of this gap.

## Proposed family

Let the frozen planner's native terminal displacement be \(q_N=r_N(T)-r_N(0)\), and define
one route-conditioned but time-independent normal

\[
N_*=\frac{(-q_{N,y},q_{N,x})}{\|q_N\|_2}.
\]

A native-conditioned challenger has

\[
r_\delta(t)=r_N(t)+\delta b(t)N_*,
\]

where the bump is supported strictly before the final comparison interval:

\[
b(t)=64u^3(1-u)^3,\quad u=t/T_b,\quad T_b=3.5\text{ s},
\]

for \(0\le t\le T_b\), and \(b(t)=0\) for \(t\ge T_b\). Thus

\[
b(0)=b'(0)=b''(0)=b(T_b)=b'(T_b)=b''(T_b)=0.
\]

The fixed normal is deliberate: after piecewise-quintic reconstruction, both challenger
coordinates remain piecewise polynomials of degree at most six. This makes the continuous
dynamic screen auditable through Bernstein bounds and avoids hiding a normalization or
time-varying-frame approximation inside the provider.

For a differentiable native curve, the bump equalities imply that challenger and native share
the initial state, the entire final `[3.5,4.0] s` positional segment, terminal position,
terminal tangent and the registered final-segment average-speed observable. Hence their
endpoint intent metrics are progress ratio 1, cross-track 0, heading difference 0 and speed
difference 0, up to certified numerical/reconstruction error. The middle horizon can still
have bidirectional swept-set non-inclusion.

## Physical and implementation obligations

This construction is not accepted merely because endpoint algebra closes. Before any
real-data supply run it must pass all of the following on synthetic and fit-only inputs:

1. reconstruct a piecewise differentiable native curve without reading map, teacher,
   learned field, image, label or development data;
2. derive challenger heading from the perturbed curve tangent rather than retaining native
   heading and silently introducing sideslip;
3. fail closed when the curve derivative degenerates or certified endpoint errors exceed
   the frozen intent budgets;
4. enforce speed, curvature, longitudinal acceleration and lateral-acceleration envelopes;
5. preserve exact native samples from `3.5 s` onward, so the comparable speed observable is
   identical by construction;
6. deduplicate zero-amplitude/equivalent candidates and retain deterministic provenance;
7. prove with analytic and adversarial unit cases that no endpoint guarantee depends on
   floating-point equality alone;
8. report candidate supply separately from continuous non-inclusion and teacher opportunity.

For each active polynomial segment, the implementation rewrites the dynamic limits as
scalar polynomial inequalities in local time:

\[
\|v\|^2\le v_{\max}^2,\qquad
(v^\top a)^2\le a_{\parallel,\max}^2\|v\|^2,
\]

\[
(v_xa_y-v_ya_x)^2\le a_{\perp,\max}^2\|v\|^2,
\qquad \|a\|^2\le a_{\max}^2.
\]

Recursive Bernstein convex-hull bounds must place every constraint below a strict negative
margin over the complete segment; unresolved intervals are rejected. This is a conservative
floating-point arithmetic screen, not a formal real-arithmetic proof, and the claim ledger
must preserve that distinction.

## Non-claims

The native trajectory is available from the frozen planner at inference, so conditioning on
it is not teacher leakage. Nevertheless this provider is only an evaluation/proposal
interface unless later evidence shows independent algorithmic value. It is not yet part of
the AORTA neural architecture, is not a safety guarantee, and cannot be called innovative
before baseline comparison and novelty review.
