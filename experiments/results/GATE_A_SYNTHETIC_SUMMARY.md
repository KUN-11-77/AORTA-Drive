# Gate A Synthetic Qualification Summary

## Verdict

**Mathematical core: PASS. Full Gate A: BLOCKED.**

R001–R003 completed on clean implementation revision
`821b0606f6739634ed72f3ca7bd663999d06a87c`. These runs validate the graph
theorem implementation, a controlled continuous-lifting route, calibration/risk
contracts, and conservative binary64 arithmetic. They do not qualify the real
NAVSIM swept-footprint geometry or provide driving-performance evidence.

R002C completed on clean revision
`943e7bb1647a88e038035bf0e153c28b04c890e3`. It validates the v8.3.2
asymmetric analytic sweep-cover construction on synthetic geometry. It does not
close the NAVSIM polygon teacher, coordinate-transform, or geometry floating
arithmetic error ledger.

R002D completed on clean revision
`497b77bf35ed9badb3a595ecfe2f3454dfece486`. It validates exact-rational
polygon boundary-distance intervals, certified node-label error, and outward-safe
map-error summation on synthetic canonical polygons. It does not validate the
actual NAVSIM polygon union or coordinate preprocessing.

## Evidence

| Run | Controlled question | Evidence | Result |
|---|---|---|---|
| R001 | Does the min-plus implementation agree with an independent LP? | 280 randomized chain/cycle/grid comparisons plus shift, zero-edge, nonzero-source, equality and inclusion cases | PASS; 5 tests |
| R002 | Does the continuous correction remain a lower bound? | exact Cartesian internal-net radius, invalid-grid rejection, and 500 adversarial 1-Lipschitz linear teachers with bounded map error | PASS; 4 tests; 0 observed false acceptances |
| R002B | Can legacy corner/discrete-pose/raster shortcuts certify the theorem? | constructive footprint-interior, between-pose sweep, and half-pixel EDT counterexamples | FALSIFIED as certificate route; 3 counterexamples reproduced |
| R002C | Does the asymmetric analytic cover include continuous challenger sweeps while keeping native nodes internal? | straight, rotating and mixed sweeps; analytic inflation; 500 linear teachers; ROI/yaw/empty-set rejection | PASS; 5 tests; 0 observed false acceptances |
| R002D | Does the continuous polygon teacher enclose exact signed distance and sum map-error terms conservatively? | rectangles, concavity, holes, disjoint unions, 500 random labels, exact rational oracle and Lipschitz checks | PASS; 7 tests; all registered intervals/ledgers enclosed |
| R003 | Are calibration, risk, and numerical directions correct? | finite-threshold sample sizes, session maxima, relative event tests, and 500 exact-fraction numerical comparisons | PASS; 9 tests; 0 optimistic bounds |

Machine-readable records: [`R001.json`](R001.json), [`R002.json`](R002.json),
[`R002B.json`](R002B.json), [`R002C.json`](R002C.json),
[`R002D.json`](R002D.json), and [`R003.json`](R003.json).

## Theory optimization accepted

The multi-source solver now rounds every path addition toward positive infinity
and every final difference toward negative infinity. For finite represented
binary64 inputs, the computed envelope is no smaller than the real-arithmetic
envelope and the returned lower bound is no larger than the mathematical bound.
Accordingly, this implementation uses \(\varepsilon_{num}=0\).

This statement does not absorb input quantization, coordinate transforms, map
construction, compiler modes that violate IEEE-754 semantics, or another solver.

## Development correction

The first development run exposed an incorrect hand-computed expected envelope:
a source label of \(-3\) propagated over two unit edges gives \(-1\), not 2.
The test oracle was corrected; the algorithm was not changed to match the wrong
expectation. Formal clean-revision runs were executed only afterward.

## Remaining blockers

1. Audit the actual NAVSIM polygon union and coordinate preprocessing to bound
   `epsilon_transform` and `epsilon_polygon`; synthetic R002D does not discharge them.
2. Add outward-safe geometry arithmetic or an explicit geometry numerical budget,
   then audit vehicle/pose/grid/transform protocol hashes on NAVSIM preprocessing.
3. Freeze the source session manifest, token-to-node projection, missing-evidence
   behavior, and candidate provider before Gate B.

No NAVSIM data, target appearance suite, GPU training, or server execution was
used in these runs.
