# Gate A Synthetic Qualification Summary

## Verdict

**Mathematical core: PASS. Full Gate A: BLOCKED.**

R001–R003 completed on clean implementation revision
`821b0606f6739634ed72f3ca7bd663999d06a87c`. These runs validate the graph
theorem implementation, a controlled continuous-lifting route, calibration/risk
contracts, and conservative binary64 arithmetic. They do not qualify the real
NAVSIM swept-footprint geometry or provide driving-performance evidence.

## Evidence

| Run | Controlled question | Evidence | Result |
|---|---|---|---|
| R001 | Does the min-plus implementation agree with an independent LP? | 280 randomized chain/cycle/grid comparisons plus shift, zero-edge, nonzero-source, equality and inclusion cases | PASS; 5 tests |
| R002 | Does the continuous correction remain a lower bound? | exact Cartesian internal-net radius, invalid-grid rejection, and 500 adversarial 1-Lipschitz linear teachers with bounded map error | PASS; 4 tests; 0 observed false acceptances |
| R002B | Can legacy corner/discrete-pose/raster shortcuts certify the theorem? | constructive footprint-interior, between-pose sweep, and half-pixel EDT counterexamples | FALSIFIED as certificate route; 3 counterexamples reproduced |
| R003 | Are calibration, risk, and numerical directions correct? | finite-threshold sample sizes, session maxima, relative event tests, and 500 exact-fraction numerical comparisons | PASS; 9 tests; 0 optimistic bounds |

Machine-readable records: [`R001.json`](R001.json), [`R002.json`](R002.json),
[`R002B.json`](R002B.json), and [`R003.json`](R003.json).

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

1. Implement and falsify the production swept-footprint checker for vehicle
   polygons, dilation, pose interpolation, ROI boundaries, and coordinate hashes.
2. Establish a deterministic continuous-teacher/node-label map-error budget on
   the actual NAVSIM preprocessing path.
3. Freeze the source session manifest, token-to-node projection, missing-evidence
   behavior, and candidate provider before Gate B.

No NAVSIM data, target appearance suite, GPU training, or server execution was
used in these runs.
