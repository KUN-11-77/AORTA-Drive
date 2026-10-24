# R004H4P2S cached-native speed-semantics audit

This read-only audit resolves one implementation ambiguity before the H4 candidate supply
result is allowed to influence theory or architecture. On the exact 128 fit anchors used by
R004H4P2, it compares the terminal velocity stored in each cached PDM-Closed `EgoState` with
the average rear-axle speed derived from the last two registered position samples.

The two quantities are not declared interchangeable in advance. A moving anchor with more
than `1.0 m/s` disagreement rejects the cached dynamic field for H4 intent matching. The
audit also records hashes of the NAVSIM generator and array-conversion sources. It reads no
teacher, prediction, geometry, development, calibration, test or target payload.
