# R004H4P4E official execution-interface audit

This fit-only audit closes the missing operator between desired 0.5-second trajectory poses
and the states actually consumed by NAVSIM's scorer. Native and six signed challengers are
sampled at the official 0.1-second proposal grid, tracked by the official batch LQR, and
propagated by the official kinematic bicycle model from one shared initial ego state.

The audit verifies source hashes, shapes, finiteness, coordinate roundtrip, shared realized
origin and bit-exact deterministic replay on the same 128 already exposed fit anchors.
Desired-to-realized deviation and realized intent supply are secondary diagnostics only and
cannot decide PASS. No geometry, teacher, learned prediction, image, development,
calibration, test or target payload is read.
