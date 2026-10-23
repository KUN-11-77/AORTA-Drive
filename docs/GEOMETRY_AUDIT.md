# NAVSIM Geometry Contract Audit

## Decision

The existing scorer-style geometry path is useful as an evaluation signal but
is not a proof-carrying continuous-footprint checker for AORTA-v8.3.2. It must
not be used to discharge the \(h_j\), internal-node, map-error, or continuous
sweep assumptions without additional construction.

## Audited legacy behavior

The legacy constraint-adapter path:

1. constructs four vehicle corners at each discrete trajectory pose;
2. bilinearly samples a raster EDT-based signed-distance map at those corners;
3. takes the minimum across sampled poses and corners.

The NAVSIM/PDM drivable-area path similarly tests whether the four corners at
each scorer horizon lie in a union of drivable polygons. These procedures do
not establish that the whole oriented vehicle rectangle is in the region, nor
that the footprint swept between adjacent poses remains in the region.

## Constructive falsifiers

R002B freezes three minimal counterexamples:

- four disconnected drivable islands contain all four corners while the
  footprint interior is outside;
- both discrete endpoint footprints are accepted while the interpolated sweep
  crosses a non-drivable gap;
- a center-sampled raster EDT reports one pixel of inside distance at the first
  inside cell although the continuous boundary is only half a pixel away.

Therefore corner checks and raster EDT values cannot receive zero geometry or
map-error budgets.

## Required production checker

Before full Gate A can pass, the implementation must freeze and verify:

- the continuous vehicle footprint and rear-axle-to-center convention;
- the interpolation model between trajectory poses;
- a conservative cover of every challenger swept footprint;
- native sample points proven internal to the native footprint;
- ROI containment and non-empty sets;
- coordinate, dilation, rasterization, and protocol hashes;
- a deterministic \(\varepsilon_{map}\) budget that includes pixel-center,
  boundary-rasterization, interpolation, and transform errors.

The official DAC remains a secondary evaluation metric. It is not interchangeable
with the continuous signed-distance teacher used by the certificate theorem.

## V8.3.2 proof-carrying replacement

The revised contract is asymmetric:

- every challenger sweep is contained in a midpoint-body outer cover dilated by
  the analytic translation, rotation, and declared pose-error radius;
- all fixed-grid cell centers within the outer-cover radius plus one cell radius
  are selected, giving \(h=r/\sqrt{2}\) without a sampling-density assumption;
- native nodes are selected only when their centers are inside an actual native
  footprint at a frozen pose;
- ROI clipping, wrapped adjacent yaw, empty sets, and protocol mismatch reject
  the pair.

R002C qualifies this analytic construction on synthetic straight, rotating,
mixed-motion, boundary, and adversarial 1-Lipschitz cases. It does not by itself
close the NAVSIM coordinate and continuous-teacher error ledger.

## Arithmetic-safe production fallback

R002E adds a deliberately more conservative route for the production proof:

- each challenger segment is enclosed by a rear-axle-centered disk whose radius
  is half the endpoint translation plus the vehicle's maximum rear-axle radius
  and the declared pose error;
- native nodes come only from a rear-axle disk contained in the body for every
  orientation;
- midpoint construction, squared-distance tests, and grid support checks use
  exact rationals for binary64 inputs, while square roots and sums round outward;
- the certificate therefore does not consume the official scorer's four corner
  coordinates and does not rely on trigonometric rounding assumptions.

The official corner path remains a compatibility observation only. A fixed
32-ULP-at-global-coordinate-scale check detects convention-level disagreements;
its measured error is reported, but it is not a correction term in the disk
certificate.

The NAVSIM map cache contains overlapping drivable polygons. The continuous
teacher consequently uses union membership with distance to all raw polygon
boundaries. Internal overlap seams make the field more conservative inside the
union, while avoiding an unproved numerical overlay operation.
