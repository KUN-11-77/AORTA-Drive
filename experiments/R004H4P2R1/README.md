# R004H4P2R1 comparable-speed candidate-supply replay

This is the single correction authorized by R004H4P2S. It replays the exact R004H4P2
anchors, raw current-speed inputs, 35-control provider, eligibility thresholds and adequacy
decision. The only change is the speed observable: native and challenger are both compared
using rear-axle displacement over the final `[3.5,4.0] s` interval divided by `0.5 s`.

This positional quantity is an interval-average speed, not an estimate claimed to equal
instantaneous speed. The replay reads no labels, geometry, predictions, development,
calibration, test or target data.
