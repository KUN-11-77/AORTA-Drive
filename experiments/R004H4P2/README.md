# R004H4P2 fit-only candidate-supply diagnosis

This run keeps the R004C candidate provider and every H4P intent threshold unchanged. It
uses all 16 anchors from the first eight fit rows in the frozen role manifest to determine
whether the one-anchor progress gap is isolated or systematic.

The primary endpoint is the fraction of anchors with at least one fully intent-eligible
candidate. The run reads cached PDM-Closed trajectories and current raw ego velocity only;
it does not read visual features, teacher labels, map fields, predictions, development,
calibration, test, or target data. PASS can authorize only a separate fit-only geometry
preflight.
