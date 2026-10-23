# R004E-F — Field signal versus tail failure

- **Question:** Does the frozen visual field contain transferable average edge signal
  that is destroyed by the group-maximum endpoint, or is even its central signal no
  better than the zero field?
- **Config:** `configs/experiment/r004ef_field_signal.yaml`.
- **Systems:** frozen visual field, zero field, and exact sign-flipped frozen field.
  Sign flip is a convention diagnostic, not a proposed method.
- **Experimental unit:** one of the same 24 development recording groups. Anchors and
  edges are nested measurements.
- **Tail-mismatch event:** visual group-mean MAE beats zero, visual group maximum loses
  to zero, and group field cosine is positive.
- **Inversion event:** the sign-flipped field beats both visual and zero on both group
  mean and maximum while the original field cosine is negative.
- **No-signal event:** neither original nor sign-flipped visual group-mean MAE beats
  zero.
- **Support:** at least 20/24 groups, with frozen precedence inversion, tail mismatch,
  no signal, then mixed/partial.
- **Status:** COMPLETED — TAIL/OBJECTIVE MISMATCH. The registered joint event held in
  24/24 groups; inversion and no-signal events held in 0/24. See
  `experiments/results/R004EF.md`.
