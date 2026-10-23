# R004H2 — Scale-only proper-loss fit

- **Question:** can the frozen H1 spatial representation fit a non-collapsed bounded edge
  scale shape using only a proper scale loss?
- **Frozen input:** H1 checkpoint SHA-256
  `8b5110b312341f1516067c55d6bb5a9cdec9b230de3837f451f993d5dbc0906b`.
- **Optimization:** exactly 194 scale-head parameters; 42 fit groups, 10 epochs; Stage P
  has zero trainable parameters and must receive no gradients.
- **Selection:** macro mean of per-group mean bounded-Laplace proper loss over the 8
  selection groups, then earliest epoch.
- **Go/no-go:** selected proper loss must strictly beat uniform 1 m on the same frozen
  potential and selection labels; fewer than 95% of scales may lie within sigmoid-position
  `1e-4` of either bound.
- **Controls frozen before execution:** learned, per-observation geometric pooled, uniform
  1 m and deterministic inverse-square-root evidence-only shape with unit geometric gauge.
- **Prohibited:** group-tail term, candidate/planner/certificate loss, Stage-P updates,
  proxy/development/calibration/test/target payloads, search and metric early stopping.
- **Interpretation:** PASS only authorizes H3 proxy-cal workflow rehearsal. It does not
  establish calibration, coverage, task utility, Gate B or novelty.
- **Status:** DRAFT REGISTRATION.
