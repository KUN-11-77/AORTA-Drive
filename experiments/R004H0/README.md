# R004H0 — Scale-interface structural qualification

- **Question:** Can the frozen potential network expose its final spatial map without
  changing potential output, and can a minimal candidate-independent scale head satisfy
  the registered shape, ordering, initialization and scale-only gradient contract?
- **Scope:** metadata, synthetic tests and exactly one existing fit-role anchor. The old
  epoch-1 checkpoint is used only to test strict state loading and interface replay; it is
  prohibited from initializing the fresh R004H1 model.
- **Must pass:** exact role manifest replay; no prohibited role access; exact potential
  output replay; 194 scale parameters; 8064 right/down scales; 1 m initialization within
  `1e-6 m`; finite bounded outputs; no candidate/planner argument; no potential gradient;
  finite nonzero scale gradients; no optimizer step.
- **Interpretation:** PASS authorizes registering fresh fit-only Stage P training. It does
  not establish learned uncertainty, proxy/official coverage, task utility, Gate B or
  novelty.
- **Status:** REGISTERED, not yet executed.

## Preserved preflight repair

The first registered local run passed exact legacy/new potential replay but failed an
unregistered test-only requirement that the float32 spatial mean be bitwise zero. The
observed means were `-9.31e-9` and `-1.49e-8 m`, inside the existing `1e-6 m` zero-gauge
contract. No real payload, optimizer step or later split was used. The repaired registration
removes the unjustified bitwise-zero assertion and makes the existing tolerance explicit in
both config and production audit.

The next local pass completed all 11 focused and 142 repository tests. Strict mypy then
found three static-only issues: a redundant Tensor cast, the generic optional type of the
registered convolution bias, and missing upstream typing for `Tensor.backward`. The repair
removes the cast, explicitly enforces a bias, and scopes one type-ignore to the untyped
third-party call. No tensor formula, parameter, threshold, payload access or experiment
decision changed.
