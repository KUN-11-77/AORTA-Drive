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
