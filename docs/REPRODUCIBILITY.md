# Reproducibility Contract

Every reported result must resolve to:

1. a Git commit;
2. a versioned configuration;
3. dataset and split fingerprints;
4. environment metadata;
5. random seeds;
6. raw per-scene outputs;
7. an aggregation command;
8. a human-readable result card.

Large artifacts live outside Git under `NAVSIM_EXP_ROOT`. The repository stores manifests and relative artifact identifiers, never machine-specific absolute paths or private credentials.

The initial repository has no valid experimental result. Creating the structure is not evidence for any research claim.
