# Configuration guide

Start with **[sweep_v2/README.md](sweep_v2/README.md)** for the currently
published benchmark. Older recipes are retained for reproducibility, not as
defaults for new benchmark runs.

| Location | Role |
|---|---|
| `sweep_v2/` | Current detached-MST, two-layer GNS and barostat-off/on sweep |
| `learning_curve/` | Exact split memberships used by sweep-v2, plus historical recipes |
| `networks.json` | Small standalone one-step example |
| `smoke_*.json` | Small development configurations |
| `mst/` | Historical MST with the earlier schedule and full rollout backpropagation |
| `ood/` | Historical highest-30% OOD splits; not the current highest-N protocol |

Keep `learning_curve/` in place: `sweep_v2/prepare.py` reads those files to
preserve the published network assignments. Do not regenerate splits merely
to tidy these directories. Generated sweep-v2 configs, raw trajectories,
checkpoints and job logs are excluded from Git.
