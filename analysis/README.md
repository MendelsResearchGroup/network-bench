# Optional analysis

Run dataset difficulty analysis from the repository root after installing the benchmark:

```bash
python -m analysis.difficulty node_optimized --systems 20 --manifest ""
```

This writes `results/node_optimized/difficulty.json`, read by
`notebooks/dataset_metrics.ipynb`. It is separate from the installed training CLI.
Optional stress analysis lives in `gnn_physics_benchmark.analysis.stress` and
is loaded only when `train.stress_metrics` is enabled. Published runs leave it off.
