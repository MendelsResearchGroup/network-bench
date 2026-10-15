# Optional analysis

Run dataset difficulty analysis from the repository root after installing the benchmark:

```bash
python -m analysis.difficulty node_optimized --systems 20 --manifest ""
```

This writes `results/node_optimized/difficulty.json` for inspection or further
analysis in Python. It is separate from training and evaluation.
Optional stress analysis lives in `network_bench.analysis.stress` and
is loaded only when `train.stress_metrics` is enabled. Published runs leave it off.
