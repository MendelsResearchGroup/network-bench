"""Train a config JSON, or evaluate a saved run directory.

    python scripts/run.py configs/networks.json
    python scripts/run.py results/<dataset>/<model>/<run>

For model/seed overrides, use dataclasses.replace on RunConfig in Python.
"""
import json
import sys
from pathlib import Path
import torch
from network_bench import benchmark
from network_bench.training import RunConfig

if __name__ == "__main__":
    torch.set_num_threads(4)
    path = Path(sys.argv[1])
    result = (benchmark.evaluate_run(path) if path.is_dir()
              else benchmark.run(RunConfig.from_json(path))["metrics"])
    print(json.dumps(result, indent=2))
