"""Run one cell of jobs.json, caching checkpoints and metrics separately."""
import json
from dataclasses import replace
from pathlib import Path
import sys

import torch

from gnn_physics_benchmark.data import registry
from gnn_physics_benchmark.results.jobs import run_job
from gnn_physics_benchmark.training import RunConfig


def main():
    torch.set_num_threads(4)
    repo = Path(__file__).resolve().parents[2]
    experiment = Path(__file__).resolve().parent
    results = repo / 'results' / 'learning-curve'
    index = int(sys.argv[1])
    job = json.loads((experiment / 'jobs.json').read_text())[index]
    config = RunConfig.from_json(repo / job['config'])
    config = replace(config, model=job['model'], model_hyperparameters={},
                     graph=config.graph.resolved(registry.get(config.dataset).schema))
    job = {**job, 'mode': 'ood' if config.split.ood else 'normal',
           'training_mode': config.train.mode,
           'validation_networks': len(config.split.explicit['val']),
           'test_networks': len(config.split.explicit['test'])}
    run_job(config, job, index, results)


if __name__ == '__main__':
    main()
