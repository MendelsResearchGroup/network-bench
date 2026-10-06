"""Run one cell of jobs.json, caching checkpoints and metrics separately."""
import json
from dataclasses import replace
from pathlib import Path
import sys
import time
import traceback

import torch

from gnn_physics_benchmark import benchmark
from gnn_physics_benchmark.data import registry
from gnn_physics_benchmark.results import cache
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
    started = time.monotonic()
    for name in ['done', 'failed']:
        (results / name).mkdir(parents=True, exist_ok=True)
    try:
        result = benchmark.run(config, root=results / 'runs')
        record = {**job, 'mode': 'ood' if config.split.ood else 'normal',
                  'training_mode': config.train.mode,
                  'validation_networks': len(config.split.explicit['val']),
                  'test_networks': len(config.split.explicit['test']),
                  'index': index, 'directory': result['directory'],
                  'seconds': time.monotonic()-started, 'metrics': result['metrics']}
        (results / 'done' / f'{index}.json').write_text(json.dumps(record, indent=1)+'\n')
        (results / 'failed' / f'{index}.txt').unlink(missing_ok=True)
        print('COMPLETED', json.dumps(record), flush=True)
    except Exception:
        (results / 'failed' / f'{index}.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
