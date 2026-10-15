"""Resolve a manifest cell identically for execution and publication."""
import json
from dataclasses import replace
from network_bench.data import registry
from network_bench.training import RunConfig


def job_config(job, source, root):
    config = RunConfig.from_json(source / job['config'])
    params = None
    if job['barostat']:
        calibration = json.loads((root / 'calibration' / f"{job['calibration']}.json").read_text())
        if calibration['training_networks'] != config.split.explicit['train'][:10]:
            raise ValueError('Calibration membership differs from first 10 TRAIN trajectories.')
        params = calibration['params']
    return replace(config, model=job['model'], model_hyperparameters=job['hyperparameters'],
                   graph=config.graph.resolved(registry.get(config.dataset).schema), barostat=params)
