"""Measure full-roster Poisson rankings at the original 100-step horizon."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import torch

from gnn_physics_benchmark.data import registry
from gnn_physics_benchmark.data.ood import prepare_ood
from gnn_physics_benchmark.training import RunConfig


def main():
    torch.set_num_threads(4)
    repo = Path(__file__).resolve().parents[2]
    dataset = ['noisy_lj', 'node_optimized', 'stiff_optimized'][int(sys.argv[1])]
    config = replace(RunConfig.from_json(repo / 'configs' / 'networks.json'), dataset=dataset)
    _, measured = prepare_ood(config)
    systems = registry.get(dataset).systems()
    audit = {'dataset': dataset, 'ranking_steps': 100, 'history_frames': 4,
             'first_frame': 0, 'frame_stride': 1,
             'data_directory': str(registry.get(dataset).root()),
             'roster_sha256': hashlib.sha256(json.dumps(systems).encode()).hexdigest(),
             'ranking': measured['ranking']}
    directory = repo / 'configs' / 'learning_curve' / 'ood_rankings'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f'{dataset}.json').write_text(json.dumps(audit, indent=1) + '\n')
    print(f'{dataset}: {len(systems)} networks ranked at frames 3–103', flush=True)


if __name__ == '__main__':
    main()
