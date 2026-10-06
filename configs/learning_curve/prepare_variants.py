"""Prepare normal MST and highest-N OOD sweeps with explicit memberships."""
from dataclasses import replace
import json
from pathlib import Path

import torch

from gnn_physics_benchmark.training import RunConfig


def normal_mst(repo, base_jobs):
    directory = repo / 'configs' / 'learning_curve'
    objective = RunConfig.from_json(repo / 'configs' / 'mst' / 'normal.json').train
    for dataset in ['noisy_lj', 'node_optimized', 'stiff_optimized']:
        destination = directory / 'normal_mst' / dataset
        destination.mkdir(parents=True, exist_ok=True)
        for path in sorted((directory / dataset).glob('seed-*.json')):
            if (destination / path.name).exists():
                continue
            config = RunConfig.from_json(path)
            train = replace(config.train, mode=objective.mode,
                            rollout_schedule=objective.rollout_schedule,
                            detach_rollout=objective.detach_rollout)
            replace(config, train=train).to_json(destination / path.name)
        audit = json.loads((directory / dataset / 'splits.json').read_text())
        audit['mode'] = 'normal'
        audit['training_mode'] = objective.mode
        (destination / 'splits.json').write_text(json.dumps(audit, indent=1) + '\n')
    return [{**job, 'mode': 'normal', 'training_mode': objective.mode,
             'config': str(Path('configs/learning_curve/normal_mst') / job['dataset'] / Path(job['config']).name)}
            for job in base_jobs]


def highest_n_ood(repo, base_jobs):
    directory = repo / 'configs' / 'learning_curve'
    objective = RunConfig.from_json(repo / 'configs' / 'mst' / 'normal.json').train
    for dataset in ['noisy_lj', 'node_optimized', 'stiff_optimized']:
        measured = json.loads((directory / 'ood_rankings' / f'{dataset}.json').read_text())
        ordered = [row['system'] for row in measured['ranking']]
        pool, remainder = ordered[:100], ordered[100:]
        audit = {**{k: v for k, v in measured.items() if k != 'ranking'},
                 'mode': 'ood', 'rule': 'highest_n', 'training_pool': pool,
                 'train_sizes': list(range(10, 101, 10)),
                 'validation_networks': 70, 'test_networks': 70,
                 'heldout_rule': 'seeded shuffle of ranks 101 onward; first 70 validation, next 70 test',
                 'same_training_membership_across_seeds': True,
                 'nested_training_subsets': True, 'splits': []}
        for training_mode in ['one_step', 'multi_step']:
            (directory / f'ood_{training_mode}' / dataset).mkdir(parents=True, exist_ok=True)
        for seed in [0, 1, 2]:
            generator = torch.Generator().manual_seed(seed)
            rest = [remainder[i] for i in torch.randperm(len(remainder), generator=generator).tolist()]
            explicit = {'train': pool, 'val': rest[:70], 'test': rest[70:140]}
            used = set(pool + rest[:140])
            audit['splits'].append({'seed': seed, 'training_pool': pool, 'val': explicit['val'],
                                    'test': explicit['test'],
                                    'unused_at_largest_size': [s for s in ordered if s not in used]})
            for size in range(10, 101, 10):
                config = RunConfig.from_json(directory / dataset / f'seed-{seed}_train-{size:03d}.json')
                split = replace(config.split, explicit={**explicit, 'train': pool[:size]}, ood=True)
                for training_mode in ['one_step', 'multi_step']:
                    train = replace(config.train, mode=training_mode,
                                    rollout_schedule=objective.rollout_schedule if training_mode == 'multi_step' else [[0, 1]],
                                    detach_rollout=False)
                    destination = directory / f'ood_{training_mode}' / dataset / f'seed-{seed}_train-{size:03d}.json'
                    if not destination.exists():
                        replace(config, split=split, train=train).to_json(destination)
        for training_mode in ['one_step', 'multi_step']:
            path = directory / f'ood_{training_mode}' / dataset / 'splits.json'
            path.write_text(json.dumps({**audit, 'training_mode': training_mode}, indent=1) + '\n')
    return [{**job, 'mode': 'ood', 'training_mode': training_mode,
             'config': str(Path(f'configs/learning_curve/ood_{training_mode}') / job['dataset'] / Path(job['config']).name)}
            for training_mode in ['one_step', 'multi_step'] for job in base_jobs]


def main():
    repo = Path(__file__).resolve().parents[2]
    path = repo / 'configs' / 'learning_curve' / 'jobs.json'
    jobs = json.loads(path.read_text())
    base_jobs = [job for job in jobs if job.get('mode', 'normal') == 'normal'
                 and job.get('training_mode', 'one_step') == 'one_step']
    additional = normal_mst(repo, base_jobs)
    if all((repo / 'configs' / 'learning_curve' / 'ood_rankings' / f'{dataset}.json').exists()
           for dataset in ['noisy_lj', 'node_optimized', 'stiff_optimized']):
        additional += highest_n_ood(repo, base_jobs)
    present = {(j['dataset'], j['model'], j['seed'], j['train_networks'],
                j.get('mode', 'normal'), j.get('training_mode', 'one_step')) for j in jobs}
    for job in additional:
        key = tuple(job[k] for k in ['dataset', 'model', 'seed', 'train_networks', 'mode', 'training_mode'])
        if key not in present:
            jobs.append(job)
            present.add(key)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(jobs, indent=1) + '\n')
    temporary.replace(path)
    print(f'{len(jobs)} cells; existing indices preserved')


if __name__ == '__main__':
    main()
