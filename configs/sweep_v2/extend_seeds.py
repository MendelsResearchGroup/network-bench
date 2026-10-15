"""Extend saved learning-curve memberships without changing existing seeds."""
import json
from pathlib import Path
import torch
from network_bench.data import registry


def extend_seeds(repo, seeds=(3, 4)):
    base = repo / 'configs/learning_curve'
    for dataset in ('node_optimized', 'stiff_optimized', 'noisy_lj'):
        normal = json.loads((base / dataset / 'splits.json').read_text())
        roster = normal['systems']
        if set(roster) != set(registry.get(dataset).systems()):
            raise ValueError(f'{dataset}: data roster changed; review the new seed population first')
        ranking = json.loads((base / 'ood_rankings' / f'{dataset}.json').read_text())['ranking']
        ranked = [row['system'] for row in ranking]
        for mode in ('normal', 'ood'):
            for training in ('one_step', 'multi_step'):
                folder = (base / dataset if mode == 'normal' and training == 'one_step' else
                          base / ('normal_mst' if mode == 'normal' else f'ood_{training}') / dataset)
                audit_path = folder / 'splits.json'
                audit = json.loads(audit_path.read_text())
                for seed in seeds:
                    generator = torch.Generator().manual_seed(seed)
                    if mode == 'normal':
                        ordered = [roster[i] for i in torch.randperm(len(roster), generator=generator).tolist()]
                        pool, val, test = ordered[:100], ordered[100:170], ordered[170:240]
                    else:
                        remainder = ranked[100:]
                        ordered = [remainder[i] for i in torch.randperm(len(remainder), generator=generator).tolist()]
                        pool, val, test = ranked[:100], ordered[:70], ordered[70:140]
                    used = set(pool + val + test)
                    record = dict(seed=seed, training_pool=pool, val=val, test=test,
                                  unused_at_largest_size=[name for name in (roster if mode == 'normal' else ranked) if name not in used])
                    previous = next((row for row in audit['splits'] if row['seed'] == seed), None)
                    if previous is not None and previous != record:
                        raise ValueError(f'{folder}: existing seed {seed} differs')
                    if previous is None:
                        audit['splits'].append(record)
                    for size in range(10, 101, 10):
                        config = json.loads((folder / f'seed-0_train-{size:03d}.json').read_text())
                        config['split'].update(seed=seed, explicit=dict(train=pool[:size], val=val, test=test))
                        config['train']['seed'] = seed
                        destination = folder / f'seed-{seed}_train-{size:03d}.json'
                        if destination.exists() and json.loads(destination.read_text()) != config:
                            raise ValueError(f'{destination}: refusing to overwrite a different config')
                        destination.write_text(json.dumps(config, indent=1) + '\n')
                audit_path.write_text(json.dumps(audit, indent=1) + '\n')
