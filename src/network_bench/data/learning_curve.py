"""Seed-specific held-out splits with nested training subsets."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

from . import loading, registry


def write_configs(config, directory, *, seeds=(0, 1, 2), train_sizes=tuple(range(10, 101, 10))):
    """Write explicit configs: 70 validation, 70 test, and up to 100 training.

    Each seed changes both the network split and training randomness. Within a
    seed the held-out membership stays fixed and training sets are nested.
    The returned audit records the complete input roster and each seed's pool.
    """
    entry = registry.get(config.dataset)
    systems = entry.systems()
    largest = max(train_sizes)
    if len(systems) < largest + 140:
        raise ValueError(f'{config.dataset}: need {largest + 140} networks, found {len(systems)}')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    audit = {
        'dataset': config.dataset,
        'data_directory': str(entry.root()),
        'systems': systems,
        'roster_sha256': hashlib.sha256(json.dumps(systems).encode()).hexdigest(),
        'train_sizes': list(train_sizes),
        'validation_networks': 70,
        'test_networks': 70,
        'nested_training_subsets': True,
        'seed_controls': ['split membership', 'model initialization', 'training randomness'],
        'splits': [],
    }
    for seed in seeds:
        full = loading.resolve_split(entry, sizes=(largest, 70, 70), seed=seed)
        used = {system for part in full.values() for system in part}
        audit['splits'].append({
            'seed': seed, 'training_pool': full['train'], 'val': full['val'],
            'test': full['test'], 'unused_at_largest_size': [s for s in systems if s not in used],
        })
        for size in train_sizes:
            explicit = {**full, 'train': full['train'][:size]}
            split = replace(config.split, explicit=explicit, seed=seed, manifest=None,
                            sizes=None, limit=None, ood=False)
            run = replace(config, split=split, train=replace(config.train, seed=seed))
            run.to_json(directory / f'seed-{seed}_train-{size:03d}.json')
    (directory / 'splits.json').write_text(json.dumps(audit, indent=1) + '\n')
    return audit
