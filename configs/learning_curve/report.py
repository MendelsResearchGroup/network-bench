"""Summarize the learning curve across seeds with different data splits.

This grouping is deliberately separate from the standard benchmark's seed
summary, which requires identical split membership. Both split membership and
training randomness vary here. Partial summaries state their completed count.
"""
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


METRICS = ['poisson_r2_100', 'position_mse', 'relative_mse', 'diverged',
           'trained_epochs', 'selected_epoch', 'val_poisson_r2_100']


def main():
    directory = Path(__file__).resolve().parent
    results = directory.parents[1] / 'results' / 'learning-curve'
    jobs = json.loads((directory/'jobs.json').read_text())
    rows = [json.loads(p.read_text()) for p in sorted((results/'done').glob('*.json'))]
    groups = {}
    for row in rows:
        key = (row['dataset'], row['model'], row['train_networks'])
        groups.setdefault(key, []).append(row)
    summary = []
    for (dataset, model, size), runs in sorted(groups.items()):
        record = {'dataset': dataset, 'model': model, 'train_networks': size,
                  'validation_networks': 70, 'test_networks': 70,
                  'seeds_completed': sorted(r['seed'] for r in runs)}
        for metric in METRICS:
            values = [r['metrics'][metric] for r in runs
                      if math.isfinite(r['metrics'].get(metric, float('nan')))]
            record[metric] = {'mean': mean(values) if values else None,
                              'std': stdev(values) if len(values)>1 else None,
                              'n': len(values)}
        summary.append(record)
    payload = {'expected_runs': len(jobs), 'completed_runs': len(rows),
               'failed_runs': len(list((results/'failed').glob('*.txt'))),
               'seeds': [0, 1, 2], 'seed_controls': ['split membership', 'training randomness'],
               'dispersion': 'sample standard deviation across split/training seeds',
               'shared_test_within_seed': True, 'nested_training_within_seed': True,
               'summary': summary}
    results.mkdir(parents=True, exist_ok=True)
    (results/'summary.json').write_text(json.dumps(payload, indent=1)+'\n')
    csv_rows = []
    for row in rows:
        csv_rows.append({**{k:row[k] for k in ['dataset','model','seed','train_networks']},
                         **{k:row['metrics'].get(k) for k in METRICS}})
    with (results/'runs.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['dataset','model','seed','train_networks',*METRICS])
        writer.writeheader(); writer.writerows(csv_rows)
    print(f"{len(rows)}/{len(jobs)} completed, {payload['failed_runs']} failed")


if __name__ == '__main__':
    main()
