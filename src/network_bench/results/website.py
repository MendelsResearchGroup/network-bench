"""Build website learning-curve metadata from saved runs and rankings."""
import json
from .cache import export_results


def learning_payload(root, ranking_directory, *, recipe=None):
    payload = export_results(root)
    rankings = {}
    for run in payload['runs']:
        run['split_id'] = run['comparison_id']
        run['train_networks'] = run['split']['train']
        run['split_seed'] = payload['splits'][run['split_id']]['seed']
        if recipe is not None:
            run['recipe'] = recipe
        if run['mode'] == 'ood':
            dataset = run['dataset']
            if dataset not in rankings:
                rankings[dataset] = json.loads((ranking_directory / f'{dataset}.json').read_text())['ranking']
            payload['splits'][run['split_id']].update(
                rule='highest_n', training_pool_size=100, ranking_steps=100,
                validation_distribution='lower_ratio',
                cutoff_poisson_ratio=rankings[dataset][run['train_networks']-1]['poisson_ratio'])
    return flag_incomplete_runs(payload)


def flag_incomplete_runs(payload):
    """Keep survivor-only scores in the download, not in full-population plots.

    Older metrics store only final completion. Missing intermediate counts are
    unknown, not assumed complete. Original values remain in survivor_scores.
    """
    for run in payload['runs']:
        total = run['split'][run['protocol']['evaluation_split']]
        counts = {str(k): v for k, v in run.get('completion_by_step', {}).items()}
        horizons = run['poisson_r2'].keys()
        for step in horizons:
            key = str(step)
            if key not in counts:
                counts[key] = (total if not run['diverged'] else
                               total - run['diverged'] if int(step) == run['rollout_steps'] else None)
        run['completion_by_step'] = counts
        run['evaluation_systems'] = total
        run['incomplete_horizons'] = [int(k) for k, count in counts.items() if count != total]
        for metric in ('poisson_r2', 'position_mse_by_step', 'relative_mse_by_step'):
            for step, value in run[metric].items():
                if counts[str(step)] != total:
                    run.setdefault('survivor_scores', {}).setdefault(metric, {})[step] = value
                    run[metric][step] = None
        if run['diverged']:
            for metric in ('position_mse', 'relative_mse'):
                run.setdefault('survivor_scores', {})[metric] = run[metric]
                run[metric] = None
    return payload
