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
    return payload
