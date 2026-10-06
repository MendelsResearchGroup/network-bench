"""Export the completed learning-curve study without changing rollout results."""
import json
from datetime import datetime, timezone
from pathlib import Path

from gnn_physics_benchmark.results.cache import export_results


def main():
    repo = Path(__file__).resolve().parents[2]
    payload = export_results(repo / 'results' / 'learning-curve' / 'runs')
    rankings = {}
    for run in payload['runs']:
        run['split_id'] = run['comparison_id']
        run['comparison_id'] = f"learning-{run['dataset']}-{run['mode']}-{run['training_mode']}"
        run['train_networks'] = run['split']['train']
        run['split_seed'] = payload['splits'][run['split_id']]['seed']
        if run['mode'] == 'ood':
            dataset = run['dataset']
            if dataset not in rankings:
                rankings[dataset] = json.loads((repo / 'configs' / 'learning_curve' /
                                                'ood_rankings' / f'{dataset}.json').read_text())['ranking']
            payload['splits'][run['split_id']].update(
                rule='highest_n', training_pool_size=100, ranking_steps=100,
                validation_distribution='lower_ratio',
                cutoff_poisson_ratio=rankings[dataset][run['train_networks'] - 1]['poisson_ratio'])
    payload['generated'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    payload['study'] = 'learning'
    path = repo / 'docs' / 'learning-curves.json'
    path.write_text(json.dumps(payload, indent=1) + '\n')
    print(f"{len(payload['runs'])} runs -> {path}")


if __name__ == '__main__':
    main()
