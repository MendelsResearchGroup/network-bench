"""Export the completed learning-curve study without changing rollout results."""
import json
from datetime import datetime, timezone
from pathlib import Path

from gnn_physics_benchmark.results.website import learning_payload


def main():
    repo = Path(__file__).resolve().parents[2]
    payload = learning_payload(repo / 'results/learning-curve/runs',
                               repo / 'configs/learning_curve/ood_rankings')
    for run in payload['runs']:
        run['comparison_id'] = f"learning-{run['dataset']}-{run['mode']}-{run['training_mode']}"
    payload['generated'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    payload['study'] = 'learning'
    path = repo / 'docs' / 'learning-curves.json'
    path.write_text(json.dumps(payload, indent=1) + '\n')
    print(f"{len(payload['runs'])} runs -> {path}")


if __name__ == '__main__':
    main()
