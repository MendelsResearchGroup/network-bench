"""Replace the completed sweep atomically; never mix training recipes."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from gnn_physics_benchmark.results.website import learning_payload


def main():
    repo=Path.cwd();root=repo/'results/sweep-v2'
    source=Path(__file__).resolve().parents[2]
    destination=Path(sys.argv[1]) if len(sys.argv)>1 else repo/'docs'
    jobs=json.loads((source/'configs/sweep_v2/jobs.json').read_text())
    missing=[i for i in range(len(jobs)) if not (root/'done'/f'{i}.json').exists()]
    if missing:
        (root/'incomplete.json').write_text(json.dumps(dict(missing_jobs=missing),indent=1)+'\n')
        raise RuntimeError(f'{len(missing)} sweep cells unfinished; existing website results retained.')
    payload=learning_payload(root/'runs', source/'configs/learning_curve/ood_rankings', recipe='sweep-v2')
    if len(payload['runs']) != len(jobs):
        raise RuntimeError('Finished result count differs from manifest; existing website results retained.')
    generated=datetime.now(timezone.utc).isoformat(timespec='seconds')
    controls=[r for r in payload['runs'] if r['model'] in ('frozen','zero_acceleration')]
    payload['runs'].extend(dict(r,training_mode='multi_step') for r in controls)
    for run in payload['runs']:
        run['comparison_id']=f"learning-v2-{run['dataset']}-{run['mode']}-{run['training_mode']}-barostat{int(run['barostat'])}"
    previous=json.loads((destination/'learning-curves.json').read_text())
    scopes={(r['dataset'],r['mode'],r['training_mode'],r['barostat']) for r in payload['runs']}
    keep=[r for r in previous['runs'] if (r['dataset'],r.get('mode','normal'),r.get('training_mode','one_step'),bool(r.get('barostat'))) not in scopes]
    learning=dict(previous,runs=keep+payload['runs'],splits=previous['splits']|payload['splits'],generated=generated)
    # Rollout view uses the same new 100-network cells; Learning Curves uses all sizes.
    rollout=json.loads((destination/'results.json').read_text())
    keep=[r for r in rollout['runs'] if (r['dataset'],r.get('mode','normal'),r.get('training_mode','one_step'),bool(r.get('barostat'))) not in scopes]
    rollout.update(runs=keep+[r for r in payload['runs'] if r['train_networks']==100],
                   splits=rollout['splits']|payload['splits'],generated=generated)
    for name,value in [('learning-curves.json',learning),('results.json',rollout)]:
        temporary=destination/f'{name}.tmp'
        temporary.write_text(json.dumps(value,indent=1)+'\n')
        temporary.replace(destination/name)
    status=json.loads((destination/'sweep-status.json').read_text())
    status.update(state='complete',completed_jobs=len(jobs),published=generated)
    (destination/'sweep-status.json').write_text(json.dumps(status,indent=1)+'\n')
    print('EXPORTED',len(payload['runs']),'new scores; blocked Noisy LJ barostat cells excluded.')


if __name__=='__main__':
    main()
