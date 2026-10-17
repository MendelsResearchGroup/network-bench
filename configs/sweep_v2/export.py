"""Publish complete comparisons; keep previous results for unfinished scopes."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from network_bench.results.website import learning_payload
from network_bench.results.jobs import completed_job
from common import job_config


def main():
    repo=Path.cwd();root=repo/'results/sweep-v2'
    source=Path(__file__).resolve().parents[2]
    destination=Path(sys.argv[1]) if len(sys.argv)>1 else repo/'docs'
    jobs=json.loads((source/'configs/sweep_v2/jobs.json').read_text())
    completed=[completed_job(job_config(job, source, root), job, i, root)
               for i, job in enumerate(jobs)]
    scope=lambda job: (job['dataset'],job['mode'],job['training_mode'],bool(job['barostat']))
    # A job with a failure marker is final; its comparison is published as failed.
    failed=[i for i, record in enumerate(completed) if record is None and (root/'failed'/f'{i}.txt').is_file()]
    missing=[i for i, record in enumerate(completed) if record is None and i not in failed]
    pending={scope(jobs[i]) for i in missing}
    if missing:
        (root/'incomplete.json').write_text(json.dumps(dict(missing_jobs=missing),indent=1)+'\n')
        if '--completed-scopes' not in sys.argv:
            raise RuntimeError(f'{len(missing)} sweep cells unfinished; existing website results retained.')
    publish=[(job,record) for job,record in zip(jobs,completed) if record and scope(job) not in pending]
    payload=learning_payload(root/'runs', source/'configs/learning_curve/ood_rankings', recipe='sweep-v2')
    current={(job['dataset'], job['model'], Path(record['directory']).name)
             for job, record in publish}
    payload['runs']=[run for run in payload['runs']
                     if (run['dataset'], run['model'], run['name']) in current]
    if len(payload['runs']) != len(publish):
        raise RuntimeError('Finished result count differs from manifest; existing website results retained.')
    generated=datetime.now(timezone.utc).isoformat(timespec='seconds')
    controls=[r for r in payload['runs'] if r['model'] in ('frozen','zero_acceleration')]
    payload['runs'].extend(dict(r,training_mode='multi_step') for r in controls
                           if scope(dict(r,training_mode='multi_step')) not in pending)
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
    protocol=json.loads((source/'configs/sweep_v2/protocol.json').read_text())
    status.update(state='partial' if missing else 'complete',
                  completed_jobs=len(jobs)-len(missing)-len(failed), published=generated,
                  protocol=protocol,
                  **{key:protocol[key] for key in ('training_jobs','control_jobs','blocked_training_jobs')},
                  pending_scopes=[dict(zip(('dataset','mode','training_mode','barostat'), key))
                                  for key in sorted(pending)],
                  pending_jobs=missing,
                  failed_scopes=[dict(zip(('dataset','mode','training_mode','barostat'), key))
                                 for key in sorted({scope(jobs[i]) for i in failed})],
                  failed_jobs=failed)
    (destination/'sweep-status.json').write_text(json.dumps(status,indent=1)+'\n')
    print('EXPORTED',len(payload['runs']),'new scores; blocked Noisy LJ barostat cells excluded.')


if __name__=='__main__':
    main()
