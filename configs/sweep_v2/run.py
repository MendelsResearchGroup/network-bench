"""Run one immutable-manifest cell; do not discard model hyperparameters."""
import json
import sys
from dataclasses import replace
from pathlib import Path
import torch
from network_bench.data import registry
from network_bench.results.jobs import run_job
from network_bench.training import RunConfig


def main():
    torch.set_num_threads(1)
    repo=Path.cwd();root=repo/'results/sweep-v2';index=int(sys.argv[1])
    source=Path(__file__).resolve().parents[2]
    job=json.loads((source/'configs/sweep_v2/jobs.json').read_text())[index]
    config=RunConfig.from_json(source/job['config'])
    params=None
    if job['barostat']:
        calibration=json.loads((root/'calibration'/f"{job['calibration']}.json").read_text())
        if calibration['training_networks']!=config.split.explicit['train'][:10]:
            raise ValueError('Calibration membership differs from first10 TRAIN.')
        params=calibration['params']
    entry=registry.get(config.dataset)
    config=replace(config,model=job['model'],model_hyperparameters=job['hyperparameters'],
                   graph=config.graph.resolved(entry.schema),barostat=params)
    if len(sys.argv)>2 and sys.argv[2]=='pilot':
        config=replace(config,split=replace(config.split,explicit={k:v[:2] for k,v in config.split.explicit.items()}),
                       train=replace(config.train,epochs=2,selection_start_epoch=1,validate_every=1,
                                     windows_per_sim=2,rollout_schedule=((0,2),),early_stopping_patience=None))
        root=root/'pilots'
    if (root/'done'/f'{index}.json').exists():
        print('ALREADY DONE',index,flush=True)
        return
    run_job(config, job, index, root)


if __name__=='__main__':
    main()
