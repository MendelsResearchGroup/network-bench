"""Run one immutable-manifest cell; do not discard model hyperparameters."""
import json
import sys
from dataclasses import replace
from pathlib import Path
import torch
from network_bench.results.jobs import run_job
from common import job_config


def main():
    torch.set_num_threads(1)
    repo=Path.cwd();root=repo/'results/sweep-v2';index=int(sys.argv[1])
    source=Path(__file__).resolve().parents[2]
    job=json.loads((source/'configs/sweep_v2/jobs.json').read_text())[index]
    config=job_config(job, source, root)
    if len(sys.argv)>2 and sys.argv[2]=='pilot':
        config=replace(config,split=replace(config.split,explicit={k:v[:2] for k,v in config.split.explicit.items()}),
                       train=replace(config.train,epochs=2,selection_start_epoch=1,validate_every=1,
                                     windows_per_sim=2,rollout_schedule=((0,2),),early_stopping_patience=None))
        root=root/'pilots'
    run_job(config, job, index, root)


if __name__=='__main__':
    main()
