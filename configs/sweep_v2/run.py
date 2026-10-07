"""Run one immutable-manifest cell; do not discard model hyperparameters."""
import json
import sys
import traceback
from dataclasses import replace
from pathlib import Path
import torch
from gnn_physics_benchmark import benchmark, models
from gnn_physics_benchmark.data import registry
from gnn_physics_benchmark.evaluate import evaluate_model
from gnn_physics_benchmark.normalization import Normalizer
from gnn_physics_benchmark.results import cache
from gnn_physics_benchmark.training import RunConfig


def main():
    torch.set_num_threads(1)
    repo=Path.cwd();root=repo/'results/sweep-v2';index=int(sys.argv[1])
    source=Path(__file__).resolve().parents[2]
    job=json.loads((source/'configs/sweep_v2/jobs.json').read_text())[index]
    done=root/'done'/f'{index}.json'
    if done.exists():
        print('ALREADY DONE',index,flush=True)
        return
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
        done=root/'done'/f'{index}.json'
    try:
        if job['kind']=='train':
            result=benchmark.run(config,root=root/'runs')
        else:
            model=models.build(config.model,config.graph,Normalizer(config.graph.dim))
            data=benchmark.load_data(config,config.split.explicit['test'])
            measured=evaluate_model(model,data,config)
            measured.update(trained_epochs=0,early_stopped=False)
            directory=cache.run_dir(config,config.split.explicit,root/'runs')
            cache.write_result(directory,config,config.split.explicit,metrics=measured)
            result=cache.read_result(directory)
        done.parent.mkdir(parents=True,exist_ok=True)
        done.write_text(json.dumps(dict(index=index,**job,directory=result['directory'],metrics=result['metrics']),indent=1)+'\n')
        (root/'failed'/f'{index}.txt').unlink(missing_ok=True)
        print('COMPLETED',index,json.dumps(result['metrics']),flush=True)
    except Exception:
        (root/'failed').mkdir(parents=True,exist_ok=True)
        (root/'failed'/f'{index}.txt').write_text(traceback.format_exc())
        raise


if __name__=='__main__':
    main()
