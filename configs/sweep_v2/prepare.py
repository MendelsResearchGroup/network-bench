"""Fresh full sweep, preserving the published membership exactly."""
import json
from pathlib import Path
from gnn_physics_benchmark import models
from gnn_physics_benchmark.data import registry

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
DATASETS = ('node_optimized', 'stiff_optimized', 'noisy_lj')
MODELS = ('gns', 'mlp', 'tiny_mlp', 'edge_mlp', 'edge_mlp_delta', 'edge_mlp_attention', 'linear_floor')
SCHEDULE = [[0,1],[10,2],[20,3],[30,5],[40,8],[50,10]]


def main():
    jobs, blocked, calibrations = [], [], []
    for dataset in DATASETS:
        for mode in ('normal', 'ood'):
            for seed in (0,1,2):
                calibration = f'{dataset}-{mode}-seed{seed}'
                calibration_config = None
                for training in ('one_step','multi_step'):
                    folder = dataset if mode=='normal' and training=='one_step' else (
                        f'normal_mst/{dataset}' if mode=='normal' else f'ood_{training}/{dataset}')
                    for size in range(10,101,10):
                        original = REPO/'configs/learning_curve'/folder/f'seed-{seed}_train-{size:03d}.json'
                        base = json.loads(original.read_text())
                        for barostat in (False, True):
                            if barostat and registry.get(dataset).potential is None:
                                blocked.append(dict(dataset=dataset,mode=mode,seed=seed,
                                    training_mode=training,train_networks=size,models=list(MODELS),
                                    reason='No verified force field/rest lengths for Noisy LJ barostat.'))
                                continue
                            config = json.loads(json.dumps(base))
                            config['graph'] = config['graph'] | dict(history=3,prediction_stride=1)
                            config['model_hyperparameters'] = dict(n_layers=2)
                            config['poisson_method'] = 'box' if barostat else 'sides'
                            config['barostat'] = {} if barostat else None
                            config['train'].update(detach_rollout=True,selection_start_epoch=51 if training=='multi_step' else 1,
                                select_by='poisson_r2_100',validate_every=2,
                                epochs=100 if training=='multi_step' else 40,
                                early_stopping_patience=None if training=='multi_step' else 5,
                                rollout_schedule=SCHEDULE if training=='multi_step' else [[0,1]])
                            path = ROOT/'configs'/dataset/mode/training/f'barostat-{int(barostat)}'/f'seed-{seed}_train-{size:03d}.json'
                            path.parent.mkdir(parents=True,exist_ok=True)
                            path.write_text(json.dumps(config,indent=1)+'\n')
                            if size==100 and training=='one_step' and not barostat:
                                calibration_config = str(path.relative_to(REPO))
                            for model in MODELS:
                                jobs.append(dict(kind='train',dataset=dataset,mode=mode,seed=seed,
                                    model=model,training_mode=training,train_networks=size,barostat=barostat,
                                    calibration=calibration if barostat else None,config=str(path.relative_to(REPO)),
                                    hyperparameters=models.defaults(model) | (dict(n_layers=2) if model=='gns' else {})))
                            if training=='one_step':
                                for model in ('frozen','zero_acceleration'):
                                    jobs.append(dict(kind='control',dataset=dataset,mode=mode,seed=seed,
                                        model=model,training_mode='one_step',train_networks=size,barostat=barostat,
                                        calibration=calibration if barostat else None,config=str(path.relative_to(REPO)),
                                        hyperparameters={}))
                if registry.get(dataset).potential is not None:
                    calibrations.append(dict(name=calibration,config=calibration_config))
    (ROOT/'jobs.json').write_text('[\n'+',\n'.join(json.dumps(row) for row in jobs)+'\n]\n')
    (ROOT/'calibrations.json').write_text(json.dumps(calibrations,indent=1)+'\n')
    (ROOT/'blocked.json').write_text('[\n'+',\n'.join(json.dumps(row) for row in blocked)+'\n]\n')
    result = REPO/'results/sweep-v2'
    for folder in ('logs','done','failed','calibration'):
        (result/folder).mkdir(parents=True,exist_ok=True)
    protocol = dict(recipe='sweep-v2',datasets=list(DATASETS),models=list(MODELS),seeds=[0,1,2],
        train_sizes=list(range(10,101,10)),validation=70,test=70,
        training_jobs=sum(j['kind']=='train' for j in jobs),control_jobs=sum(j['kind']=='control' for j in jobs),
        blocked_training_jobs=len(blocked)*len(MODELS),gns_message_passing=2,
        mst=dict(epochs=100,detach_rollout=True,early_stopping=False,schedule=SCHEDULE,
                 selection_start_epoch=51,selection='validation reported Poisson R2@100'),
        one_step=dict(epochs=40,early_stopping_patience=5),
        calibration='PR #2 first10 TRAIN trajectories, coarse/fine grid search, dt0.01. Same first10 membership across nested sizes; reuse fixed parameters across sizes/models/training modes. No evaluation fitting.',
        metrics='GT box strain; prediction box with barostat, unchanged group graph_utils sides without. Plain per-coordinate position MSE. Mean/sampleSD of per-seed scores.',
        initialization='Real seed frames0-3; predict4-103. No ITPO, simulator bootstrap, or added input noise.',
        splits='Original exact learning_curve split membership. OOD reserves ranks1-100; trainN uses highestN; heldout shuffled ranks101 onward. Train-pool remainder unused.',
        data_root='/rg/mendels_prj/s.sergey/data_bench')
    (ROOT/'protocol.json').write_text(json.dumps(protocol,indent=1)+'\n')
    print(json.dumps(protocol,indent=1))


if __name__=='__main__':
    main()
