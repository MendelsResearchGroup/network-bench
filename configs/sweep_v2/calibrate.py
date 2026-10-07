"""PR calibration on the first10 TRAIN trajectories, never validation/test."""
import json
import sys
from dataclasses import replace
from pathlib import Path
import torch
from gnn_physics_benchmark import benchmark, barostat
from gnn_physics_benchmark.data import registry
from gnn_physics_benchmark.graph.features import potential_for
from gnn_physics_benchmark.training import RunConfig

torch.set_num_threads(1)
repo = Path.cwd()
job = json.loads((repo/'configs/sweep_v2/calibrations.json').read_text())[int(sys.argv[1])]
config = RunConfig.from_json(repo/job['config'])
entry = registry.get(config.dataset)
config = replace(config,graph=config.graph.resolved(entry.schema))
names = config.split.explicit['train'][:10]
trajectories = benchmark.load_data(config,names)
potential = potential_for(config.graph,entry)
params = barostat.fit(entry,potential,trajectories,config.graph.window_length)
error = barostat.box_error(barostat.Barostat(entry,potential,params),trajectories,config.graph.window_length)
if error > barostat.TOLERANCE:
    raise ValueError(f"TRAIN calibration {job['name']} relative box RMSE {error} exceeds {barostat.TOLERANCE}.")
result = dict(name=job['name'],params=params,training_networks=names,training_box_error=error,
              method='PR #2 grid search on first10 TRAIN; no evaluation fitting')
(repo/'results/sweep-v2/calibration'/f"{job['name']}.json").write_text(json.dumps(result,indent=1)+'\n')
print('CALIBRATED',json.dumps(result),flush=True)
