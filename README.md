# gnn-physics-benchmark

Compare autoregressive simulator models on the same molecular-dynamics data, the
same input graph, the same loss and the same metrics.

Write a model, register it under a key, and it gets trained and scored exactly
like every other model in the table. Results are cached under
`(dataset key, model key, hyperparameters, split)`, so two numbers in a results
table are comparable by construction rather than by care.

## Run a comparison

```bash
pip install -e .
export GNN_BENCH_DATA_ROOT=/path/to/datasets

gnn-bench train configs/networks.json --model gns mlp tiny_mlp --seeds 0 1 2
gnn-bench report node_optimized --seeds
```

This runs each model with three training seeds, sequentially. Each model starts
with its own default hyperparameters. To change a model setting, for example,
use `--model gns --set model_hyperparameters.hidden_size=64`. When selecting
multiple models, overrides must be supported by every selected model. Without `--model`, the model and
hyperparameters come from the config. Without `--seeds`, the config's training
seed is used.

The data split stays fixed (`split.seed`); `train.seed` controls model
initialisation and training-window sampling. Every run has its own saved config,
weights, and metrics. Repeating the command skips completed runs. `--force`
reruns them.

`report --seeds` groups runs with identical configs except `train.seed`, identical
resolved systems, and the same evaluated split. It shows mean ± sample standard
deviation and the finite count for each metric. A single seed has no standard
deviation. Plain `report` lists individual runs. The short settings ID distinguishes
configurations; full settings are stored with each run.

Use CUDA with `--set train.device=cuda` (CPU is the default):

```bash
gnn-bench train configs/networks.json --model gns --seeds 0 1 2 --set train.device=cuda
```

On the CPU PBS cluster, submit one job per model and seed:

```bash
for model in gns mlp tiny_mlp; do
  for seed in 0 1 2; do
    qsub -v CONFIG=configs/networks.json,MODEL=$model,SET=train.seed=$seed train.pbs
  done
done
```

Add `DATASET=stiff_optimized` to the job variables to override the dataset.
`gnn-bench datasets`, `models`, and `schema` describe the available inputs.

## Where the benchmark lives

| Part | Responsibility |
|---|---|
| `benchmark.py` | Run setup → training → evaluation → saved results |
| `training/loop.py` | Optimisation and validation |
| `evaluate.py` | Score held-out trajectories |
| `results/` | Store runs, compare seeds, export the results page |
| `interfaces/`, `data/`, `models/` | The data and model contracts and implementations |

`difficulty/` is a separate dataset-analysis tool, invoked with
`gnn-bench difficulty <dataset>`. The physics calculations remain in `physics.py`
and `graph/`; the benchmark uses them when graph features or optional stress
metrics require them. Running a comparison does not run difficulty analysis.

External models can import the interfaces after installing this repository from
GitHub with `pip install git+<repository-url>`.

## Add a model

Implement `SimulatorModel.predict_acceleration(graph)` and register the class:

```python
import torch
from gnn_physics_benchmark.interfaces import SimulatorModel
from gnn_physics_benchmark.models import register

class MyModel(SimulatorModel):
    def __init__(self, spec, target_scale):
        super().__init__(spec, target_scale)
        self.net = torch.nn.Linear(spec.node_feature_width, spec.dim)

    def predict_acceleration(self, graph):
        return self.target_scale.inverse(self.net(graph.x))

register("my_model", MyModel)
```

Add it to `models/registry.py` for CLI use, or register it in your own Python
script and call `benchmark.run(config)`. The benchmark builds the graph, fits the
target scale, trains, and scores it. The model interface integrates acceleration
into the next frame.

## Results

Each run writes `config.json`, `split.json`, `history.json`, `metrics.json`, and
`checkpoints/` under `results/<dataset>/<model>/seed-<seed>_<hash>/`.
The hash covers the config and actual split membership. Failures write
`failed.txt` and are retried. Only the current result format is supported.

`relative_mse` compares position error to frozen positions (1 means no better
than standing still). `poisson_r2_<steps>` measures Poisson's ratio across systems
at that rollout horizon. Optional stress metrics remain in the saved metrics.

`gnn-bench evaluate <run-directory>` scores the selected checkpoint using the
saved split. It loads only that split, without preparing training data again.

`gnn-bench export` updates `docs/results.json` for the static results page;
PBS jobs do this when they finish. View it with
`python -m http.server --directory docs 8000` or publish `docs/` with GitHub Pages.

Dataset layouts, force fields, difficulty measures, and scientific caveats are
in the [data and physics reference](docs/reference.md).
