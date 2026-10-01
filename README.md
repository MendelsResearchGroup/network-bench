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

gnn-bench train configs/networks.json --model gns mlp edge_mlp linear_floor frozen --seeds 0 1 2
gnn-bench report node_optimized --seeds
```

This runs each model with three training seeds, sequentially. Each model starts
with its own default hyperparameters. To change a model setting, for example,
use `--model gns --set model_hyperparameters.hidden_size=64`. When selecting
multiple models, overrides must be supported by every selected model. Without `--model`, the model and
hyperparameters come from the config. Without `--seeds`, the config's training
seed is used.

The `edge_mlp` and `mlp` defaults use width 128 and depth 4.
`edge_mlp_delta` uses the same edge-MLP architecture and parameter count, replacing
raw velocity history with the latest velocity and successive differences before
normalization. Run it with `--model edge_mlp_delta`; `edge_mlp` remains the original
baseline for comparison. Both use width 128 and depth 4.

`edge_mlp_attention` adds one attention pool to `edge_mlp_delta`: a learned weight
per node pools its local features into a 32-channel global context, which is
projected back and added before the node MLP. It shares context only within the
current system, uses no future frames, and adds no message-passing rounds.

`configs/networks.json` trains on the first **20 frames per training trajectory**:
four input frames and 16 one-step targets. Validation and test use separate
trajectories, with **100 predicted steps starting from their first four frames**
(104 frames total). The split contains 50 training, 50 validation and 70 test
trajectories; validation rollouts select the checkpoint.

For the 200 Noisy LJ chunks (1500 frames each) under `~/work/data/noisy-lj`:

```bash
GNN_BENCH_DATA_ROOT=~/work/data gnn-bench train configs/networks.json \
  --dataset noisy_lj --model gns mlp edge_mlp linear_floor frozen --seeds 0 1 2
```

Noisy LJ uses the stored bond edges and stiffnesses. The benchmark does not infer
missing rest lengths or LJ parameters; force and stress analysis is unavailable.

### Out-of-distribution comparison

Prepare a fixed split for each dataset, then train the same models and seeds:

```bash
gnn-bench --threads 4 ood-config configs/networks.json \
  --dataset noisy_lj --out configs/ood/noisy_lj.json
gnn-bench --threads 4 train configs/ood/noisy_lj.json \
  --model gns mlp edge_mlp edge_mlp_delta edge_mlp_attention linear_floor frozen --seeds 0 1 2
```

Networks are ranked by ground-truth, position-based Poisson's ratio at the
100-step evaluation horizon: frame 3 is the reference and frame 103 is the
target, using four initial frames. The highest 30% (rounded up) forms the
training/validation pool. Shuffle that pool with data-split seed 42; use its
first 30 networks for training and all the rest for validation. Shuffle the
lower 70% using the same generator, then take at most 100 test networks. The
remaining lower-ratio networks are unused. Equal ratios are ordered by network
ID. `ood-config --train-networks` changes the training count.

| Dataset | Training | Validation | OOD test |
|---|---:|---:|---:|
| Node optimized | 30 | 85 | 100 |
| Stiffness optimized | 30 | 57 | 100 |
| Noisy LJ | 30 | 30 | 100 |

The configs store every network assignment explicitly; accompanying
`*.ranking.json` files record every ratio and the cutoff. Training still uses
the first 20 frames, and checkpoints are selected on high-ratio validation
networks. Test trajectories do not enter training, normalization, or checkpoint
selection. Ground-truth test responses are used to define the OOD split.

The website's **Normal / OOD** toggle switches all results together. **Exact
network split** lists every train/validation/test and unused network. Normal uses 50/50/70
randomly assigned networks, so its sample counts differ from OOD. R² is computed
within each mode's own test population. Parameter counts appear beside model
names, in model comparison cards, and in the Runs table.

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
for model in gns mlp edge_mlp linear_floor frozen; do
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
PBS jobs do this when they finish. The browser plots every dataset with Plotly,
filters models and seeds, and compares seed means with sample standard deviations.
The exported comparison ID keeps different run settings and system splits apart.
Plotly is loaded from its versioned CDN; no build step or backend is needed.
View it with `python -m http.server --directory docs 8000` or publish `docs/`
from `main` with GitHub Pages.

Dataset layouts, force fields, difficulty measures, and scientific caveats are
in the [data and physics reference](docs/reference.md).
