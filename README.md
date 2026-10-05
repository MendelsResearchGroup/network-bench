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

gnn-bench train configs/networks.json --model gns mlp edge_mlp linear_floor frozen --seeds 0 1 2
gnn-bench report node_optimized --seeds
```

Data is read directly from `/rg/mendels_prj/s.sergey/data_bench` by default:
`node_optimized/` (381 trajectories), `stiff_optimized/` (288 trajectories), and
`data_LJ_noisy_eps0.01_sigma1.0_cutoff1.122/` (1348 trajectories). Set
`GNN_BENCH_DATA_ROOT=/path/to/data` to use another root with these directory names.
The repository does not store trajectory copies.

Existing results record their original network memberships. New seeded splits
use the full current file roster, so expanding Noisy LJ from 200 to 1348 files
changes membership even with split seed 42. Pointing at the data does not start
training or regenerate the published results.

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

All supplied training configs enable early stopping after **five validation
checks without improvement**. `train.epochs` is the maximum, currently 40 in the
benchmark configs. With validation every two epochs, patience spans ten epochs.
Stopping follows `train.select_by` (the benchmark uses `poisson_r2_100`); without
an explicit selection metric, it follows validation loss. Evaluation restores
the best validation checkpoint, including when training reaches the epoch cap.
Set `train.early_stopping_patience` to `null` to disable stopping. Existing
saved runs without this setting retain their original fixed-epoch behavior.
Results record actual trained epochs separately from the best checkpoint epoch.

Multi-step training (MST) feeds predictions back into the model and averages the
loss over every rollout step, with gradients through the entire rollout.
`configs/mst/normal.json` uses the same network split as the normal benchmark;
`configs/mst/*_ood.json` keep the exact OOD memberships. The curriculum uses
1 step in epochs 1–2, 2 in 3–4, 3 in 5–6, 5 in 7–8, and 10 from epoch 9.
With 16 starting windows and four input frames, MST consumes the first **29
frames** per training trajectory. Validation and test still predict 100 steps.
Early stopping uses five validation checks without improvement, with a 40-epoch
cap. The frozen-position baseline has no learned dynamics and is included for
reference in both training modes.

```bash
gnn-bench train configs/mst/normal.json --dataset node_optimized \
  --model gns mlp edge_mlp edge_mlp_delta edge_mlp_attention linear_floor frozen \
  --seeds 0 1 2
gnn-bench train configs/mst/node_optimized_ood.json \
  --model gns mlp edge_mlp edge_mlp_delta edge_mlp_attention linear_floor frozen \
  --seeds 0 1 2
```

For the 1348 Noisy LJ chunks (1500 frames each) under the shared data root:

```bash
gnn-bench train configs/networks.json \
  --dataset noisy_lj --model gns mlp edge_mlp linear_floor frozen --seeds 0 1 2
```

The autoregressive Noisy LJ benchmark uses the stored bond edges and stiffnesses.
Its dataset declaration does not infer missing rest lengths or LJ parameters;
force and stress analysis is unavailable.

### Noisy LJ: two-frame predictions

**Every run seeds the rollout with exactly frames 0–3 (history 3), then predicts
to frame 103.** The [comparison plot](configs/noisy_lj_stride/predict2_comparison.png),
[PDF](configs/noisy_lj_stride/predict2_comparison.pdf),
[CSV](configs/noisy_lj_stride/predict2_comparison.csv) and
[full results](configs/noisy_lj_stride/predict2_comparison.json) compare GNS and
MLP predictions over the same original 3→103 interval.

| Model | Frames per prediction | Test Poisson R² | Relative position MSE |
|---|---:|---:|---:|
| GNS | 1 | 0.252 ± 0.024 | 0.204 |
| GNS | 2 | 0.183 ± 0.038 | 0.106 |
| MLP | 1 | 0.050 ± 0.175 | 0.217 |
| MLP | 2 | 0.145 ± 0.190 | 0.112 |

These are spread-window training results, with means and sample standard
deviations across three seeds. Two-frame GNS predictions reduce mean position
MSE by about 48% versus the one-frame spread control, while Poisson R² falls.
The published MST GNS remains stronger for Poisson R² (0.264). The corrected
MLP score of 0.145 is below the website's one-step MLP result (0.197).

All runs use the same Normal split: 50 training, 50 validation and 70 test
networks, split seed 42, and training seeds 0, 1 and 2. New spread-window runs
use 16 windows, a 40-epoch cap and early stopping after five validation checks.
Checkpoints are selected by validation Poisson R² at 100 original stored-frame
intervals. Exact network IDs and training target frames are saved in the JSON;
window starts differ slightly because the two-frame target needs one additional
frame. Historical one-step head-window controls used fixed 40-epoch training;
historical MST controls used early stopping.

`graph.prediction_stride=2` keeps the four consecutive seed frames and predicts
two frames per call. Intermediate rollout frames are linearly interpolated from
predictions to maintain the consecutive velocity history. Future ground truth
is never fed back in, and the reported 10/20/…/100-frame metrics land on predicted
endpoints. `val_rollout_steps=100` still means 100 stored-frame intervals:
50 model calls reach frame 103. Differentiable multi-step training also supports
this prediction stride. The separate `split.frame_stride` setting spaces the
input history itself and is kept at 1.

```bash
gnn-bench --threads 4 --root results/noisy-lj-predict2/runs train \
  configs/noisy_lj_stride/predict2_original.json --model gns mlp --seeds 0 1 2
```

Earlier exploratory runs changed the initial history, including the MLP score
of 0.733 observed through frame 15. Those scores are **excluded from benchmark
claims**. Their [archived measurements](configs/noisy_lj_stride/comparison.json)
mark which initial histories match the requirement. Experimental runs remain
under their own results roots; the website's original results are unchanged.

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
The plot initially shows the top three models by mean test R² at the selected
horizon. Click model names to toggle other lines, or use **Show all models** /
**Show top 3**. Rankings and the Runs table include every model.
The separate **One-step / MST** toggle selects the training objective for the
plot, rankings, table and CSV. The protocol line shows the rollout curriculum
and training-frame count. Empty combinations say that results are pending;
historical results without a training-mode field count as one-step runs.
Filters update the URL, so copying the address shares the current dataset,
Normal/OOD split, One-step/MST objective, horizon, seeds, selected models and
display settings. For example, `?dataset=noisy_lj&mode=ood&training=mst` opens
the Noisy LJ OOD results with MST selected. Model lines use distinct markers
and dash patterns as well as colors; labels and plot text stay high contrast.

The R² and position-MSE plots share one filter bar, model legend and horizon
slider. Both show the same selected models; the default top three are ranked by
R² at the selected horizon. Position MSE is recorded every ten rollout steps,
averaging squared position errors over node coordinates and then over test
networks. Lower MSE is better. The table and CSV include position MSE and
relative MSE at the selected horizon; relative MSE divides the mean position
error by the mean frozen-position baseline at that same step. Plot bands show
sample standard deviation across training seeds.

The data split stays fixed (`split.seed`); `train.seed` controls model
initialisation and training-window sampling. Every run has its own saved config,
weights, and metrics. Repeating the command skips completed runs. `--force`
reruns them.

`report --seeds` groups runs with identical configs except `train.seed`, identical
resolved systems, and the same evaluated split. It shows mean ± sample standard
deviation and the finite count for each metric. A single seed has no standard
deviation. Plain `report` lists individual runs. The short settings ID distinguishes
configurations; full settings are stored with each run.

For learning curves with different splits per seed, use
[`configs/learning_curve`](configs/learning_curve/README.md). Each seed selects
100 training-pool networks, 70 validation networks and 70 test networks from the
full dataset roster. Training sizes 10, 20, …, 100 use nested prefixes of that
pool; validation and test stay fixed within a seed. Seeds 0, 1 and 2 change both
membership and training randomness. The saved `splits.json` files list exact
membership. These results have a separate summary because the ordinary
`report --seeds` requires identical membership.

The website's **Learning curves** view plots test R² and position MSE against
training-network count at a selected rollout step. The training-size selector
chooses the ranking, run table and exact split to inspect. Its URL preserves
the selected view and filters; JSON and CSV downloads keep the raw scores.

```bash
gnn-bench learning-curve-config configs/networks.json \
  --dataset noisy_lj node_optimized stiff_optimized \
  --seeds 0 1 2 --out configs/learning_curve
```

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
