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
The repository does not store trajectory copies. The OOD split reads each
network's Poisson's ratio from the Zenodo registry CSVs under the same root:
`data_registry_mini.csv` (node and stiffness optimized) and
`data_LJ_noisy_eps0.01_sigma1.0_cutoff1.122/data_registry.csv` (Noisy LJ).

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

`gns` defaults to hidden size 64 and two message-passing layers, as in the
GNNInverseDesign reference. The `edge_mlp` and `mlp` defaults use width 128 and depth 4.
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
(104 frames total). The split contains 100 training, 50 validation and 70 test
trajectories; validation rollouts select the checkpoint. A run with fewer than
**100 training networks** prints a warning: that is too few to support a comparison.

Training runs the full **150 epochs**. Every two epochs the checkpoint is saved
and rolled out on the validation networks; afterwards the checkpoint with the
best `train.select_by` (the benchmark uses `poisson_r2_100`) is restored and
tested. Early stopping is off; `train.early_stopping_patience` turns it back on.

### Rollout box and Poisson's ratio

The box is not predicted. In every rollout, during validation, test and
multi-step training, the driven edge Lx keeps the per-frame increment of the last
two seed frames, and the free edge Ly follows a Langevin-piston barostat pushed by
the virial pressure of the predicted positions, as LAMMPS' `fix nph` did. Bond
vectors are recomputed in each new box. Poisson's ratio is read off the box, from
the last seed frame to each scored step, for prediction and ground truth alike.

The piston constants are registered per dataset. Before training, every run
drives the barostat with ground-truth positions on ten training trajectories and
compares Ly with the true box. A relative error above 0.1 stops the run. The
registered constants reach 0.02–0.05 and recover the true Poisson's ratio with
R² ≥ 0.997. A dataset with no constants says so and has them fitted at the start
of the run; register the printed values. Check or refit a dataset on its own:

```bash
gnn-bench barostat node_optimized        # check the registered constants
gnn-bench barostat dePablo_random --fit  # search for new ones
```

Noisy LJ declares no force field, so it has no pressure to drive the barostat
and cannot be rolled out until its potential is declared.

### Multi-step training

Multi-step training (MST) feeds predictions back into the model and averages the
loss over every rollout step. Predictions are detached before they are fed back,
as in the reference; `--set train.detach_rollout=false` backpropagates through
the whole rollout instead. Lx and Ly move exactly as in a rollout.
`configs/mst/normal.json` uses the same network split as the normal benchmark;
`configs/mst/*_ood.json` keep the exact OOD memberships. The reference curriculum
uses 1 step in epochs 1–10, 2 in 11–20, 3 in 21–30, 5 in 31–40, 8 in 41–50 and 10
from epoch 51. With 16 starting windows and four input frames, MST consumes the
first **29 frames** per training trajectory. Validation and test still predict
100 steps. The frozen-position baseline has no learned dynamics and is included
for reference in both training modes.

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
force and stress analysis is unavailable, and so, for now, are rollouts (see above).

### Noisy LJ: two-frame predictions

These results predate the barostat rollout: the transverse box was frozen and
Poisson's ratio was fitted to node displacements.

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
  --dataset node_optimized --out configs/ood/node_optimized.json
gnn-bench --threads 4 train configs/ood/node_optimized.json \
  --model gns mlp edge_mlp edge_mlp_delta edge_mlp_attention linear_floor frozen --seeds 0 1 2
```

Networks are ranked by the Poisson's ratio the dataset's registry CSV records.
The highest 30% (rounded up) forms the training pool and the lower 70% the test
pool; each is shuffled with data-split seed 42. Training takes the first 100
networks of the upper pool, and test at most 100 of the lower one. Validation
spans the full range of ratios: 15 of its 50 networks come from what the upper
pool has left, as many as there are, and the rest from the lower pool's
remainder. The remaining networks are unused. Equal ratios are ordered by
network ID. `ood-config --train-networks` and `--val-networks` change the counts.

| Dataset | Training | Validation (upper / lower) | OOD test | Cutoff ν |
|---|---:|---:|---:|---:|
| Node optimized | 100 | 50 (15 / 35) | 100 | 0.129 |
| Stiffness optimized | 87 | 50 (0 / 50) | 100 | 0.134 |
| Noisy LJ | 100 | 50 (15 / 35) | 100 | 0.469 |

The stiffness-optimized set has only 288 networks, so its upper pool holds 87:
all of them train, the run warns about the count, and validation has no
high-ratio networks. The full Zenodo set (1085 networks) would fill both.

The configs store every network assignment explicitly; accompanying
`*.ranking.json` files record every ratio and the cutoff. Training still uses
the first 20 frames. Test trajectories do not enter training, normalization, or
checkpoint selection. Ground-truth test responses are used to define the OOD split.

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
Plotly is loaded from its versioned CDN; no backend is needed. Preview with
`python -m http.server --directory docs 8000`. GitHub Actions publishes the
website when its files change on `main`. The small static build stamps local
CSS, JavaScript, images and results URLs with the deployed commit SHA. Results
JSON also revalidates on reload, so new deployments use fresh cache entries.
Open tabs need a reload to receive an update. Build a deployment locally with
`python scripts/build_website.py /tmp/network-bench-site local-preview`.

Dataset layouts, force fields, difficulty measures, and scientific caveats are
in the [data and physics reference](docs/reference.md).
