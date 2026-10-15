# Network Bench

Compare neural models on autoregressive network dynamics with reproducible
training splits and held-out evaluation.

**[Interactive results](https://mendelsresearchgroup.github.io/network-bench/)** ·
[Published sweep](configs/sweep_v2/README.md) ·
[Configuration guide](configs/README.md) ·
[Data and physics reference](docs/reference.md)

## Published benchmark

The website uses the completed **sweep-v2** recipe: 4,200 trained runs and
600 untrained controls. It compares seven trainable models on Node optimized,
Stiffness optimized, and Noisy LJ, using training sizes 10–100 and seeds 0–2.
Each split has 70 validation and 70 test networks. Rollouts observe frames
0–3 and predict frames 4–103. No ITPO or simulator bootstrap is used.

| Setting | Published recipe |
|---|---|
| GNS | Two message-passing layers, width 64 |
| One-step training | Up to 40 epochs; early stopping after five validation checks |
| Multi-step training (MST) | Detached feedback; 100 epochs; no early stopping |
| MST curriculum | 1 / 2 / 3 / 5 / 8 / 10 steps starting at epochs 1 / 11 / 21 / 31 / 41 / 51 |
| MST checkpoint selection | Validation Poisson R² at 100 steps, starting at epoch 51 |
| Barostat | Separate off/on runs; constants calibrated on the first 10 training trajectories only |

Normal splits use nested random training subsets. OOD reserves the highest
100 networks by the saved Poisson-ratio ranking: training size N uses the
highest N, and validation/test come from outside that pool. Exact membership
is retained in the [learning-curve configs](configs/learning_curve/README.md).

Noisy LJ has **no barostat-on results** because its force field and rest lengths
have not been verified. For the optimized datasets, the Barostat sidebar
checkbox selects the corresponding runs in both plots and learning curves.
With barostat on, predicted Poisson ratio uses box strain; with it off, it
uses the side-node estimator. Ground truth uses box strain in both cases.
Position MSE is also reported. These estimator differences matter when
interpreting the toggle.

Filters are shareable through the URL. The plots initially show the top four
models; other models can be enabled in the sidebar. R² and position MSE share
the filters. Seed bands show sample standard deviation, and JSON downloads
retain negative R² values.

## Run a comparison

```bash
pip install -e .
export GNN_BENCH_DATA_ROOT=/path/to/data_bench

gnn-bench train configs/networks.json --model gns mlp --seeds 0 1 2
gnn-bench report node_optimized --seeds
```

`configs/networks.json` is a small standalone example, **not the full published
sweep**. Use [sweep-v2](configs/sweep_v2/README.md) to reproduce the website's
training protocol. The [configuration guide](configs/README.md) distinguishes
current, historical, and exploratory recipes.

The default data root is `/rg/mendels_prj/s.sergey/data_bench`, containing
`node_optimized/`, `stiff_optimized/`, and
`data_LJ_noisy_eps0.01_sigma1.0_cutoff1.122/`. Trajectories remain outside Git.
Changing the available file roster can change newly generated splits; saved
explicit memberships preserve existing comparisons.

Model overrides use `--set`, for example
`--model gns --set model_hyperparameters.hidden_size=64`.
`gnn-bench datasets`, `models`, and `schema` describe the available interfaces.
The default device is CPU. For a single CPU PBS job:

```bash
qsub -v CONFIG=configs/networks.json,MODEL=gns,SET=train.seed=0 train.pbs
```

Training jobs save runs without changing the website. Export explicitly with
`gnn-bench export --out /path/to/results.json`; use the sweep-specific export
for the published two-view website. Completed runs are cached; `--force` reruns them.

## Repository layout

| Path | Purpose |
|---|---|
| `src/gnn_physics_benchmark/` | Importable data/model interfaces, training and evaluation |
| `configs/sweep_v2/` | Published training recipe and job manifests |
| `configs/learning_curve/` | Exact split inputs and historical learning-curve recipes |
| `configs/` | Configuration index, examples and historical experiments |
| `docs/` | Static website and published result JSON |
| `scripts/build_website.py` | Version website assets for deployment |
| `analysis/`, `notebooks/` | Optional dataset analysis, outside the training package |
| `results/` | Local runs, checkpoints and logs; ignored by Git |

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

## Results and website

Each run saves its configuration, exact split, training history, metrics, and
checkpoints under `results/`. `gnn-bench evaluate <run-directory>` evaluates
the selected checkpoint on its saved split. `report --seeds` aggregates runs
with identical membership; the learning-curve export handles differing splits.

The website is static; no backend is needed. Preview it with:

```bash
python -m http.server --directory docs 8000
```

GitHub Actions publishes changes to website files on `main`. Each deployment
versions assets and result URLs by commit; reload an open tab to get the new
version. Build locally with:

```bash
python scripts/build_website.py /tmp/network-bench-site local-preview
```

For the full sweep, `configs/sweep_v2/export.py` checks completion and exports
both plot views together. Preserve the exact splits and saved run metadata
when updating published results.
