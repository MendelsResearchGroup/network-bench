# gnn-physics-benchmark

Compare autoregressive simulator models on the same molecular-dynamics data, the
same input graph, the same loss and the same metrics.

Write a model, register it under a key, and it gets trained and scored exactly
like every other model in the table. Results are cached under
`(dataset key, model key, hyperparameters, split)`, so two numbers in a results
table are comparable by construction rather than by care.

```
pip install git+https://github.com/<org>/gnn_physics_benchmark
export GNN_BENCH_DATA_ROOT=/path/to/datasets

gnn-bench datasets                       # what is registered
gnn-bench schema                         # what the data holds, what a model is handed
gnn-bench difficulty cold_wca_fixyz_n1024
gnn-bench train configs/smoke_gns.json
gnn-bench report cold_wca_fixyz_n1024
```

## The data

A dataset is a directory of `.pt` files, one per simulated system, each a plain
`list[Data]` in time order. Datasets differ in what they store, so the layout is
not a constant: each one registers a `FrameSchema`, and everything downstream
reads that rather than a hard-coded dimension or column order. `gnn-bench schema
<dataset>` prints the full table.

Two datasets ship registered:

| | `cold_wca_fixyz_n1024` | `dePablo_random` |
|---|---|---|
| what | Kremer-Grest polymer melts, uniaxial compression | 2D random spring networks, uniaxial compression |
| `x` | `[N, 3]` | `[N, 2]` |
| `edge_attr` | `dx, dy, dz, r, fene_k` | `dx, dy, r, stiffness, rest_length` |
| bonds | FENE, one constant | harmonic, per-bond stiffness and rest length |
| pair interaction | WCA | none |
| transverse box | clamped | relaxed to zero transverse stress |
| dynamics | inertial | overdamped |

Nothing derived is stored: velocities, forces, stress, strain and Poisson's ratio
are all computed.

### Declaring a layout

```python
"dePablo_random": DatasetEntry(
    schema=FrameSchema(
        edge_columns=("dx", "dy", "r", "stiffness", "rest_length"),
        derived_columns=("rest_length",),   # computed at load from frame 0
        stored_fields=(),                   # pos, box_tensor, time, ids all derived
        pickle_module="network",            # which module the Box was pickled from
    ),
    potential={"kind": "harmonic", "stiffness_from": "column",
               "stiffness_column": "stiffness", "rest_length_from": "first_frame"},
    driven_axis=0, free_axes=(1,), ...
)
```

The benchmark needs only two facts about `edge_attr`: which columns are
**geometry**, recomputed whenever positions or the box move, and which are
**parameters**, carried through unchanged. Naming the columns gives both, and the
spatial dimension falls out of how many vector components there are — `dx, dy` is
two-dimensional, `dx, dy, dz` is three. Nothing anywhere holds a `DIM` constant.

Everything a dataset does not store is derived on load: `pos` from `x`,
`box_tensor` from `box`, `time` from the frame index, identity fields from
`arange` and `ones`. `derived_columns` handles per-bond quantities that need the
whole trajectory — `rest_length` is read off the undeformed first frame.

The data is never in the repository. Register a dataset in `data/registry.py` and
point `GNN_BENCH_DATA_ROOT` at where it lives.

## The input graph

A model never sees a raw frame. The **run** — not the model — decides what the
model is fed, so two models on the same dataset key see identical input:

```python
InputGraphSpec(
    history=8,                    # velocity steps; needs history + 1 frames
    velocity="total",             # "total" | "residual" | "affine"
    fractional_coordinates=False,
    node_force=False,
    edges="bond+pair",            # bonds symmetrised, plus an LJ neighbour list
    graph_cutoff=None,            # None = the dataset's own force-field cutoff
)
```

The graph carries the velocity history as `x` (`[N, 3h]`, most recent step first,
step-major), `pos` and `prev_pos`, `edge_index`/`edge_attr`, the box, and whatever
extras the spec asked for. Nothing it did not ask for.

Two things about it are load-bearing:

- **Velocities, not positions.** The target is an acceleration of order 3e-6
  sigma while positions are of order 3 sigma. Feeding positions would make the
  network find a difference of five orders in float32.
- **`graph_cutoff` may never exceed the force-field cutoff.** An edge past the
  cutoff carries no interaction, so a wider graph adds edges that correspond to
  nothing. Pulling it *in* is a real tradeoff and is what the knob is for.

`dim` and `edge_width` are filled in from the dataset when a run is resolved, so a
config need not repeat them; the resolved values are recorded in the saved config.

`node_force` is off by default and how badly it behaves depends on the force law.
Harmonic forces are linear in displacement and stay bounded; Lennard-Jones goes as
`r^-13`, so two particles drifting close during a rollout produce an enormous
input and the rollout runs away. Measured on a small Kremer-Grest run it left the
training loss essentially unchanged and diverged the rollout after 6 of 15 steps.

## Writing a model

```python
import torch
from gnn_physics_benchmark.interfaces import InputGraphSpec
from gnn_physics_benchmark.interfaces.model import integrate_acceleration
from gnn_physics_benchmark.models import register

class MyModel(torch.nn.Module):
    def __init__(self, spec: InputGraphSpec, target_scale, *, hidden_size: int = 64):
        super().__init__()
        self.hyperparameters = {"hidden_size": hidden_size}   # goes into the cache key
        self.target_scale = target_scale
        self.net = torch.nn.Linear(spec.node_feature_width, 3)

    def forward(self, graph):
        acceleration = self.target_scale.inverse(self.net(graph.x))
        return integrate_acceleration(graph, acceleration)

register("my_model", MyModel)
```

That is the whole contract. `model(graph) -> Data` returns the **next frame in
the raw dataset schema**, so a rollout feeds it straight back in with no special
casing — `validate_raw_frame` will confirm it.

`target_scale` is the acceleration normaliser the benchmark fitted for this run.
Your decoder works in standardised space and `inverse` puts the result back into
physical units; that is what makes one model's loss comparable with another's.
`integrate_acceleration` applies `x_{t+1} = x_t + (x_t - x_{t-1}) + a` and stashes
the acceleration on the frame, so the loss never has to recover it by
differencing float32 positions.

If your model's idea *is* a different target normalisation, define
`normalize_target(acceleration, *, accumulate)` and the benchmark will score you
through it instead — deliberately off the shared scale.

## Baselines

| key | what it is |
|---|---|
| `frozen` | nothing moves. It **is** the `relative_mse` denominator, so it must score exactly 1.0 — the harness's own self-test |
| `linear_floor` | one weight per history step, shared across axes: the best linear predictor from the velocity history. A model that does not beat this is reproducing velocity persistence, not learning the material |
| `mlp` | the same node features, no message passing. The control for whether the graph is doing anything |
| `gns` | encode / message-passing / decode, with an axis-shared node encoder |

## Metrics

**Dataset difficulty** (`gnn-bench difficulty`) needs no model. Read `floor_model`
first — the linear floor matched to what a model is actually fed — then
`r2_ceiling` beside it, because a floor only means something relative to how much
of the target is signal at all. Positions are stored in float32 at order 3 sigma
while the target is order 3e-6, so the stored precision puts real noise into the
target; `r2_ceiling` is the best R² any predictor could reach.

**Model performance** (`gnn-bench report`):

- `position_mse` — raw position MSE at the last rollout frame reached.
- `frozen_mse` — the same error for beads frozen at the last seed frame.
- `relative_mse` — their ratio. 1.0 is no better than standing still. This is the
  number that survives a change of system size or dimensionality; the raw MSE is
  kept beside it so the ratio can be audited rather than trusted.
- `poisson_r2` — from the box alone, so it is NaN wherever the transverse box is
  clamped.
- `ratio_r2`, `sxx_slope_r2`, `stress_rel_mse` — the stress response, from the
  force field's own virial on a neighbour list rebuilt at the full cutoff.

The two are complementary, and which one carries Poisson's ratio depends on how
the dataset was generated. With the transverse box **clamped** the stress ratio is
`nu/(1-nu)`, so `ratio_r2` scores it and `poisson_r2` is NaN. With the transverse
box **relaxed to zero transverse stress** the transverse stress is zero by
construction, so the ratio says nothing and `poisson_r2` is the real measurement.

Three R² conventions coexist and are deliberately kept distinct: uncentred for
the acceleration floors (the target's mean is physically zero), centred for
per-system scalars, and `relative_mse` which is a ratio of means, not an R² at all.

## Results cache

```
results/<dataset>/<model>/<readable_name>_<hash8>/
    config.json  split.json  metrics.json  history.json  checkpoints/  [failed.txt]
```

`readable_name` names only what differs from the defaults, so an ordinary run has
a short name and an unusual one announces what is unusual. `hash8` is taken over
the whole config *including the resolved list of systems in each split* — so
regenerating a manifest misses the cache rather than silently comparing unlike
runs. A failure writes `failed.txt` and no `metrics.json`, so a rerun retries it.

## Provenance

Extracted from the `KG_chains` research repository, trimmed to what a benchmark
needs. The physics is validated against LAMMPS: on `rubber_0000` frame 0 the
virial stress agrees to five significant figures (Pxx 6.830349 vs 6.8303184) and
the potential energy to seven (6779.0783 vs 6779.0779); the stress-strain slopes
over the first hundred frames agree to four (C11 −104.26 vs −104.32), giving
nu = 0.4451 from the ratio.

Not included in this version: the Langevin-piston barostat (so a rollout's
transverse box is frozen), input-noise injection, the auxiliary edge-decoder
head, and hyperparameter search.
