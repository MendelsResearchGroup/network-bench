# Data and physics reference

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

Three more 2D spring-network sets from the inverse-design project share the
`dePablo_random` layout: `node_optimized` (node positions optimised for a target
Poisson's ratio), `stiff_optimized` (per-bond stiffnesses optimised) and
`noisy_dump200` (stiffnesses 1/l0 with noise). They differ in what the stiffness
column holds -- exactly 1/l0, optimised and unrelated to l0, or 1/l0 with 13%
noise -- so every harmonic set reads it from the column and none reconstructs it
from the rest length. `gnn-bench datasets` has the details.

`node_optimized` and `stiff_optimized` are registered against the mini release on
Zenodo ([doi:10.5281/zenodo.20181262](https://doi.org/10.5281/zenodo.20181262)):
extract `data_mini.tar.gz` with `tar xzf data_mini.tar.gz -C $GNN_BENCH_DATA_ROOT
--strip-components=2` and its `node_optimized/` (381 systems) and
`stiff_optimized/` (288) land where the registry looks. The legacy `network.Box`
pickles load as they are; no conversion step. The full `data.tar.gz` (1998 and
1085 systems) uses the same layout, and since a run's cache hash covers its
resolved system list, results on the two never collide.

| median of 60 systems | `node_optimized` | `stiff_optimized` | `noisy_dump200` | `cold_wca_fixyz_n1024` |
|---|---|---|---|---|
| `floor_model` | 0.729 | 0.734 | 0.134 | 0.476 |
| `r2_ceiling` | 0.951 | 0.934 | 0.968 | 0.998 |
| `affine_target_share` | 0.653 | 0.674 | 0.510 | 0.000 |
| `noise_over_signal` | 0.221 | 0.257 | 0.178 | 0.044 |

(`gnn-bench difficulty <key> --systems 60 --history 3 --count 20 --max-frames 28`,
with `--manifest ""` for the 2D sets and `clean` for the Kremer-Grest one. The
`node_optimized` and `stiff_optimized` columns are measured on the mini release.)

`notebooks/dataset_metrics.ipynb` puts every measured dataset side by side --
tables, one distribution panel per metric, the floor against its ceiling -- from
the cached `difficulty.json` files alone. Open it in VS Code with this project's
`.venv` as the kernel, or run
`uv run --extra analysis --group dev --with jupyterlab jupyter lab`.

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

The data is never in the repository. The default data root is
`/rg/mendels_prj/s.sergey/data_bench`. The registry reads `node_optimized/`,
`stiff_optimized/`, and `data_LJ_noisy_eps0.01_sigma1.0_cutoff1.122/` directly
there. Register a dataset in `data/registry.py` and set `GNN_BENCH_DATA_ROOT`
when using another root with the same directory names.

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

Subclass `SimulatorModel` and implement one method:

```python
import torch
from gnn_physics_benchmark.interfaces import SimulatorModel
from gnn_physics_benchmark.models import register

class MyModel(SimulatorModel):
    def __init__(self, spec, target_scale, *, hidden_size: int = 64):
        super().__init__(spec, target_scale, hidden_size=hidden_size)   # -> self.hyperparameters
        self.net = torch.nn.Linear(spec.node_feature_width, spec.dim)

    def predict_acceleration(self, graph):
        return self.target_scale.inverse(self.net(graph.x))           # [N, dim]

register("my_model", MyModel)
```

`predict_acceleration` gets the input graph and returns every node's
acceleration. The base class's `forward` turns it into the **next frame in the
raw dataset schema** (`x_{t+1} = x_t + (x_t - x_{t-1}) + a`), so a rollout feeds
it straight back in. The acceleration is stashed on that frame, so the loss never
recovers it by differencing float32 positions.

`self.spec` is the run's `InputGraphSpec`: `node_feature_width`, `edge_width` and
`dim` size the layers. `self.target_scale` is the acceleration normaliser the
benchmark fitted for this run: decode in standardised space and apply `inverse`.
That is what makes one model's loss comparable with another's. A model with its
own input `Normalizer`s returns them from `input_normalizers()`, so training
freezes them at `freeze_input_norm_epoch`.

The contract is checked three times:
- by a type checker, since the package ships type information: a class that is
  not a `SimulatorModel`, or one without `predict_acceleration`, is flagged;
- by Python, which refuses to create a subclass without `predict_acceleration`,
  and by `register`, which refuses anything that is not a `SimulatorModel`;
- by every run, on the model's first prediction, before training starts.

If your model's idea *is* a different target normalisation, define
`normalize_target(acceleration, *, accumulate)` and the benchmark will score you
through it instead — deliberately off the shared scale.

## Baselines

| key | what it is |
|---|---|
| `frozen` | nothing moves. It **is** the `relative_mse` denominator, so it must score exactly 1.0 — the harness's own self-test |
| `linear_floor` | one weight per history step, shared across axes: the best linear predictor from the velocity history. A model that does not beat this is reproducing velocity persistence, not learning the material |
| `mlp` | the same node features, no message passing. The control for whether the graph is doing anything |
| `tiny_mlp` | 6 -> 4 -> 2 over the velocity history alone, 46 parameters |
| `edge_mlp` | the velocity history plus the sum of the node's own bond encodings: one hop, no message passing |
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
- `poisson_r2_10`, `poisson_r2_20`, ... — R² across systems of Poisson's ratio
  10, 20, ... rollout steps past the seed. It is fitted to the node positions (the
  affine transverse stretch), because a rollout does not predict the box. NaN
  wherever the transverse box is clamped.
- `ratio_r2`, `sxx_slope_r2`, `stress_rel_mse` — the stress response, from the
  force field's own virial on a neighbour list rebuilt at the full cutoff. Off
  unless `train.stress_metrics` is set, since it costs a virial per sampled frame.

The two are complementary, and which one carries Poisson's ratio depends on how
the dataset was generated. With the transverse box **clamped** the stress ratio is
`nu/(1-nu)`, so `ratio_r2` scores it and `poisson_r2_<k>` is NaN. With the transverse
box **relaxed to zero transverse stress** the transverse stress is zero by
construction, so the ratio says nothing and `poisson_r2_<k>` is the real measurement.

Three R² conventions coexist and are deliberately kept distinct: uncentred for
the acceleration floors (the target's mean is physically zero), centred for
per-system scalars, and `relative_mse` which is a ratio of means, not an R² at all.

**Choosing the epoch.** By default the test scores the last epoch. With
`train.select_by` set to a validation metric, e.g. `"poisson_r2_100"`, it scores
the checkpoint of the validation epoch that did best on it instead; the test split
plays no part in the choice. `metrics.json` then records `selected_epoch` and the
validation score it was chosen on.

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
