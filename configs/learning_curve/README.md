# Learning curves with seed-specific splits

Each dataset is evaluated at training sizes **10, 20, …, 100**, with **70
validation networks and 70 test networks**. Seeds **0, 1, 2** change both split
membership and training randomness. The experiment reads the shared
`/rg/mendels_prj/s.sergey/data_bench` data directly.

For Normal, for each dataset and seed, shuffle the sorted full file roster with a local
PyTorch generator. Select 100 training-pool networks, 70 validation networks,
and 70 test networks. Training size N uses the **first N of that seed's training
pool**. Therefore the training subsets are nested, and validation/test membership
stays fixed across training sizes and models within a seed. Other seeds use
other splits; a network can have different roles in different runs, but never
more than one role within the same split.

The exact full roster, its hash, the three training pools, validation/test IDs,
and unused IDs are in each dataset's `splits.json`. Individual configs contain
explicit membership and record **both split seed and training seed**.

| Dataset | Available networks | Unused at training size 100 |
|---|---:|---:|
| Noisy LJ | 1348 | 1108 |
| Node optimized | 381 | 141 |
| Stiffness optimized | 288 | 48 |

Seven trainable models participate: `gns`, `mlp`, `tiny_mlp`, `edge_mlp`,
`edge_mlp_delta`, `edge_mlp_attention`, and the linear baseline `linear_floor`.
All four variants are complete: **630 runs each**, for **2,520 total**, with
no failed runs or divergent test trajectories. The elastic-response model
is outside this experiment.

The existing one-step training recipe is used: 16 head windows, history 3,
learning rate 0.001, maximum 40 epochs, validation every two epochs, and early
stopping after five checks without improvement. Validation Poisson R² at 100
steps selects each checkpoint; test data never selects checkpoints. Every test
rollout observes exactly **frames 0–3** and predicts **frames 4–103**. These are
normal-distribution splits. The new variants follow the protocols below.

## MST and highest-N OOD

Normal MST keeps every original normal split exactly. Its curriculum predicts
1 step in epochs 1–2, 2 in 3–4, 3 in 5–6, 5 in 7–8, and 10 from epoch 9,
with gradients through predictions. It consumes 29 training frames per
trajectory at the maximum curriculum horizon. Epoch cap, early stopping,
validation selection and test rollout seeds remain the same.

OOD learning curves use **highest counts rather than the earlier highest-30%
rule**. Rank the full current roster by the ground-truth position-based ratio
from frame 3 to frame 103, descending, with network ID breaking ties. Reserve
ranks 1–100 for training. Size N uses ranks 1–N, identical across seeds. The
unselected training-pool remainder is unused, never validation or test.

For each seed, shuffle ranks 101 onward with a local PyTorch generator; the
first 70 are validation, the next 70 test, and the rest unused. These held-out
IDs stay fixed across sizes, models and One-step/MST within a seed. Validation
and test therefore have lower ratios than the reserved top-100 training pool.
Both held-out membership and training randomness vary across seeds, while
top-N training membership does not. Checkpoint selection uses lower-ratio
validation R² at 100 steps. This differs from the old fixed-size OOD protocol,
where validation also came from the upper 30%.

Fresh full-roster rankings are in `ood_rankings/<dataset>.json`. Exact split
audits and configs are under `normal_mst/`, `ood_one_step/`, and
`ood_multi_step/`. All four variants have 70 validation and 70 test networks
per seed, but their held-out populations differ between Normal and OOD.

```bash
qsub -J '0-2' configs/learning_curve/rank.pbs
# After the ranking array finishes:
.venv/bin/python configs/learning_curve/prepare_variants.py
```

The preparation script preserves existing `jobs.json` indices and appends
missing variants. It writes the job list atomically so queued workers can read
it while the OOD cells are added. Old indices 0–629 stay the completed normal
one-step sweep; 630–1259 are normal MST, 1260–1889 OOD one-step, and 1890–2519
OOD MST. `report.py` groups by distribution and training objective as well as
dataset, model and training size, so variants are never averaged together.

Generate the configs from the repository root:

```bash
.venv/bin/gnn-bench learning-curve-config configs/networks.json \
  --dataset noisy_lj node_optimized stiff_optimized \
  --seeds 0 1 2 --out configs/learning_curve
```

Run one cell normally, without overriding its already specified training seed:

```bash
.venv/bin/gnn-bench --root results/learning-curve/runs --threads 4 train \
  configs/learning_curve/noisy_lj/seed-0_train-010.json --model gns
```

`jobs.json` lists the cells for all prepared variants. The CPU-only PBS script runs an array index
through `run.py`; results live in `results/learning-curve/runs/`, with individual
completion records and logs beside them. PBS concurrency is capped when the
array is submitted. `benchmark.run` reuses a completed cell when its config and
explicit membership hash match.

Both `mendels_q` and `mendels_comb_q` can run this CPU workload. Override the
script's default queue with `qsub -q mendels_comb_q`; the array indices select
the same entries in `jobs.json`. For example, after the first three checks:

```bash
qsub -q mendels_q -J '3-315%64' configs/learning_curve/train.pbs
qsub -q mendels_comb_q -J '316-629%64' configs/learning_curve/train.pbs
```

```bash
.venv/bin/python configs/learning_curve/report.py
```

This writes `results/learning-curve/summary.json` and `runs.csv`. Each summary
reports its completed seeds, mean, and sample standard deviation. Here the
variation includes **both split selection and training randomness**; it should
not be interpreted as initialization-only uncertainty. Different split seeds
can have overlapping held-out memberships, so the three repeats are not three
disjoint test populations. Partial summaries include completion and failure
counts. The standard benchmark seed summary continues to require identical
network membership.

## Website view

Select **Learning curves** on the results page. Both R² and position MSE use
training-network count on the x-axis and score the selected rollout horizon.
Dataset, models, seed, mean/individual lines and SD bands control both plots.
Click a plotted training-size point to choose the model summary and exact
membership shown below; a dotted line marks that size without hiding other sizes.
The top four models are shown by default, ranked at that size and horizon.
Negative R² is drawn at zero, while hover, model summary and CSV retain raw scores.
Filter changes animate the R² chart. Position MSE animates only when the
"Score at step" slider changes; other filters update it immediately.
The R² axis stays fixed at 0–1. Reloading revalidates the results JSON. Studio is the main layout, with light mode
by default; the theme toggle also offers dark mode.

The seed filter changes split and training seed together. With all seeds shown,
the exact-split panel lets you inspect one seed explicitly. Normal/OOD and
One-step/MST filters appear in both views. Selections without published runs
show "not run yet" with empty plots, rankings and tables. URL parameters
preserve these filters along with the view,
dataset, horizon, training size,
seeds, models, lines and bands. Downloaded JSON contains individual published
runs with their per-step metrics and exact splits. Distribution and training
objective have separate comparison IDs, so their scores are never mixed.

Export the completed study independently of the original `docs/results.json`:

```bash
.venv/bin/python configs/learning_curve/export.py
```

This writes `docs/learning-curves.json`. Scores are averaged per seed in the
browser; test populations are never pooled into a single R².
