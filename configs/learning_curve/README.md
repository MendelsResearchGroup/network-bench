# Learning curves with seed-specific splits

Each dataset is evaluated at training sizes **10, 20, …, 100**, with **70
validation networks and 70 test networks**. Seeds **0, 1, 2** change both split
membership and training randomness. The experiment reads the shared
`/rg/mendels_prj/s.sergey/data_bench` data directly.

For each dataset and seed, shuffle the sorted full file roster with a local
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
That is **630 runs**. The elastic-response model is outside this experiment.

The existing one-step training recipe is used: 16 head windows, history 3,
learning rate 0.001, maximum 40 epochs, validation every two epochs, and early
stopping after five checks without improvement. Validation Poisson R² at 100
steps selects each checkpoint; test data never selects checkpoints. Every test
rollout observes exactly **frames 0–3** and predicts **frames 4–103**. These are
normal-distribution splits; MST and OOD are separate existing protocols.

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

`jobs.json` lists all 630 cells. The CPU-only PBS script runs an array index
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
The training-size selector chooses the ranking, table and exact membership
shown below; it marks that size with a dotted line without hiding other sizes.
Top three models are ranked at that size and horizon. Negative R² is drawn at
zero, while hover, ranking, table and CSV retain the raw scores.

The seed filter changes split and training seed together. With all seeds shown,
the exact-split panel lets you inspect one seed explicitly. Normal/OOD and
One-step/MST filters appear in both views. This study ran normal one-step
training; other learning-curve selections show "not run yet" with empty plots,
rankings and tables. URL parameters preserve these filters along with the view,
dataset, horizon, training size,
seeds, models, lines and bands. Downloaded JSON contains all 630 individual
runs with their per-step metrics and all 90 exact splits.

Export the completed study independently of the original `docs/results.json`:

```bash
.venv/bin/python configs/learning_curve/export.py
```

This writes `docs/learning-curves.json`. Scores are averaged per seed in the
browser; test populations are never pooled into a single R².
