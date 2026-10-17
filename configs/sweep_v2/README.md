# Detached MST and barostat sweep

Fresh training on the original `learning_curve` memberships: seven trainable
models, Normal/OOD, training sizes 10–100 by 10, seeds 0–4, 70 validation and 70
test networks. Real frames 0–3 seed every 100-step rollout. No ITPO or simulator
bootstrap. Shared data stays at `/rg/mendels_prj/s.sergey/data_bench`.

GNS explicitly uses 2 message-passing layers. MST detaches feedback, trains 100
epochs without early stopping, and grows its horizon at zero-based epochs
0/10/20/30/40/50 to 1/2/3/5/8/10 steps. Checkpoint selection starts at numbered
epoch 51 (the 10-step stage), uses validation Poisson R²@100, and retains the
final epoch 100 checkpoint. One-step keeps cap 40 and patience 5.

The barostat is from PR #2 at `5adb396f08c59c0e8e00c0a990f10f36cfdff074`.
Its two constants are calibrated by the PR coarse/fine grid search using the
first 10 **training** trajectories only. Those IDs are identical across the
nested sizes for a dataset/distribution/seed, so frozen parameters are reused
across sizes, models and objectives. Evaluation never fits missing parameters.
MST uses the updated box to construct subsequent inputs; one-step uses it in
validation and evaluation. Calibration details are stored separately.

Reported Poisson R² compares to ground-truth box strain. With the barostat,
predictions use box strain; without it, predictions use the group's unchanged
side-node estimator at `734c6b81b5cddc539d1fad2e465e333f45ee5a90`. Thus the
toggle changes both the physical controller and the prediction estimator;
position MSE against GT is also shown. Negative R² is retained in JSON.

`prepare.py` creates 7,000 training cells and 1,000 untrained frozen/zero-acceleration
controls. Another 1,400 Noisy LJ barostat training cells are explicitly blocked:
that archived dataset has no verified force field/rest lengths. No spring-only
approximation is published as a verified Noisy LJ barostat.

`calibrate.pbs` runs the 20 train-only calibrations. `train.pbs` uses an immutable
source snapshot created by `prepare.py` under `results/sweep-v2/snapshots/<hash>`; each array index runs one manifest
cell. All old results remain untouched. `publish.pbs` runs after the arrays,
checks that every eligible cell completed, and commits/pushes only the three
website JSON files from an isolated checkout. Incomplete sweeps are not exported.

```bash
.venv/bin/python configs/sweep_v2/prepare.py
qsub -J 0-19 configs/sweep_v2/calibrate.pbs
# Submit training arrays only after calibration and pilot checks pass.
```

Preparation writes `results/sweep-v2/snapshot.json`; calibration, training and
publication all use that frozen source/config snapshot. Existing snapshots are
never overwritten. Completion markers are checked against the current job and
saved run configuration before reuse or publication.

Full-population scores are unavailable at horizons where any trajectory failed.
Completion counts are exported alongside scores; survivor-only values remain
explicitly labeled in JSON. For previously saved runs with no intermediate
counts, those horizons are conservatively unverified when final divergence is
nonzero. Re-evaluation can recover their exact intermediate coverage.

## Additional seeds 3 and 4

The manifest now appends 2,800 training cells and 400 controls for seeds 3–4,
keeping all 4,800 original indices intact. The full five-seed experiment contains
7,000 training runs and 1,000 controls. The website publishes all five seeds
except for Stiffness optimized OOD with the barostat on (see below).

Normal seeds use the same saved roster and shuffle rule. OOD training remains
the highest N; seeds change the held-out shuffle and training randomness.
All held-out splits remain 70 validation / 70 test. Noisy LJ barostat stays blocked.

PBS submissions must pass the path printed by preparation using
`qsub -v SNAPSHOT=/absolute/path/to/snapshot ...`; it is pinned for each array.
Submission IDs and queue caps for this extension are recorded locally in
`results/sweep-v2/seeds34-submissions.json`.

To publish newly completed comparisons while retaining previous results for
unfinished dataset/distribution/training/barostat combinations, run
`python configs/sweep_v2/export.py docs --completed-scopes`. The status JSON
records the withheld scopes and missing jobs; the default still requires the
entire sweep to finish. Jobs with a marker in `results/sweep-v2/failed/` count as
final: their comparison is listed under `failed_scopes` and marked failed on the site.

## Stiffness optimized OOD with the barostat on

This comparison has no published scores. Its calibration (top-10 OOD training
networks) picks coupling 0.0032 and damping 0.00032, an order of magnitude below
the Normal calibrations. With those constants the box diverges on many held-out
networks even when the barostat is driven by ground-truth positions (28 of 70
seed-0 validation networks exceed 0.1 relative box error, several reach 1e5),
so every model's rollouts diverge. Seeds 0–2 ran before rollout scores became
NaN on any divergence, so their saved survivor-only scores and checkpoint
selection are not published; seeds 3–4 failed checkpoint selection outright.
