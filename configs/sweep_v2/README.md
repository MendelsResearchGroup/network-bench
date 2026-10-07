# Detached MST and barostat sweep

Fresh training on the original `learning_curve` memberships: seven trainable
models, Normal/OOD, training sizes10–100 by10, seeds0–2, 70 validation and70
test networks. Real frames0–3 seed every100-step rollout. No ITPO or simulator
bootstrap. Shared data stays at `/rg/mendels_prj/s.sergey/data_bench`.

GNS explicitly uses2 message-passing layers. MST detaches feedback, trains100
epochs without early stopping, and grows its horizon at zero-based epochs
0/10/20/30/40/50 to1/2/3/5/8/10 steps. Checkpoint selection starts at numbered
epoch51 (the10-step stage), uses validation Poisson R²@100, and retains the
final epoch100 checkpoint. One-step keeps cap40 and patience5.

The barostat is from PR#2 at `5adb396f08c59c0e8e00c0a990f10f36cfdff074`.
Its two constants are calibrated by the PR coarse/fine grid search using the
first10 **training** trajectories only. Those IDs are identical across the
nested sizes for a dataset/distribution/seed, so frozen parameters are reused
across sizes, models and objectives. Evaluation never fits missing parameters.
MST uses the updated box to construct subsequent inputs; one-step uses it in
validation and evaluation. Calibration details are stored separately.

Reported Poisson R² compares to ground-truth box strain. With the barostat,
predictions use box strain; without it, predictions use the group's unchanged
side-node estimator at `734c6b81b5cddc539d1fad2e465e333f45ee5a90`. Thus the
toggle changes both the physical controller and the prediction estimator;
position MSE against GT is also shown. Negative R² is retained in JSON.

`prepare.py` creates4,200 training cells and600 untrained frozen/zero-acceleration
controls. Another840 Noisy LJ barostat training cells are explicitly blocked:
that archived dataset has no verified force field/rest lengths. No spring-only
approximation is published as a verified Noisy LJ barostat.

`calibrate.pbs` runs the12 train-only calibrations. `train.pbs` uses an immutable
source snapshot at `results/sweep-v2/source`; each array index runs one manifest
cell. All old results remain untouched. `publish.pbs` runs after the arrays,
checks that every eligible cell completed, and commits/pushes only the three
website JSON files from an isolated checkout. Incomplete sweeps are not exported.

```bash
.venv/bin/python configs/sweep_v2/prepare.py
qsub -J 0-11 configs/sweep_v2/calibrate.pbs
# Submit training arrays only after calibration and pilot checks pass.
```
