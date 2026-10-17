# CLAUDE.md

Project ground rules live in AGENTS.md, imported here so Claude Code loads them:

@AGENTS.md

The project overview, workflow and layout are in the README:

@README.md

## Working with the team

- **Ask before doing anything.** The owner's guidelines in `plan.md` require
  it. Propose a change and wait for approval before editing files, running
  training or submitting jobs.
- Work on a branch. Never push to `main` directly.

## Environment

- Python 3.14 (`.python-version`); `pyproject.toml` requires >=3.12.
- Use uv. `uv sync` creates `.venv` from `uv.lock` (`train.pbs` runs
  `.venv/bin/python`); run scripts with `uv run python ...`. Add dependencies
  with `uv add`, never by hand-editing the lock.
- The `analysis` extra (matplotlib, pandas) is optional. `network_bench`, and
  `network_bench.interfaces` in particular, must import without it.

## Data and results

- Trajectories are several GB and stay outside Git. Point
  `GNN_BENCH_DATA_ROOT` at them; on the cluster the default is
  `/rg/mendels_prj/s.sergey/data_bench`.
- `results/`, `datasets/`, `*.pt` and the generated
  `configs/sweep_v2/configs/` are gitignored. Never commit them.
- Published results depend on exact split memberships and saved run metadata.
  Do not change `configs/learning_curve/` splits or `docs/` result JSON unless
  you are asked to.

## Checking a change

There are no tests, and none should be added. To check that training still
runs end to end, use a small development config (`configs/smoke_*.json`):

```bash
uv run python scripts/run.py configs/smoke_tiny.json
```

Full runs go to the CPU-only PBS cluster (`mendels_q`) through `train.pbs`.
Don't start them locally.

## Website deploys

Pushing changes under `docs/`, `scripts/build_website.py` or
`.github/workflows/website.yml` to `main` publishes the public site through
GitHub Actions. Treat those pushes as releases.
