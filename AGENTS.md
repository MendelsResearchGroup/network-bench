# Ground rules for agents

From the technical guidelines in `plan.md`.

- **Keep it simple.** The project should be straightforward and easy for a person
  to read. Prefer the smallest thing that works; no frameworks, layers or extra
  options nobody asked for.
- **No defensive checks.** Checking that consumed data or a model fits the
  interface is fine. Do not add input validation anywhere else unless it is critical.
- **No tests.** Do not add any for now.
- **Standalone, importable.** The data and model interfaces must stay importable
  from outside, via an install from GitHub.
- **CPU HPC cluster.** The project runs on a CPU-only PBS cluster (`mendels_q`).
