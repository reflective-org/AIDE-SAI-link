Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

[Unreleased]

Added

- `models/` — `tomas-jax` (pinned `787c991`, branch `gpu-fast`) and `jax-rrtmgp`
  (pinned `99d2d71`) are now git submodules, so a clone reconstructs the exact
  code behind a result instead of relying on two unpinned sibling clones.
- `inputs/` — small tracked static input data (`rad_data/`) plus a README
  recording the provenance of all three input datasets, including the external
  CESM archive and its `CESM_DIR` / `CESM_PREFIX` / `CESM_SUF` overrides.
- `runs/<TAG>/` for raw model output and `outputs/` for derived products. Both
  gitignored, each with a tracked README.
- `docs/REPO_LAYOUT.md` — the tree, and the rule for where a new file goes.
- `docs/DEFERRED.md` — work deliberately not done, with reasons.
- `src/run_prod.py` — the launcher, converted from `run_prod.sh`. Every one of
  the 83 environment variables the model reads is now a documented `--flag`;
  `--help` is the inventory and `--dry-run` prints the resolved environment with
  the source of each value. Verified to hand `driver_fast.py` a byte-identical
  environment to the shell version it replaces.
- `FRAME_LEVELS` — `probe` (default, unchanged) or `all`, which adds a
  zonal-mean frame at every band level (`(nf, NBINS, nlev, nlat)`, +1.1 GB/year)
  beside the probe-level slabs. Without it the frames have a size axis but no
  vertical axis, so a Hovmöller at any altitude but the probe level, and any
  latitude–height animation, were not constructible from a run's own output.
- `scripts/utils/plot_run.py --level HPA` and a fourth figure,
  `<TAG>_zonal.png`; `scripts/utils/gif_run.py` gains `<TAG>_zonal_so4.gif` and
  `<TAG>_zonal_dTrad.gif` when the run carries zonal frames.
- `scripts/utils/run_summary.py` — parses a run log's `[prof]` lines into
  `<TAG>_summary.md`, quarter by quarter so a cost trend is visible rather than
  averaged away.

Changed

- `coupling._dep_path` and its deliberate duplicate in `radiation.py` resolve
  the dependencies from `models/<name>` rather than `../<name>`. The
  `TOMAS_JAX_PATH` / `RRTMGP_PATH` overrides are unchanged and still win.
- `patches/jax-rrtmgp-zenith.patch` regenerated against the pinned commit; the
  old one targeted v0.2.1 (`d7abe2e`) and no longer applied. Applying it is now
  documented as required setup, because a gitlink cannot carry the fix.
- `radiation.RI_FILE` resolves from the repo root rather than `__file__`.
- Documentation that told the reader to clone the dependencies as siblings —
  README, MANIFEST, `docs/README.md`, `docs/CONFIGURATION.md`,
  `docs/VALIDATION.md` and two `ImportError` messages — now describes the
  submodule layout. Following the old instructions would have produced a tree
  the resolver does not look at.

- `src/` — all source needed to advance the coupled state, grouped by process:
  `coupling.py`, `driver_fast.py`, `run_prod.sh`, `advection/`, `radiation/`,
  `settling/`, `microphysics/`. Module names stay flat and **no import statement
  changed**; `src/_paths.py` puts the subdirectories on `sys.path`.
- `src/microphysics/tomas_fast.py` — the `tomas_jax.fast` adapter, extracted from
  `driver_fast.py` so every coupled process has a home under `src/`.
- `run_prod.sh` refuses to run from anywhere inside the checkout except under
  `runs/`. The old test was `$PWD == <script dir>`; from `src/` that would no
  longer have protected the repo root.

- `scripts/` — everything that reads a run rather than producing one:
  `scripts/validation/` (the six physics harnesses) and `scripts/utils/`
  (`plot_run.py`, `gif_run.py`). `src/` now holds only what advances the coupled
  state; a run depends on nothing in `scripts/`.

Removed

- `src/run_prod.sh` — replaced by `src/run_prod.py`. It survived briefly as a
  forwarding shim so that in-flight runs whose `run_chain.sh` called it by name
  could still `RESUME`; those runs finished, their `run_chain.sh` now call the
  Python launcher, and the shim is gone.
- `rad_data/` at the repo root (moved to `inputs/rad_data/`).
- `validation/`, `plot_run.py`, `gif_run.py` at the repo root (moved to `scripts/`).
- `fast_advection/` (moved to `src/advection/`).
- `sai_runs/` beside the repo, and the loose `coupled_*prod1d*` artifacts in its
  parent directory (moved to `runs/smoke/` and `runs/prod1d/`; nothing deleted).