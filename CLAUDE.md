# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CESM-forced TOMAS-JAX sectional aerosol model for stratospheric aerosol
injection, in JAX on GPU. The meteorology is prescribed — one-way *to CESM*, so
the winds are read and the circulation never responds — but aerosol, radiation
and microphysics are interactively coupled: radiative heating from the model's
own aerosol accumulates into the temperature the microphysics, the settling and
the next radiation call all see.

**Direction of travel.** The end goal is a model-agnostic dycore–aerosol–radiation
coupler: other meteorology sources (including emulated dycores, which would close
the circulation loop too) and other aerosol schemes (CARMA, MAM, GLOMAP). CESM and
TOMAS are what is wired in first, not the target. At any seam — the CESM reader,
the wind interface, `AER_SRC`, the engine swap in `driver_fast.py`, the radiation
driver — prefer keeping the seam generic over specialising it, and say so if a
change would make a slot harder to swap. `README.md` is the orientation;
**`MANIFEST.md` is the canonical reference for this tree** — it records every
default and, more importantly, *why* each one is what it is. Where `docs/`
disagrees with MANIFEST, MANIFEST is current. Read it before changing physics.

## Layout invariants

- **Source lives in `src/`, but the module names are flat on purpose.**
  `coupling.py` does bare `import settling` / `import radiation`, and
  `src/_paths.py` puts each `src/` subdirectory on `sys.path` so they resolve.
  `src/` is NOT a package: do not add `__init__.py` to a subdirectory whose name
  collides with a module (`src/settling/` vs `settling.py`), and do not let an
  import-sorting pass reorder modules that set up `sys.path` before importing
  what that setup makes importable. See `docs/REPO_LAYOUT.md`.
- `driver_fast.py` is the production entry point: it imports `coupling.py` and
  monkeypatches in the batched `tomas_jax.fast` engine. `coupling.py` alone is
  the standalone/dev path and uses the same advection, so the two agree.
- `fct_core.py` is legacy and **not on the run path** — it exists only for the
  bit-identical comparison in `validation/test_conservation.py`.
- Analysis scripts read `coupled_*_<TAG>.npz` from the *current working
  directory* and write `<TAG>_*.png` beside them.
- Prefer extending `coupling.py` over adding helper scripts; the four
  overlapping plot scripts this repo used to have are why `plot_run.py` is one
  file.

## Working on this code

- **A GPU is not optional** for anything but a smoke test, and JAX will
  *silently* fall back to CPU if it cannot load `libcuda.so.1` — `run_prod.sh`
  handles that. If a step takes minutes instead of seconds, check
  `jax.devices()` first.
- **Validate advection changes at `ADV_F32=1 ADV_CFL=0.5`**, the production
  precision, not the f64/cfl=0.2 module defaults. Two positivity-limiter bugs
  were invisible in f64 and fatal in f32.
- Changing a default changes what a bare run *means*. Every env var is read once
  at module import with `os.environ.get`, and the resolved config is echoed in
  the run header so any log is self-describing — keep it that way.
- The injection scenario (`INJ_*`) is stamped into the state checkpoint and a
  `RESUME` onto a mismatched checkpoint is refused; physics-mode flags
  (`WET_*`, `SETTLE`, `ADV_VPOS`) only warn. Both arrays are append-only:
  adding a field must not lock out older checkpoints.
- Diagnostics and plots report concentrations at STP; the microphysics kernels
  keep ambient density. Don't mix the two.
- The budget printout closes to roundoff (`sum` vs `M/M0-1`). If a change breaks
  that closure, the change is wrong — that line is the model's own audit.

## Run outputs

**Outputs live in `runs/<TAG>/`**, one directory per `OUT_TAG`, gitignored.
Every output path in `coupling.py` and `plot_run.py` is relative to the working
directory, so the working directory is the run:

```bash
mkdir -p runs/prod90d && cd runs/prod90d
OUT_TAG=prod90d ../../src/run_prod.sh
python3 ../../plot_run.py prod90d
```

`run_prod.sh` **refuses** to start from anywhere inside the checkout except
under `runs/`. That guard exists because the mistake is otherwise invisible: `.npz`/`.png` are gitignored, so a
run launched from the wrong place looks entirely normal while filling the tree
(this is how 29 GB accumulated before 2026-08-12). The scripts resolve their own
tree from `__file__`, so invoking them by absolute path from anywhere is correct
and is the intended usage — do not `cd` into the repo to run them.

A single 90-day run writes GBs of `.npz`, gitignored by broad patterns (`*.npz`,
`*.png`, `*.gif`) — never commit them, and never widen the tracked set to
include them. Nothing untracked belongs outside `runs/`: the checkpoint-reading
validation harnesses take their `STATE` from `$PWD` too, so run those from the
run directory as well.
