# Deferred work

Things deliberately not done, with the reason and what it would take. Each entry
is a decision, not a TODO that nobody got to.

## Convert `src/` into an installable Python package

**Status:** deferred to its own PR — tracked in issue #7.

`src/` is a flat collection of modules on `sys.path`, not a package. A proper
`src/aide_sai_link/` with `__init__.py` files and a `pyproject.toml` would remove
every `sys.path.insert()` in the tree, let `ruff`'s import-sorting rule (`I001`)
be switched back on, remove the `src/settling/` vs `settling.py` name shadowing,
and make the coupler importable and testable from anywhere.

**Why not now:** it rewrites every local `import` in `coupling.py` (160 KB),
`driver_fast.py`, `radiation.py`, both advection modules and all six validation
harnesses. The 2026-08-27 restructuring was verified by running the smoke case
before and after and comparing every output array — an acceptance test that only
means something while the executing bytes are unchanged. Bundling the import
rewrite in would have removed exactly that check.

**Also needs:** `run_prod.py` and `driver_fast.py` would depend on the package
being installed in the active environment, which introduces a stale-editable-
install failure mode. The launcher was already rewritten once to stop running
code from a path other than the one being edited; the same care applies here.

## Point the plotting scripts at `outputs/`

**Status:** deferred; `outputs/` exists and is documented but nothing writes there.

`scripts/utils/plot_run.py` still saves its dashboard/filmstrip/sizedist PNGs
into `$PWD`, which is the run directory under `runs/<TAG>/`. That mixes derived
products in with raw model output, which is the split `outputs/` was created to
make. Deferred from the 2026-08-27 `scripts/` move so that move stayed a pure
relocation with no behaviour change.

## Retire `fct_core.py`?

**Status:** open question, no action taken.

`src/advection/fct_core.py` is not on the runtime path. `coupling.py` records
that it deliberately replaced it with `fct_lr` because reaching `fct_core` gave
silently *different* transport (sealed vertical faces, no air-mass tracking)
under identical diagnostics. The only remaining importer is
`scripts/validation/test_conservation.py`, which loads it as a legacy reference to check
the modern sweeps against.

So it is not dead code exactly — it is a reference implementation with one
consumer. Deleting it would remove that comparison; keeping it leaves a module
that must never be imported by anything else. Worth an explicit decision rather
than a silent one.
