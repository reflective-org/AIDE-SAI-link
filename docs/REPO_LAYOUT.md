# Repository layout

Where everything lives and, more usefully, the rule for where a new file goes.

```
AIDE-SAI-link/
  models/          the two model repos, as pinned git submodules
    tomas-jax/       sectional aerosol microphysics  (branch gpu-fast)
    jax-rrtmgp/      radiative transfer              (main + zenith patch)
  src/             ALL source code needed for the coupling
    _paths.py        import-path setup; also defines REPO_ROOT and MODELS
    coupling.py      the orchestrator: forcing, IC/BC, budget, main()
    driver_fast.py   production entry point (swaps in the fast engines)
    run_prod.sh      the launcher
    advection/       fct_lr.py (production), fct_fast.py, fct_core.py (legacy)
    radiation/       radiation.py -- Mie + RRTMGP -> heating, AOD, ARF
    settling/        settling.py -- closed-form gravitational settling
    microphysics/    tomas_fast.py -- the tomas_jax.fast adapter
  inputs/          small static input data + provenance of the external archives
  runs/            raw model output, one directory per OUT_TAG   (gitignored)
  outputs/         derived figures and tables                    (gitignored)
  validation/      physics validation harnesses
  docs/            documentation
  patches/         patches that must be applied to a submodule
  plot_run.py  gif_run.py                                        (plotting)
```

## Where does a new file go?

| the file is… | it goes in |
|---|---|
| needed to advance the coupled state | `src/<process>/` |
| a new physical process | a new `src/<process>/` directory |
| a check that the physics is right | `validation/` |
| a figure or a plot script | `plot_run.py` / `gif_run.py` today, `outputs/` for their products |
| data the model reads and that is small | `inputs/` |
| data the model writes | `runs/<TAG>/` — never committed |
| a change to tomas-jax or jax-rrtmgp | that repo, upstream; or `patches/` if it cannot be |

## How imports work inside `src/`

`src/` is deliberately **not** a Python package. Every module is a flat,
top-level name (`coupling`, `settling`, `fct_lr`, `radiation`, `tomas_fast`) and
the subdirectories are organisational only. `src/_paths.py` puts each
subdirectory on `sys.path`, so `import settling` works from anywhere in the tree
with no package prefix.

Two consequences worth knowing before editing:

1. **Do not add `__init__.py`** to a `src/` subdirectory without converting all
   the imports at the same time. `src/settling/` and `settling.py` share a name;
   the module wins today, but an `__init__.py` would turn the directory into a
   regular package and shadow it. `_paths.py` inserts the subdirectories ahead of
   `src/` itself for the same reason.
2. **Do not let an "organize imports" pass reorder these files.** Several modules
   run path setup before importing what that setup makes importable, and
   `driver_fast.py` must set its environment defaults before importing anything
   that reads the environment at import time. This is why `ruff` rule `I001` is
   off in CI.

Converting `src/` into a proper installable package removes all of the above.
That is deliberately deferred — see `docs/DEFERRED.md`.

## The two submodules

`models/tomas-jax` and `models/jax-rrtmgp` are pinned to exact commits, so a
clone reconstructs the code that produced a result.

```bash
git clone --recurse-submodules https://github.com/reflective-org/AIDE-SAI-link.git
# or, in an existing clone:
git submodule update --init
git -C models/jax-rrtmgp apply ../../patches/jax-rrtmgp-zenith.patch
```

`src/coupling.py` and `src/radiation/radiation.py` resolve them from `models/`,
overridable with `TOMAS_JAX_PATH` / `RRTMGP_PATH`.
