# Repository layout

Where everything lives and, more usefully, the rule for where a new file goes.

```
AIDE-SAI-link/
  models/          the two model repos, as pinned git submodules
    tomas-jax/       sectional aerosol microphysics  (branch gpu-fast)
    jax-rrtmgp/      radiative transfer              (main + zenith patch)
  coupling.py      the orchestrator: forcing, IC/BC, budget, main()
  driver_fast.py   production entry point (swaps in the fast engines)
  run_prod.sh      the launcher
  radiation.py  settling.py  fct_core.py  fast_advection/
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
| needed to advance the coupled state | the repo root (moving to `src/`) |
| a check that the physics is right | `validation/` |
| a figure or a plot script | `plot_run.py` / `gif_run.py`; products in `outputs/` |
| data the model reads and that is small | `inputs/` |
| data the model writes | `runs/<TAG>/` — never committed |
| a change to tomas-jax or jax-rrtmgp | that repo, upstream; or `patches/` if it cannot be |

## The two submodules

`models/tomas-jax` and `models/jax-rrtmgp` are pinned to exact commits, so a
clone reconstructs the code that produced a result.

```bash
git clone --recurse-submodules https://github.com/reflective-org/AIDE-SAI-link.git
# or, in an existing clone:
git submodule update --init
git -C models/jax-rrtmgp apply ../../patches/jax-rrtmgp-zenith.patch
```

`coupling.py` and `radiation.py` resolve them from `models/`,
overridable with `TOMAS_JAX_PATH` / `RRTMGP_PATH`.
