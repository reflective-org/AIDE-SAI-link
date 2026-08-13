# aide_sai_core

[![CI](https://github.com/reflective-org/aide_sai_core/actions/workflows/ci.yml/badge.svg)](https://github.com/reflective-org/aide_sai_core/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](./LICENSE)

**CESM → TOMAS-JAX** sectional aerosol model for stratospheric aerosol injection
(SAI), in JAX on GPU. The CESM forcing is one-way; aerosol, radiation and
microphysics are fully coupled to each other.

> [!NOTE]
> CI covers what a GPU-less runner without the CESM archive can cover: that the
> tree imports, that advection still conserves mass, and that the closed-form
> settling physics is unchanged. The rest of the validation needs a GPU and the
> CESM archive — see [docs/VALIDATION.md](./docs/VALIDATION.md).

> [!WARNING]
> This project is under active development, there is no guarantee that it will
> operate as expected. In particular the model is **not calibrated for absolute
> radiative forcing** — see the standing caveats in [MANIFEST.md](./MANIFEST.md)
> before quoting any number from it.

**What "one-way" does and does not mean.** CESM meteorology is prescribed: the
winds are read hour by hour and nothing this model does ever changes them, so
there is no dynamical response — no self-lofting, no altered Brewer-Dobson
ascent — and nothing propagates back to CESM. *Inside* that forcing, though,
the loop closes: the aerosol sets the optics, the optics heat the layer, and
that heating accumulates into the temperature which the microphysics, the
settling and the next radiation call all see. Aerosol in the band is internally
generated and internally removed:

```
CESM h1 hourly ──► U, V, OMEGA ──────► transport (Lin-Rood flux-form)
               ├─► T, p, RELHUM ─────► microphysics (TOMAS, 40 bins)
               ├─► MAM4 num/so4 ─────► IC, open-BC reservoir, polar caps
               └─► SO2, H2SO4 ───────► gas-phase IC/BC

  SOURCE  continuous SO2 injection    SO2 ─OH─► H2SO4 ─► nucleation/condensation
  SINK    gravitational settling out of the band bottom
  OPTICS  RRTMGP + Mie on the wet H2SO4/H2O droplet ──► heating, AOD550, ARF_toa

  CLOSED LOOP   aerosol ─► optics ─► heating ─► dT accumulates into T
                    ▲                                    │
                    └──── microphysics + settling ◄───────┘
  NOT CLOSED    U, V, OMEGA are CESM's throughout; the circulation never responds
```

- **State**: 40 TOMAS bins × (number, dry SO4 mass) + 2 gas tracers = 82
  advected 3-D fields, on the native CESM f09 grid (192 × 288) over a
  1–150 hPa band (24 levels), 6 h coupling step.
- **Scale**: a 90-day run is ~33 h on one H100; microphysics is ~94% of it.

## Scope, and where this is going

Read this as a **dycore–aerosol–radiation coupler**. **This version has CESM,
TOMAS and RRTMGP plugged in**, and every number, default and validation on this
page refers to that combination.

The intent is for each slot to be swappable — other meteorology sources
including emulated dycores, and other aerosol schemes (CARMA, MAM, GLOMAP).
Some of that seam already exists: `AER_SRC` selects the aerosol IC/BC source
(`mam4` or `carma`), and `driver_fast.py` swaps the microphysics engine into
`coupling.py` at import rather than branching inside it. The meteorology reader
is **not** generic yet — it expects the CESM `h1` layout — so treat a new
forcing source as work, not configuration.

## Installation

Two dependencies are separate repos, not on PyPI. Clone them **beside** this one
and they are found automatically; otherwise set `TOMAS_JAX_PATH` / `RRTMGP_PATH`:

| repo | role | required state |
|---|---|---|
| [`reflective-org/tomas-jax`](https://github.com/reflective-org/tomas-jax) | sectional aerosol microphysics | branch **`gpu-fast`** — `main` has no `tomas_jax.fast` |
| [`climate-analytics-lab/jax-rrtmgp`](https://github.com/climate-analytics-lab/jax-rrtmgp) | radiative transfer (`RAD=0` skips it) | `main`, **plus `patches/jax-rrtmgp-zenith.patch`** from this repo |

Neither is a plain clone of its default branch:

* **tomas-jax must be on `gpu-fast`.** `driver_fast.py` — the production entry
  point — imports `tomas_jax.fast`, the batched reduced engine, which only
  exists on that branch. On `main` the import fails outright.
* **jax-rrtmgp needs the zenith patch.** `radiation.py` passes a per-column
  solar zenith field, shape `(nlat, nlon, 1)`; upstream `sw_cell_source`
  assumes a scalar and mis-broadcasts it against the `(nlat, nlon)` TOA flux.
  The one-hunk fix is checked in here as
  [`patches/jax-rrtmgp-zenith.patch`](./patches/jax-rrtmgp-zenith.patch) and is
  needed for any `RAD=1` run.

```bash
git clone https://github.com/reflective-org/aide_sai_core.git

# microphysics -- the gpu-fast branch, not main
git clone -b gpu-fast https://github.com/reflective-org/tomas-jax

# radiation -- clone, then apply the zenith patch shipped with this repo
git clone https://github.com/climate-analytics-lab/jax-rrtmgp
git -C jax-rrtmgp apply ../aide_sai_core/patches/jax-rrtmgp-zenith.patch

pip install -r aide_sai_core/requirements.txt
pip install --upgrade "jax[cuda12]>=0.6.2"   # GPU wheel -- the CPU one cannot do production
```

The patch was made against jax-rrtmgp v0.2.1 (`d7abe2e`); `git apply` fails
loudly rather than half-applying if upstream has moved, in which case pin that
tag. Do **not** activate the tomas-jax `.venv` — it is CPU-only jaxlib with no
xarray. The working combination is system `python3` with the GPU jax wheel.

Verify the install before touching CESM data — this resolves both sibling repos
exactly the way the model does, so it fails the same way a run would:

```bash
REPO=~/aide_sai_core
grep -q mu_2d $(dirname $REPO)/jax-rrtmgp/rrtmgp/rte/monochromatic_two_stream.py \
  && echo "zenith patch applied"
python3 -c "
import sys; sys.path.insert(0, '$REPO')
import radiation                      # puts both repos on sys.path; imports rrtmgp + tomas-jax Mie
from tomas_jax.fast import run_fast   # exists only on tomas-jax gpu-fast
import jax; print('deps ok:', jax.devices())
"
```

`jax.devices()` printing `[CpuDevice(id=0)]` here is expected outside
`run_prod.sh`, which is what puts `libcuda.so.1` on `LD_LIBRARY_PATH` (see
`CUDA_DRIVER_LIB` below); inside a run it must show a GPU.

**The CESM archive is ~23 TB, so it can't be bundled in a git repo.** The model
reads 21 hourly `h1` variables — 14 for the coupling itself (`U V OMEGA T
num_a{1,2,3} so4_a{1,2,3} SO2 H2SO4 OH RELHUM`) and, unless `RAD=0`, 7 more for
radiation (`Q O3 CH4 FLDS FLNS FSDS FSNS`). Point `CESM_DIR` at an archive laid
out as
`$CESM_DIR/hour_1/$CESM_PREFIX.h1.<VAR>$CESM_SUF`. Reads are one hour × band
levels at a time (~5.3 MB per variable-hour), so a subset is enough: ~3.6 GB for
a 2-day test, ~160 GB for a 90-day run. Full variable list and units:
[docs/COUPLING_VARIABLES.md](./docs/COUPLING_VARIABLES.md).

**On the shared H100 box these runs were made on, set none of this.** The three
defaults in `coupling.py` already resolve to the FWHIST archive under `/data`,
which is readable by every account on the machine, so the install above is the
whole setup and `run_prod.sh` works as written. `CESM_DIR`/`CESM_PREFIX`/
`CESM_SUF` matter only off that box. The one place any CESM path is built is
`coupling.py` (`H1`); `radiation.py` is handed the opener rather than
constructing paths of its own, so there is nothing else to repoint.

CESM's own *internally calculated* aerosol radiative forcing is neither read nor
needed: this model computes forcing from its own aerosol, through RRTMGP and its
own Mie tables. The four flux fields above are surface boundary conditions only
— `FLNS`/`FLDS` give the surface temperature and `FSDS`/`FSNS` the albedo.

## Quick Start

**Run from a directory that is not the repo.** Every output path is relative to
the working directory, so that directory is where the run lands — and a single
90-day run writes GBs of `.npz`. `run_prod.sh` refuses to start with the repo
itself as the working directory rather than let that happen silently.

```bash
REPO=~/aide_sai_core               # wherever this clone lives
mkdir -p ~/sai_runs && cd ~/sai_runs   # anywhere BUT the repo

# 1-step smoke test: N_HOURS=6 is one 6 h coupling step, ~6 min on an H100.
N_HOURS=6 OUT_TAG=smoke INJ_SO2_TG_YR=10 $REPO/run_prod.sh
python3 $REPO/plot_run.py smoke               # -> smoke_{dashboard,filmstrip,sizedist}.png

# the production scenario: 10 Tg SO2/yr equatorial ring, 90 days.
# N_HOURS=2160 (2160 h / 6 h = 360 steps = 90 days) is run_prod.sh's default,
# so it is written out here only to make the run length explicit.
N_HOURS=2160 INJ_SO2_TG_YR=10 OUT_TAG=prod90d $REPO/run_prod.sh          # ~33-36 h on one H100
RESUME=1 N_HOURS=2160 INJ_SO2_TG_YR=10 OUT_TAG=prod90d $REPO/run_prod.sh # continue after a kill
python3 $REPO/plot_run.py prod90d

# a full year is the same launcher with a longer clock (8760 h = 365 days, ~140 h)
RESUME=1 N_HOURS=8760 INJ_SO2_TG_YR=10 OUT_TAG=prod1yr $REPO/run_prod.sh
```

Run length is set **in hours**, always: `N_HOURS` counts forcing hours from
`H0`, and the model takes one coupling step every `STEP_HOURS` (6). So
`N_HOURS=6` is one step, `N_HOURS=48` is two days, `N_HOURS=2160` is the 90-day
production run. `N_DAYS` exists only as the fallback that supplies `N_HOURS`
when it is unset (`N_HOURS = 24 × N_DAYS`, default 2 days) — for anything
scripted, set `N_HOURS`.

`run_prod.sh` sets the whole production environment (GPU pin, `libcuda` path,
memory policy) and execs `driver_fast.py` from its own tree — so it always runs
the code you just edited, wherever you launched it from. **A bare `run_prod.sh`
is a no-injection control** — `INJ_SO2_TG_YR` defaults to 0 so a forgotten flag
gives an obviously unforced baseline rather than a silent SAI result. Always
pair a scenario with its own `OUT_TAG`; outputs and checkpoints are keyed by it,
and reusing a tag overwrites the other scenario's results — including across two
checkouts launched from the same runs directory.

Each run writes, into the launch directory:

| file | what |
|---|---|
| `coupled_final_<TAG>.npz` | full 3-D prognostic state at the end of the run |
| `coupled_frames_<TAG>.npz` | probe-level snapshots every `FRAME_EVERY` hours |
| `coupled_timeseries_<TAG>.npz` | per-step scalar diagnostics (burdens, AOD550, ARF) |
| `coupled_state_<TAG>_ckpt.npz` | restart checkpoint (+ `_frames_`/`_timeseries_` twins) |

Then `python3 plot_run.py <TAG>` writes the three figures and
`python3 gif_run.py <TAG> [--fps 8] [--width 900]` animates the filmstrip
panels. Both read `coupled_*_<TAG>.npz` from the **current working directory**.

## Configuration

Every knob is an environment variable, read **once at module import** and echoed
in the run header, so any log is self-describing. Set them by prefixing the
launcher:

```bash
OUT_TAG=inj20_30N INJ_SO2_TG_YR=20 INJ_LAT=30 INJ_MIRROR=1 $REPO/run_prod.sh
```

**The complete reference is [docs/CONFIGURATION.md](./docs/CONFIGURATION.md)** —
every variable, its production default, and which ones your run actually reads
(that last part matters: most knobs are inert for any given run, since
`run_prod.sh` swaps in the fast microphysics engine and the per-cell chain's
knobs are then never reached).

For a first run you need at most five variables. The rest have production
defaults for a reason — change them only with `MANIFEST.md` open.

| you want to | set |
|---|---|
| run the standard scenario | `OUT_TAG=<name> INJ_SO2_TG_YR=10 N_HOURS=2160 $REPO/run_prod.sh` |
| run its control | the same, minus `INJ_SO2_TG_YR` (it defaults to 0) |
| continue after a kill or crash | add `RESUME=1`, and **repeat the same `INJ_*` and `OUT_TAG`** |
| move the injection | `INJ_HPA`, `INJ_LAT` (+ `INJ_MIRROR=1` for a two-hemisphere ring) |
| share the GPU / recover from an OOM | `FAST_SORT=0` first; then `TRACER_CHUNK` (advection) or `RAD_LAT_CHUNK` (radiation), whichever phase OOMed |
| do a cheap dynamics-only test | `RAD=0` (drops the `jax-rrtmgp` dependency too) |

> [!IMPORTANT]
> `run_prod.sh` **hard-sets** seven variables (`INIT_BIN`, `STATE_CKPT`,
> `FRAME_EVERY`, `FAST_CELL_CAP`, `ADV_VPOS`, `DEBUG`, `PROFILE`). Passing those
> on the command line is *silently ignored* — see
> [docs/CONFIGURATION.md](./docs/CONFIGURATION.md) for how to override them.

Every default, and the reasoning behind it, is in
**[MANIFEST.md](./MANIFEST.md) — the canonical reference for this tree.** Where
`docs/` disagrees with it, MANIFEST is current.

## Citations

The parameterizations this model composes are other people's work:

- Lin & Rood (1996), *Mon. Weather Rev.* 124, 2046 — flux-form advection
- Adams & Seinfeld (2002), *J. Geophys. Res.* 107, 4370 — TOMAS sectional aerosol
- Tabazadeh et al. (1997), *Geophys. Res. Lett.* 24, 1931 — H2SO4/H2O equilibrium composition
- Tang (1997), *J. Geophys. Res.* 102, 1883 — solution density
- Palmer & Williams (1975), *Appl. Opt.* 14, 208 — H2SO4 refractive indices (via HITRAN Aerosols-2016)
- Pincus et al. (2019), *J. Adv. Model. Earth Syst.* 11, 3074 — RRTMGP
- Hanisco et al. (2001), *Geophys. Res. Lett.* 28, 4423 — the diurnal OH curve

## License

This project is released under the Apache 2.0 License - see the [LICENSE](./LICENSE) file for details.
