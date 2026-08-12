# aide_sai_core

One-way coupled **CESM → TOMAS-JAX** sectional aerosol model for stratospheric
aerosol injection (SAI), in JAX on GPU.

> [!WARNING]
> This project is under active development, there is no guarantee that it will
> operate as expected. In particular the model is **not calibrated for absolute
> radiative forcing** — see the standing caveats in [MANIFEST.md](./MANIFEST.md)
> before quoting any number from it.

CESM meteorology forces the aerosols, but there is no feedback from the aerosols to the
circulation. The aerosols in the stratosphere are internally generated
and internally removed:

```
CESM h1 hourly ──► U, V, OMEGA ──────► transport (Lin-Rood flux-form)
               ├─► T, p, RELHUM ─────► microphysics (TOMAS, 40 bins)
               ├─► MAM4 num/so4 ─────► IC, open-BC reservoir, polar caps
               └─► SO2, H2SO4 ───────► gas-phase IC/BC

  SOURCE  continuous SO2 injection    SO2 ─OH─► H2SO4 ─► nucleation/condensation
  SINK    gravitational settling out of the band bottom
  OPTICS  RRTMGP + Mie on the wet H2SO4/H2O droplet ──► heating, AOD550, ARF_toa
```

- **State**: 40 TOMAS bins × (number, dry SO4 mass) + 2 gas tracers = 82
  advected 3-D fields, on the native CESM f09 grid (192 × 288) over a
  1–150 hPa band (24 levels), 6 h coupling step.
- **Scale**: a 90-day run is ~33 h on one H100; microphysics is ~94% of it.

## Installation

```bash
git clone https://github.com/reflective-org/aide_sai_core.git
cd aide_sai_core
pip install -r requirements.txt
pip install --upgrade "jax[cuda12]>=0.6.2"   # GPU wheel -- the CPU one cannot do production
```

Two dependencies are separate repos, not on PyPI. Clone them **beside** this one
and they are found automatically; otherwise set `TOMAS_JAX_PATH` / `RRTMGP_PATH`:

| repo | role |
|---|---|
| [`reflective-org/tomas-jax`](https://github.com/reflective-org/tomas-jax) | sectional aerosol microphysics |
| [`climate-analytics-lab/jax-rrtmgp`](https://github.com/climate-analytics-lab/jax-rrtmgp) | radiative transfer (`RAD=0` skips it) |

**The CESM archive is ~23 TB, so it can't be bundled in a git repo.** The model
reads 21 hourly `h1` variables — 14 for the coupling itself (`U V OMEGA T
num_a{1,2,3} so4_a{1,2,3} SO2 H2SO4 OH RELHUM`) and, unless `RAD=0`, 7 more for
radiation (`Q O3 CH4 FLDS FLNS FSDS FSNS`). Point `CESM_DIR` at an archive laid
out as
`$CESM_DIR/hour_1/$CESM_PREFIX.h1.<VAR>$CESM_SUF`. Reads are one hour × band
levels at a time (~5.3 MB per variable-hour), so a subset is enough: ~3.6 GB for
a 2-day test, ~160 GB for a 90-day run. Full variable list and units:
[docs/COUPLING_VARIABLES.md](./docs/COUPLING_VARIABLES.md).

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
REPO=~/noah/coupling_prod          # wherever this clone lives
mkdir -p ~/noah/coupling_runs && cd ~/noah/coupling_runs

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
OUT_TAG=inj20_30N INJ_SO2_TG_YR=20 INJ_LAT=30 INJ_MIRROR=1 ~/noah/coupling_prod/run_prod.sh
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
| run the standard scenario | `OUT_TAG=<name> INJ_SO2_TG_YR=10 N_HOURS=2160 run_prod.sh` |
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

## Running Tests

```bash
python3 $REPO/validation/test_conservation.py                        # advection conservation
# needs a checkpoint, so run it from your runs directory (or set STATE=<path>)
ADV_F32=1 ADV_CFL=0.5 python3 $REPO/validation/validate_vpos_f32.py  # positivity limiter
```

These resolve the repo from their own location, so they run from anywhere. The
ones that read a checkpoint (`validate_vpos_f32.py`, `floor_anatomy.py`) take it
from the working directory by default, and the one that draws a figure
(`validate_radiation.py`) writes it there — so run them where your output lives,
not in the repo.

Validate advection changes at `ADV_F32=1 ADV_CFL=0.5`, the production precision
— **not** the f64/cfl=0.2 module defaults: two positivity-limiter bugs were
invisible in f64 and fatal in f32.

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
