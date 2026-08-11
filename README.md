# aide_sai_core

One-way coupled **CESM → TOMAS-JAX** sectional aerosol model for stratospheric
aerosol injection (SAI), in JAX on GPU.

> [!WARNING]
> This project is under active development, there is no guarantee that it will
> operate as expected. In particular the model is **not calibrated for absolute
> radiative forcing** — see the standing caveats in [MANIFEST.md](./MANIFEST.md)
> before quoting any number out of it.

CESM meteorology forces the aerosol; the aerosol does not feed back on the
circulation. Within the stratospheric band the aerosol is internally generated
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

**The CESM forcing is not included and cannot be.** The model reads 14 hourly
`h1` variables (`U V OMEGA T num_a{1,2,3} so4_a{1,2,3} SO2 H2SO4 OH RELHUM`); as
archived they total ~23 TB. Point `CESM_DIR` at an archive laid out as
`$CESM_DIR/hour_1/$CESM_PREFIX.h1.<VAR>$CESM_SUF`. Reads are one hour × band
levels at a time (~5.3 MB per variable-hour), so a subset is enough: ~3.6 GB for
a 2-day test, ~160 GB for a 90-day run. Full variable list and units:
[docs/COUPLING_VARIABLES.md](./docs/COUPLING_VARIABLES.md).

## Quick Start

```bash
# 1-step smoke test: N_HOURS=6 is one 6 h coupling step, ~6 min on an H100.
N_HOURS=6 OUT_TAG=smoke INJ_SO2_TG_YR=10 ./run_prod.sh
python3 plot_run.py smoke                     # -> smoke_{dashboard,filmstrip,sizedist}.png

# the production scenario: 10 Tg SO2/yr equatorial ring, 90 days.
# N_HOURS=2160 (2160 h / 6 h = 360 steps = 90 days) is run_prod.sh's default,
# so it is written out here only to make the run length explicit.
N_HOURS=2160 INJ_SO2_TG_YR=10 OUT_TAG=prod90d ./run_prod.sh          # ~33-36 h on one H100
RESUME=1 N_HOURS=2160 INJ_SO2_TG_YR=10 OUT_TAG=prod90d ./run_prod.sh # continue after a kill
python3 plot_run.py prod90d

# a full year is the same launcher with a longer clock (8760 h = 365 days, ~140 h)
RESUME=1 N_HOURS=8760 INJ_SO2_TG_YR=10 OUT_TAG=prod1yr ./run_prod.sh
```

Run length is set **in hours**, always: `N_HOURS` counts forcing hours from
`H0`, and the model takes one coupling step every `STEP_HOURS` (6). So
`N_HOURS=6` is one step, `N_HOURS=48` is two days, `N_HOURS=2160` is the 90-day
production run. `N_DAYS` exists only as the fallback that supplies `N_HOURS`
when it is unset (`N_HOURS = 24 × N_DAYS`, default 2 days) — for anything
scripted, set `N_HOURS`.

`run_prod.sh` sets the whole production environment (GPU pin, `libcuda` path,
memory policy) and execs `driver_fast.py`. **A bare `./run_prod.sh` is a
no-injection control** — `INJ_SO2_TG_YR` defaults to 0 so a forgotten flag gives
an obviously unforced baseline rather than a silent SAI result. Always pair a
scenario with its own `OUT_TAG`; outputs and checkpoints are keyed by it, and
reusing a tag overwrites the other scenario's results.

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
OUT_TAG=inj20_30N INJ_SO2_TG_YR=20 INJ_LAT=30 INJ_MIRROR=1 ./run_prod.sh
```

Anything not listed in `run_prod.sh`'s own prefix block simply passes through
from your environment — `AER_SRC=carma ./run_prod.sh` works. The exception:

> [!IMPORTANT]
> `run_prod.sh` **hard-sets** `INIT_BIN=so4`, `STATE_CKPT=1`, `FRAME_EVERY=24`,
> `FAST_CELL_CAP=50000`, `ADV_VPOS=1`, `DEBUG=1`, `PROFILE=1`. Passing those on
> the command line is silently ignored. Override them by editing the script, or
> run `driver_fast.py` directly. The remaining launcher variables —
> `N_HOURS`, `OUT_TAG`, `RESUME`, all `INJ_*`, `FAST_SORT`, `GPU`,
> `CUDA_DRIVER_LIB`, `XLA_PYTHON_CLIENT_PREALLOCATE` — are overridable.

The **Default** column below is the *effective* production default, i.e. what
you get from `./run_prod.sh`; where the bare `coupling.py` module default
differs it is noted.

### Injection scenario — the knobs meant to change run to run

| variable | default | meaning |
|---|---|---|
| `INJ_SO2_TG_YR` | `0.0` | continuous SO2 release, Tg(SO2)/yr. **0 = no-injection control.** Pass `10` to reproduce `prod90d` |
| `INJ_H2SO4_TG_YR` | `0.0` | direct gas-phase H2SO4 release at the same geometry. S-equivalent of *X* Tg SO2/yr is `X*1.531` |
| `INJ_HPA` | `55.0` | target altitude [hPa], snapped to the nearest model level |
| `INJ_LAT` | `0.0` | target latitude [deg], snapped to the nearest row |
| `INJ_LON` | `180.0` | target longitude [deg E]; **ignored** when `INJ_ZONAL=1` |
| `INJ_ZONAL` | `1` | `1` = spread over the whole latitude ring; `0` = single cell (5.6× slower, drives runaway nucleation) |
| `INJ_MIRROR` | `0` | `1` = release at **both** ±`INJ_LAT`, splitting the *same* total 50/50 (not doubled). No-op at `INJ_LAT=0` |

`INJ_*` is stamped into the state checkpoint, and a `RESUME` onto a mismatched
checkpoint is **refused** — repeat the scenario flags on the resume command
line, they are not read back.

### Run length, output and restart

| variable | default | meaning |
|---|---|---|
| `N_HOURS` | `2160` | forcing hours to integrate (module: `24*N_DAYS` = 48) |
| `N_DAYS` | `2` | fallback that supplies `N_HOURS` when it is unset |
| `H0` | `0` | start hour index into the CESM h1 series |
| `STEP_HOURS` | `6` | coupling step [h]. `driver_fast.py` **requires 6** and exits otherwise |
| `OUT_TAG` | `prod90d` | names every output file (module: `<N_DAYS>day`; `driver_fast.py` alone: `tomas_fast`) |
| `RESUME` | `0` | `1` = continue from `coupled_state_<TAG>_ckpt.npz` |
| `STATE_CKPT` | `1` | write the restart checkpoint (~400 MB, atomic, overwritten in place) |
| `FRAME_EVERY` | `24` | hours between frame + checkpoint writes |
| `LOG_EVERY` | `1` | progress line every N hours |
| `PROBE_HPA` | `50` | level [hPa] used for frames and probe diagnostics |
| `DIAG_CORE_HPA` | *(unset)* | `lo,hi` diagnostic core window; unset = symmetric −1 level per end (1.6–121.5 hPa) |
| `DEBUG` | `1` | verbose per-step budget/diagnostic output |
| `PROFILE` | `1` | per-phase timing breakdown |

Checkpoints are written frames → timeseries → **state last**, so the physics
state is never newer than the diagnostics.

### Input data and dependency paths

| variable | default | meaning |
|---|---|---|
| `CESM_DIR` | *(the FWHIST archive these results were made from)* | root of the CESM tseries archive; layout `$CESM_DIR/hour_1/$CESM_PREFIX.h1.<VAR>$CESM_SUF` |
| `CESM_PREFIX` | `f.e21.FWHIST.f09_f09_mg17...cam` | filename prefix inside that archive |
| `CESM_SUF` | `.1996010100-2014123100.nc` | filename suffix (the date range) |
| `TOMAS_JAX_PATH` | `../tomas-jax` | microphysics repo, else the normal import path |
| `RRTMGP_PATH` | `../jax-rrtmgp` | radiation repo, same order. Unused when `RAD=0` |
| `CARMA_FILE` | *(a site path)* | CARMA history file, only read when `AER_SRC=carma` |

A variable that is *set but points nowhere* is an error, not a silent fallback.

### Domain and resolution

| variable | default | meaning |
|---|---|---|
| `P_LO_HPA` | `1.0` | top of the band [hPa] |
| `P_HI_HPA` | `150.0` | bottom of the band [hPa] (module: `100.0`; `driver_fast.py` sets 150). 1–150 hPa = 24 native levels, 1,327,104 cells |
| `N_LEV` | `0` | `0` = every level in the band; else sub-sample to ~N levels |
| `N_BINS` | `0` | `0` = tomas-jax default (40). `driver_fast.py` accepts only `0` or `40` — the fast engine is hard-fixed at 40 |

### Initial and boundary conditions

| variable | default | meaning |
|---|---|---|
| `AER_SRC` | `mam4` | aerosol IC/BC/reservoir source: `mam4` (per-step dynamic from the hourly h1) or `carma` (static, only ~1 week of output exists). Gases are always CESM-forced |
| `INIT_BIN` | `so4` | bin MAM4 by `so4_a*` mass. `dgnum` is the legacy path that inflated sulfate mass 4.29× |
| `INIT_SIGMA` | *(unset)* | `s1,s2,s3` mode widths; unset = physical MAM4 widths |
| `BC_EDGE` | `open` | vertical faces: real flux boundaries. Derived — `clamp` when `ADV_WCONT=0` |
| `BC_GAS` | `flux` | SO2/H2SO4 edge treatment. **Derived from the resolved `BC_EDGE`** so gas and aerosol cannot desync. `clamp` reproduces pre-2026-07-30 runs |
| `BC_BOT_AER` | `1.0` | scale on the aerosol concentration flowing in through the bottom face (`0.0` = aerosol-free upwelling). Gases unaffected |
| `N_BC_TOP` | `1` | top band levels pinned to hourly MAM4 |
| `N_BC_BOT` | `1` | bottom band levels pinned to hourly MAM4 |
| `CARMA_FRAME` | `0` | time index into the CARMA file (48 frames) |
| `CARMA_RHO` | `1923.0` | CARMA sulfate mass-grid density [kg/m³] |
| `CARMA_SUBBIN` | `1` | sub-bin CARMA onto the TOMAS grid; `0` reproduces the pre-2026-07-26 "comb" |

### Transport

| variable | default | meaning |
|---|---|---|
| `ADV_SCHEME` | `lr` | `lr` = Lin-Rood flux form, conservative to roundoff. `fast` = advective form, ~3e-4/day residual, cheaper |
| `ADV_CFL` | `0.5` | CFL target for internal sub-stepping |
| `ADV_F32` | `1` | single precision. **Validate advection changes at `ADV_F32=1 ADV_CFL=0.5`** |
| `ADV_VPOS` | `1` | vertical positivity limiter. `0` is a forensic escape hatch, **not a supported configuration** — it reproduces a corrupt number field |
| `ADV_WCONT` | `1` | rederive omega from discrete continuity and open the vertical faces |
| `ADV_METRIC` | `1` | carry the cos(φ) area metric in the y-sweep |
| `ADV_DXFIX` | `1` | true grid spacing in the x-sweep |
| `ADV_POLAR` | `zonal` | mass-conserving well-mixed polar caps above \|lat\| 80°; anything else = legacy reservoir overwrite, which discards mass |
| `TRACER_CHUNK` | `0` | tracers advected per batch (`0` = all 82 at once). Lower it if advection OOMs |

### Microphysics and settling

| variable | default | meaning |
|---|---|---|
| `MICRO` | `full` | `full` = chemistry + nucleation + coagulation + condensation. `coag` = legacy coagulation only. `driver_fast.py` requires `full` |
| `MICRO_SUBSTEPS` | `6` | substeps per coupling step for the per-cell chain (1 h each) |
| `N_COAG_SUBSTEPS` | `3` | coagulation substeps |
| `COAG_MAX_SUBSTEPS` | `256` | ceiling on adaptive coag substeps. Critical for speed — a vmapped `while_loop` runs every lane to the slowest one |
| `CELL_CHUNK` | `300000` | cells per micro vmap batch (per-cell chain) |
| `ALPHA_COND` | `1.0` | H2SO4 accommodation coefficient |
| `NUC_ORG` | `1e7` | organic vapour [molec/cm³] for the nucleation scheme |
| `NUC_NH3` | `1e9` | NH3 [pptv] |
| `NUC_FION` | `3.0` | ion-pair production [cm⁻³ s⁻¹] |
| `NUC_FN_MAX` | `1e6` | cap on total nucleation rate [cm⁻³ s⁻¹]; a numerical guard against gas-clamped ultrafine bursts |
| `SETTLE` | `1` | gravitational settling, the model's only true aerosol sink |
| `WET_SETTLING` | `1` | size the settling particle as the wet H2SO4/H2O droplet; `0` restores dry-core sizing |

### Fast microphysics engine (`driver_fast.py`)

Memory/scheduling only — micro is per-cell independent, so chunk grouping cannot
change results.

| variable | default | meaning |
|---|---|---|
| `FAST_DT` | `360` | inner step [s]; 60 inner steps fill one 6 h coupling step |
| `FAST_CELL_CAP` | `50000` | max cells per chunk (module: `250000`). Lower is **faster** here, and fits alongside another job |
| `FAST_SORT` | `1` | stiffness-sort chunks, worth ~27% of micro. Does one unchunked ~8 GB allocation — **first thing to set to `0` if the card is loaded** |
| `FAST_FN_SCALE` | `1.0` | nucleation rate scale |
| `FAST_COAG_SUB_CAP` | `256` | coagulation substep cap |
| `FAST_COND_SUB_CAP` | `40` | condensation substep cap |
| `FAST_COAG_CMAX` | `0.05` | coagulation adaptive-step tolerance |

### OH chemistry

| variable | default | meaning |
|---|---|---|
| `OH_SZA` | `1` | diurnal OH parabola in cos(SZA) (Hanisco 2001). **Replaces** CESM's OH field — the step-mean magnitude changes too, not just the time resolution. `0` restores CESM OH held constant across the step |
| `OH_PEAK` | `2.3e6` | noon peak OH [molec/cm³] (the fit is least-squares, so peak OH is 4% above this) |
| `OH_SUBSTEPS` | *(per driver)* | samples of the curve per coupling step. `driver_fast.py` sets it to its inner-step count (60 at `FAST_DT=360`); the per-cell chain uses `MICRO_SUBSTEPS` |

### Radiation

| variable | default | meaning |
|---|---|---|
| `RAD` | `1` | `0` skips radiation entirely — and with it the `jax-rrtmgp` dependency |
| `RAD_MODE` | `anomaly` | `anomaly` = heating relative to the reference MAM4 bins; `full` = absolute (drifts toward radiative equilibrium). Under `AER_SRC=mam4` the anomaly baseline is **time-varying** |
| `RAD_EVERY` | `1` | coupling steps between radiation calls |
| `ARF_AVG_H` | `24` | trailing window [h] for the *reported* forcing; 24 h = 4 uniformly spaced local times = a proper diurnal mean. `0` disables (the raw instantaneous value is always stored) |
| `WET_OPTICS` | `1` | Mie on the wet droplet. Dry sizing underestimates 550 nm extinction by ~47%. Keep in step with `WET_SETTLING` |
| `RAD_LAT_CHUNK` | `96` | latitude rows per RRTMGP chunk; bit-identical either way. Drop to 48 if radiation OOMs |
| `CO2_PPM` | `380.0` | background CO2 (~2005, for 1996–2014 forcing) |
| `N2O_PPB` | `319.0` | background N2O |

### GPU and process environment (`run_prod.sh`)

| variable | default | meaning |
|---|---|---|
| `GPU` | `0` | which card to pin (`CUDA_VISIBLE_DEVICES`). Microphysics shards across all *visible* devices, so exporting more than one is what makes it multi-GPU |
| `CUDA_DRIVER_LIB` | `/run/nvidia/driver/usr/lib/x86_64-linux-gnu` | where `libcuda.so.1` lives. Without it **JAX silently falls back to CPU**. Set it empty for a normal CUDA install |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | `false` | grow GPU memory on demand instead of preallocating 75% of the card |

### Debug-only

| variable | default | meaning |
|---|---|---|
| `DUMP_PREMICRO` | *(unset)* | dump the pre-microphysics state for one step |
| `DUMP_PREMICRO_STEP` | `0` | which step to dump |

### Resuming across a changed configuration

`INJ_*` mismatches against the checkpoint are **refused**. The physics-mode flags
`WET_SETTLING` / `WET_OPTICS` / `SETTLE` / `ADV_VPOS` only **warn** — the state is
still valid, the model integrating it forward just changes at the seam. Both
arrays are append-only, so adding a field never locks out an older checkpoint.

Every default above, and the reasoning behind it, is in
**[MANIFEST.md](./MANIFEST.md) — the canonical reference for this tree.** Where
`docs/` disagrees with it, MANIFEST is current.

## Running Tests

```bash
python3 validation/test_conservation.py                        # advection conservation
ADV_F32=1 ADV_CFL=0.5 python3 validation/validate_vpos_f32.py  # positivity limiter
```

Run them from the repo root. Validate advection changes at `ADV_F32=1
ADV_CFL=0.5`, the production precision — **not** the f64/cfl=0.2 module
defaults: two positivity-limiter bugs were invisible in f64 and fatal in f32.

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
