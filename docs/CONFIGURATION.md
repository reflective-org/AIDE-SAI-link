# Configuration reference

Every knob in this model is an environment variable. This page is the complete
list; [`../README.md`](../README.md) carries the five you need for a first run.

> **[`../MANIFEST.md`](../MANIFEST.md) is canonical for this tree.** It records
> every default *and why it is what it is*. This page tells you what the knobs
> are; MANIFEST tells you why. Where the two disagree, MANIFEST is current.

Each variable is read **once at module import** and echoed in the run header, so
any log is self-describing. Set them by prefixing the launcher:

```bash
REPO=$PWD/AIDE-SAI-link                    # wherever this clone lives -- must be ABSOLUTE, the next line cd's away
cd "$PWD/sai_runs"                         # launch from a runs dir, NOT the repo
OUT_TAG=inj20_30N INJ_SO2_TG_YR=20 INJ_LAT=30 INJ_MIRROR=1 $REPO/run_prod.sh
```

Anything not listed in `run_prod.sh`'s own prefix block simply passes through
from your environment — `AER_SRC=carma $REPO/run_prod.sh` works. The exception:

> [!IMPORTANT]
> `run_prod.sh` **hard-sets** `INIT_BIN=so4`, `STATE_CKPT=1`, `FRAME_EVERY=24`,
> `FAST_CELL_CAP=50000`, `ADV_VPOS=1`, `DEBUG=1`, `PROFILE=1`. Passing those on
> the command line is silently ignored. Override them by editing the script, or
> run `driver_fast.py` directly. The remaining launcher variables —
> `N_HOURS`, `OUT_TAG`, `RESUME`, all `INJ_*`, `FAST_SORT`, `GPU`,
> `CUDA_DRIVER_LIB`, `XLA_PYTHON_CLIENT_PREALLOCATE` — are overridable.

The **Default** column below is the *effective* production default, i.e. what
you get from `run_prod.sh`; where the bare `coupling.py` module default differs
it is noted.

## Which flags actually apply to my run?

Most of this page is inert for any given run. Two things decide which part is
live: **which entry point you launched** and **which options you switched on**.

`run_prod.sh` execs `driver_fast.py`, which imports `coupling.py` and then
*replaces* its microphysics with the batched `tomas_jax.fast` engine. So the
per-cell chain's own microphysics knobs are never reached on the production
path — the `FAST_*` ones take their place. Everything else (injection, run
length, domain, IC/BC, transport, settling, radiation) is shared, because both
paths run the same code for it.

| you launched | microphysics knobs that are live | the ones that do nothing |
|---|---|---|
| `run_prod.sh` or `python3 driver_fast.py` — **the normal case** | `ALPHA_COND`, `FAST_DT`, `FAST_CELL_CAP`, `FAST_SORT`, `FAST_FN_SCALE`, `FAST_COAG_SUB_CAP`, `FAST_COND_SUB_CAP`, `FAST_COAG_CMAX` | `MICRO_SUBSTEPS`, `N_COAG_SUBSTEPS`, `COAG_MAX_SUBSTEPS`, `CELL_CHUNK`, all `NUC_*` |
| `python3 coupling.py` — standalone/dev only | `MICRO_SUBSTEPS`, `COAG_MAX_SUBSTEPS`, `CELL_CHUNK`, `ALPHA_COND`, `NUC_ORG`, `NUC_NH3`, `NUC_FION`, `NUC_FN_MAX` | every `FAST_*` |

The two engines also differ *physically*, not just in speed: the fast engine's
nucleation is **binary** (H2SO4–H2O only), which is why it has no `NUC_ORG` /
`NUC_NH3` / `NUC_FION` to set — `FAST_FN_SCALE` is its only nucleation dial.

Rows below marked **[chain]** are read only by the standalone `coupling.py`
path and are ignored under `run_prod.sh`.

Beyond the entry point, four switches gate whole blocks of knobs:

| switch | at this setting | these stop mattering |
|---|---|---|
| `AER_SRC` | `mam4` (the default) | all `CARMA_*`, and `CARMA_FILE` |
| `RAD` | `0` | the whole Radiation section, and `RRTMGP_PATH` |
| `SETTLE` | `0` | `WET_SETTLING` |
| `INJ_SO2_TG_YR` / `INJ_H2SO4_TG_YR` | both `0` (a control run) | the geometry knobs `INJ_HPA`, `INJ_LAT`, `INJ_LON`, `INJ_ZONAL`, `INJ_MIRROR` |

## Injection scenario — the knobs meant to change run to run

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

## Run length, output and restart

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

## Input data and dependency paths

| variable | default | meaning |
|---|---|---|
| `CESM_DIR` | *(the FWHIST archive these results were made from)* | root of the CESM tseries archive; layout `$CESM_DIR/hour_1/$CESM_PREFIX.h1.<VAR>$CESM_SUF` |
| `CESM_PREFIX` | `f.e21.FWHIST.f09_f09_mg17...cam` | filename prefix inside that archive |
| `CESM_SUF` | `.1996010100-2014123100.nc` | filename suffix (the date range) |
| `TOMAS_JAX_PATH` | `../tomas-jax` | microphysics repo, else the normal import path |
| `RRTMGP_PATH` | `../jax-rrtmgp` | radiation repo, same order. Unused when `RAD=0` |
| `CARMA_FILE` | *(a site path)* | CARMA history file, only read when `AER_SRC=carma` |

A variable that is *set but points nowhere* is an error, not a silent fallback.

## Domain and resolution

| variable | default | meaning |
|---|---|---|
| `P_LO_HPA` | `1.0` | top of the band [hPa] |
| `P_HI_HPA` | `150.0` | bottom of the band [hPa] (module: `100.0`; `driver_fast.py` sets 150). 1–150 hPa = 24 native levels, 1,327,104 cells |
| `N_LEV` | `0` | `0` = every level in the band; else sub-sample to ~N levels |
| `N_BINS` | `0` | `0` = tomas-jax default (40). `driver_fast.py` accepts only `0` or `40` — the fast engine is hard-fixed at 40 |

## Initial and boundary conditions

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

More detail: [BOUNDARY_CONDITIONS.md](./BOUNDARY_CONDITIONS.md).

## Transport

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

## Microphysics and settling

Under `run_prod.sh` only `MICRO`, `ALPHA_COND`, `SETTLE` and `WET_SETTLING` are
live here; the **[chain]** rows are the standalone `coupling.py` engine's and
are ignored. Their fast-engine counterparts are in the next section.

| variable | default | meaning |
|---|---|---|
| `MICRO` | `full` | `full` = chemistry + nucleation + coagulation + condensation. `coag` = legacy coagulation only. `driver_fast.py` requires `full` |
| `ALPHA_COND` | `1.0` | H2SO4 accommodation (sticking) coefficient — the fraction of vapour–particle collisions that actually condense. `1.0` = every collision sticks, the fastest condensation physically allowed. **Read by both engines** |
| `SETTLE` | `1` | gravitational settling, the model's only true aerosol sink |
| `WET_SETTLING` | `1` | size the settling particle as the wet H2SO4/H2O droplet; `0` restores dry-core sizing. Ignored when `SETTLE=0` |
| `MICRO_SUBSTEPS` | `6` | **[chain]** substeps per coupling step (1 h each); an accuracy dial, not a stability requirement. The fast engine uses `FAST_DT` instead |
| `COAG_MAX_SUBSTEPS` | `256` | **[chain]** ceiling on the *adaptive* coagulation solver's substeps. A speed knob: the vmapped `while_loop` runs every lane to the slowest one. A cell that hits the cap is returned **partially integrated** and logs a `coag substep cap hit` warning |
| `CELL_CHUNK` | `300000` | **[chain]** cells per micro vmap batch — peak GPU memory vs. call overhead only. Micro is per-cell independent, so results are identical at any value |
| `N_COAG_SUBSTEPS` | `3` | **[chain]** fixed forward-Euler substeps for the legacy `MICRO=coag` path only; dead under `MICRO=full` |
| `NUC_ORG` | `1e7` | **[chain]** background organic vapour [molec/cm³] fed to the ternary rate. A boundary-layer value inherited from the tomas-jax SAI box model — retune before reading anything into it |
| `NUC_NH3` | `1e9` | **[chain]** NH3 [pptv]; same caveat |
| `NUC_FION` | `3.0` | **[chain]** ion-pair production [cm⁻³ s⁻¹] |
| `NUC_FN_MAX` | `1e6` | **[chain]** cap on total nucleation rate [cm⁻³ s⁻¹]; a numerical guard against gas-clamped ultrafine bursts |

## Fast microphysics engine (`driver_fast.py`) — the production microphysics

**This is the section that applies to any run started with `run_prod.sh`.**
Apart from `FAST_FN_SCALE`, these are memory/scheduling knobs: micro is per-cell
independent, so how cells are grouped into chunks cannot change results.

| variable | default | meaning |
|---|---|---|
| `FAST_DT` | `360` | inner step [s]; 60 inner steps fill one 6 h coupling step. The fast engine's counterpart to `MICRO_SUBSTEPS` |
| `FAST_CELL_CAP` | `50000` | max cells per chunk (module: `250000`). Lower is **faster** here, and fits alongside another job. Counterpart to `CELL_CHUNK` |
| `FAST_SORT` | `1` | stiffness-sort chunks, worth ~27% of micro. Does one unchunked ~8 GB allocation — **first thing to set to `0` if the card is loaded** |
| `FAST_FN_SCALE` | `1.0` | nucleation rate scale — the **only** nucleation knob on this path, since the scheme is binary (no organics/NH3/ions to set) |
| `FAST_COAG_SUB_CAP` | `256` | coagulation substep cap. Counterpart to `COAG_MAX_SUBSTEPS` |
| `FAST_COND_SUB_CAP` | `40` | condensation substep cap |
| `FAST_COAG_CMAX` | `0.05` | coagulation adaptive-step tolerance — how much error per step is accepted, i.e. how many substeps you actually spend under the cap |

## OH chemistry

| variable | default | meaning |
|---|---|---|
| `OH_SZA` | `1` | diurnal OH parabola in cos(SZA) (Hanisco 2001). **Replaces** CESM's OH field — the step-mean magnitude changes too, not just the time resolution. `0` restores CESM OH held constant across the step |
| `OH_PEAK` | `2.3e6` | noon peak OH [molec/cm³] (the fit is least-squares, so peak OH is 4% above this) |
| `OH_SUBSTEPS` | *(per driver)* | samples of the curve per coupling step. `driver_fast.py` sets it to its inner-step count (60 at `FAST_DT=360`); the per-cell chain uses `MICRO_SUBSTEPS` |

## Radiation

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

## GPU and process environment (`run_prod.sh`)

| variable | default | meaning |
|---|---|---|
| `GPU` | `0` | which card to pin (`CUDA_VISIBLE_DEVICES`). Microphysics shards across all *visible* devices, so exporting more than one is what makes it multi-GPU |
| `CUDA_DRIVER_LIB` | `/run/nvidia/driver/usr/lib/x86_64-linux-gnu` | where `libcuda.so.1` lives. Without it **JAX silently falls back to CPU**. Set it empty for a normal CUDA install |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | `false` | grow GPU memory on demand instead of preallocating 75% of the card |

## Debug-only

| variable | default | meaning |
|---|---|---|
| `DUMP_PREMICRO` | *(unset)* | dump the pre-microphysics state for one step |
| `DUMP_PREMICRO_STEP` | `0` | which step to dump |

## Resuming across a changed configuration

`INJ_*` mismatches against the checkpoint are **refused**. The physics-mode flags
`WET_SETTLING` / `WET_OPTICS` / `SETTLE` / `ADV_VPOS` only **warn** — the state is
still valid, the model integrating it forward just changes at the seam. Both
arrays are append-only, so adding a field never locks out an older checkpoint.
