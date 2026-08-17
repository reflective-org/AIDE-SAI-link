# Configuration reference

Every knob in this model is an environment variable. This page is the complete
list; [`../README.md`](../README.md) carries the recipe you actually run — on the
`advection-mip` branch that is the advection-only experiment
(`DRIVER=coupling.py MICRO=off RAD=0 AER_SRC=fixed`, settling on or off), and the
sections below marked *transport-only* are the ones it reaches.

> **[`../MANIFEST.md`](../MANIFEST.md) is canonical for this tree.** It records
> every default *and why it is what it is*. This page tells you what the knobs
> are; MANIFEST tells you why. Where the two disagree, MANIFEST is current.

Each variable is read **once at module import** and echoed in the run header, so
any log is self-describing. Set them by prefixing the launcher:

```bash
REPO=$PWD/aide_sai_core                    # wherever this clone lives -- must be ABSOLUTE, the next line cd's away
cd "$PWD/sai_runs"                         # launch from a runs dir, NOT the repo
OUT_TAG=inj20_30N INJ_SO2_TG_YR=20 INJ_LAT=30 INJ_MIRROR=1 $REPO/run_prod.sh
```

Anything not listed in `run_prod.sh`'s own prefix block simply passes through
from your environment — `AER_SRC=carma $REPO/run_prod.sh` works. The exception:

> [!IMPORTANT]
> `run_prod.sh` **hard-sets** `INIT_BIN=so4`, `STATE_CKPT=1`,
> `FAST_CELL_CAP=50000`, `ADV_VPOS=1`, `DEBUG=1`, `PROFILE=1`. Passing those on
> the command line is silently ignored. Override them by editing the script, or
> run `driver_fast.py` directly. The remaining launcher variables —
> `N_HOURS`, `P_LO_HPA`, `P_HI_HPA`, `FRAME_EVERY`, `OUT_TAG`, `RESUME`, all
> `INJ_*`, `FAST_SORT`, `GPU`, `CUDA_DRIVER_LIB`,
> `XLA_PYTHON_CLIENT_PREALLOCATE` — are overridable. `FRAME_EVERY` became
> overridable on 2026-08-14 and **must** be raised for runs much longer than 90
> days: the whole frames history is rewritten at every frame, so I/O cost grows
> as (frames)².

The **Default** column below is the *effective* production default, i.e. what
you get from `run_prod.sh`; where the bare `coupling.py` module default differs
it is noted.

## Which flags actually apply to my run?

Most of this page is inert for any given run. Two things decide which part is
live: **which entry point you launched** and **which options you switched on**.

`run_prod.sh` execs whatever `DRIVER` names, `driver_fast.py` by default, which
imports `coupling.py` and then *replaces* its microphysics with the batched
`tomas_jax.fast` engine. So the per-cell chain's own microphysics knobs are never
reached on the production path — the `FAST_*` ones take their place. Everything
else (injection, run length, domain, IC/BC, transport, settling, radiation) is
shared, because both paths run the same code for it.

| you launched | microphysics knobs that are live | the ones that do nothing |
|---|---|---|
| `run_prod.sh` or `python3 driver_fast.py` — **the normal case** | `ALPHA_COND`, `FAST_DT`, `FAST_CELL_CAP`, `FAST_SORT`, `FAST_FN_SCALE`, `FAST_COAG_SUB_CAP`, `FAST_COND_SUB_CAP`, `FAST_COAG_CMAX` | `MICRO_SUBSTEPS`, `N_COAG_SUBSTEPS`, `COAG_MAX_SUBSTEPS`, `CELL_CHUNK`, all `NUC_*` |
| `DRIVER=coupling.py run_prod.sh`, or `python3 coupling.py` — the standalone chain, and **the only path that runs `MICRO=off`** | `MICRO_SUBSTEPS`, `COAG_MAX_SUBSTEPS`, `CELL_CHUNK`, `ALPHA_COND`, `NUC_ORG`, `NUC_NH3`, `NUC_FION`, `NUC_FN_MAX` | every `FAST_*` |

At `MICRO=off` — the advection-only experiment — *neither* column is live: there
is no microphysics call for any of it to configure.

The two engines also differ *physically*, not just in speed: the fast engine's
nucleation is **binary** (H2SO4–H2O only), which is why it has no `NUC_ORG` /
`NUC_NH3` / `NUC_FION` to set — `FAST_FN_SCALE` is its only nucleation dial.

Rows below marked **[chain]** are read only by the per-cell `coupling.py` engine
— i.e. ignored whenever `driver_fast.py` is the driver, which is `run_prod.sh`'s
default but not what `DRIVER=coupling.py` selects.

Beyond the entry point, these switches gate whole blocks of knobs:

| switch | at this setting | these stop mattering |
|---|---|---|
| `AER_SRC` | `mam4` (the default) | all `CARMA_*`, and `CARMA_FILE` |
| `AER_SRC` | `fixed` | all `CARMA_*`, `INIT_BIN`, `INIT_SIGMA` — no CESM aerosol is read at all. The `FIXED_*` knobs take their place |
| `MICRO` | `off` | every microphysics knob, `FAST_*` included, plus `ALPHA_COND`, all `NUC_*` and all `OH_*` — the bins become independent passive tracers |
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
| `DRIVER` | `driver_fast.py` | which entry point `run_prod.sh` execs, from its own tree. **`coupling.py` is required at `MICRO=off`** — the fast driver *is* a microphysics engine and exits when there is none to run. One launcher either way, so the GPU pin, libcuda path, memory policy and repo-as-`$PWD` guard cannot drift between the two |
| `N_HOURS` | `2160` | forcing hours to integrate (module: `24*N_DAYS` = 48) |
| `N_DAYS` | `2` | fallback that supplies `N_HOURS` when it is unset |
| `H0` | `0` | start hour index into the CESM h1 series |
| `STEP_HOURS` | `6` | coupling step [h]. `driver_fast.py` **requires 6** and exits otherwise |
| `OUT_TAG` | `prod90d` | names every output file (module: `<N_DAYS>day`; `driver_fast.py` alone: `tomas_fast`) |
| `RESUME` | `0` | `1` = continue from `coupled_state_<TAG>_ckpt.npz` |
| `STATE_CKPT` | `1` | write the restart checkpoint (~400 MB, atomic, overwritten in place) |
| `FRAME_EVERY` | `24` | hours between frame + checkpoint writes. Raise it for long runs: the whole frames history is rewritten every time, so I/O cost grows as (frames)² — `120` for multi-year |
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
| `P_LO_HPA` | `1.0` | top of the band [hPa]. The advection-only runs use **`0.03`** (33 levels): at 1 hPa, 6.9% of the domain's air descends through the top face per 90 days carrying `q=0`, which dilutes without appearing in any mass term — see MANIFEST |
| `P_HI_HPA` | `150.0` | bottom of the band [hPa] (module: `100.0`; `driver_fast.py` sets 150). 1–150 hPa = 24 native levels, 1,327,104 cells; the floor lands on the 143 hPa level |
| `N_LEV` | `0` | `0` = every level in the band; else sub-sample to ~N levels |
| `N_BINS` | `0` | `0` = tomas-jax default (40); a smaller value keeps the same physical size range, just coarser. `driver_fast.py` accepts only `0` or `40` — the fast engine is hard-fixed at 40. **`1` is valid only at `MICRO=off` *and* `SETTLE=0`**, where the bins are identical passive tracers; with settling on it collapses the fall-speed spectrum, and with one bin `Dp(M/N)` in the log is meaningless |

## Initial and boundary conditions

| variable | default | meaning |
|---|---|---|
| `AER_SRC` | `mam4` | aerosol IC/BC/reservoir source: `mam4` (per-step dynamic from the hourly h1), `carma` (static, only ~1 week of output exists), or `fixed` (a prescribed uniform, time-invariant PSD — no CESM aerosol at all; the *transport-only* source, see below). Gases are always CESM-forced |
| `INIT_BIN` | `so4` | bin MAM4 by `so4_a*` mass. `dgnum` is the legacy path that inflated sulfate mass 4.29× |
| `INIT_SIGMA` | *(unset)* | `s1,s2,s3` mode widths; unset = physical MAM4 widths |
| `BC_EDGE` | `open` | vertical faces: real flux boundaries. Derived — `clamp` when `ADV_WCONT=0` |
| `BC_GAS` | `flux` | SO2/H2SO4 edge treatment. **Derived from the resolved `BC_EDGE`** so gas and aerosol cannot desync. `clamp` reproduces pre-2026-07-30 runs |
| `BC_BOT_AER` | `1.0` | scale on the aerosol concentration flowing in through the bottom face (`0.0` = aerosol-free upwelling). Gases unaffected |
| `BC_TOP_AER` | `1.0` | the same, for the top face. `0.0` = aerosol-free air descending in. **Both at `0.0` is the advection-only setting**: nothing is re-injected, so the band drains and the burden is a closed budget minus what leaves. Outflow is always free either way |
| `N_BC_TOP` | `1` | top band levels pinned to hourly MAM4 |
| `N_BC_BOT` | `1` | bottom band levels pinned to hourly MAM4 |
| `CARMA_FRAME` | `0` | time index into the CARMA file (48 frames) |
| `CARMA_RHO` | `1923.0` | CARMA sulfate mass-grid density [kg/m³] |
| `CARMA_SUBBIN` | `1` | sub-bin CARMA onto the TOMAS grid; `0` reproduces the pre-2026-07-26 "comb" |

More detail: [BOUNDARY_CONDITIONS.md](./BOUNDARY_CONDITIONS.md).

### Prescribed fixed PSD (`AER_SRC=fixed`) — *transport-only*

One particle size distribution, identical in every cell of the band and constant
in time, advected and settled with no microphysics and no radiation. Read only
when `AER_SRC=fixed`, and stamped into every output `.npz` so a file states the
PSD it was run with. A uniform mixing ratio is an **exact steady state of the
advection operator alone**, which is what makes every departure attributable —
see MANIFEST for why, and why the absolute scale does not matter.

| variable | default | meaning |
|---|---|---|
| `FIXED_PSD` | `lognormal` | `lognormal` = `FIXED_N` spread over the bins by a lognormal in ln(D), through the same `bin_mode_dgnum` kernel the MAM4 path uses. `flat` = `FIXED_N`/NBINS in **every** bin — deliberately unphysical, for the drainage-vs-size curve |
| `FIXED_N` | `1.0e8` | total number mixing ratio over all bins [#/kg]. Scale-free: with no microphysics the system is linear in the aerosol. ~8 #/cm³ at 50 hPa/210 K |
| `FIXED_DG_NM` | `200.0` | number-median **dry** diameter [nm]. The knob that matters — settling goes as D² |
| `FIXED_SIGMA` | `1.6` | geometric width of the `lognormal` shape |
| `FIXED_P_LO_HPA` | `0.0` | top of the pressure window the background is placed in [hPa], intersected with the band |
| `FIXED_P_HI_HPA` | `1e9` | bottom of that window [hPa]; the default pair = the whole band. Narrowing it (e.g. `FIXED_P_HI_HPA=30`) starts the run with a real vertical gradient for the winds to act on **immediately**, instead of one settling has to create first — under a uniform IC the winds only enter at second order in time |
| `FIXED_LAT_MAX_DEG` | `91.0` | latitude half-width of the window; `91` = every row. Set it (with the pressure window) to make the run a **tagged pulse** instead of a drainage run — `run_pulse_bdc.sh` uses `15` |

The window narrows the **face-inflow reservoir** as well, since both come from
`aer_fill`. That is only safe because `BC_TOP_AER`/`BC_BOT_AER` scale it
afterwards: with both `0` the faces carry no aerosol either way. With
`BC_*_AER != 0` a windowed reservoir means "aerosol-free inflow outside the
window", a different boundary condition from the uniform run's — say so if you
use that combination.

A window that selects no level of the band is an error, not an empty run. The
run header echoes `==> TRANSPORT-ONLY run` only when microphysics, radiation,
injection and the CESM aerosol source are *all* off, and names whatever is still
active otherwise — `AER_SRC=fixed` with radiation on is a legitimate run, but it
is not the comparison experiment, and the output files look identical.

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

Under `run_prod.sh` with its default driver only `MICRO`, `ALPHA_COND`, `SETTLE`
and `WET_SETTLING` are live here; the **[chain]** rows are the per-cell
`coupling.py` engine's and are ignored. Their fast-engine counterparts are in the
next section. At `MICRO=off` only `SETTLE` and `WET_SETTLING` do anything at all.

| variable | default | meaning |
|---|---|---|
| `MICRO` | `full` | `full` = chemistry + nucleation + coagulation + condensation. `coag` = legacy coagulation only. `off` = **advect (+ settle) only**: the bins become independent passive tracers and every knob in this section but `SETTLE`/`WET_SETTLING` goes inert. `driver_fast.py` requires `full` — use `DRIVER=coupling.py` for `off` |
| `ALPHA_COND` | `1.0` | H2SO4 accommodation (sticking) coefficient — the fraction of vapour–particle collisions that actually condense. `1.0` = every collision sticks, the fastest condensation physically allowed. **Read by both engines** |
| `SETTLE` | `1` | gravitational settling. With the faces serving a reservoir it is the model's only true aerosol sink; with them aerosol-free (`BC_*_AER=0`) it is one of two channels out, the advective flux through the same face being the other |
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
