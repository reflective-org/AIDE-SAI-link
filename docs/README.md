# CESM → TOMAS aerosol model — design rationale

> **`MANIFEST.md` in the repo root is canonical for the current tree**, and
> `../README.md` is the orientation. This file is kept for the *design rationale*
> behind the grid, the MAM4 initialization and the open boundaries — the parts
> that have not changed. Where it disagrees with MANIFEST, MANIFEST is current.
> Numbers here were re-checked against the tree on 2026-08-12.

Combines flux-form advection with the sectional microphysics from `models/tomas-jax`,
driven **one-way** by CESM meteorology — one-way meaning *to CESM*: the winds are
prescribed and the circulation never responds. The aerosol–radiation–microphysics
loop inside that forcing is closed (see [Not included](#not-included-the-one-way-scope)).
Production transport is the Lin-Rood scheme in
`src/advection/fct_lr.py`; the older PPM+FCT `fct.py` is the lineage and the
source of the pressure convention described below.

```
CESM h1 hourly fields ──► winds U,V,OMEGA ──► transport (Lin-Rood flux form)
                     ├───► T, p, RELHUM    ──► microphysics (TOMAS, 40 bins)
                     └───► SO2, H2SO4, OH  ──► gas-phase chemistry
```

## Files
| File | Role |
|------|------|
| `coupling.py`  | The model: load CESM, init from MAM4, advect+microphysics loop, save. Runnable standalone. |
| `driver_fast.py` | **The production entry point** — imports `coupling.py` and swaps in the batched `tomas_jax.fast` engine. Launched by `../src/run_prod.sh`. |
| `src/advection/fct_lr.py` | Lin-Rood flux-form advection — the production transport scheme. |
| `src/advection/fct_fast.py` | PPM/Zalesak primitives that `fct_lr` imports. |
| `fct_core.py`  | Legacy sealed-face PPM/FCT transport (from `fct.py`). **Not on the run path** as of 2026-08-03; kept only for the bit-identical legacy check in `scripts/validation/test_conservation.py`. |
| `settling.py`  | Gravitational settling sink. |
| `radiation.py` | RRTMGP + Mie optics. |
| `scripts/utils/plot_run.py`  | Diagnostics: dashboard, filmstrip, size distribution. (Replaced `plot_coupled.py`, `plot_size_dist.py` and `viz_coupled_month.py`, all deleted 2026-08-03.) |
| `scripts/utils/gif_run.py`   | Animated versions of the filmstrip panels. |

## State (what is tracked & advected)
Two moments per size bin — no per-species aerosol chemistry:
- `num[bin]` — number mixing ratio `[#/kg air]`
- `mas[bin]` — total dry-mass mixing ratio `[kg/kg air]`

plus two gas tracers, `SO2` and `H2SO4`.

40 TOMAS bins ⇒ 80 aerosol fields + 2 gases = **82 advected 3-D tracer fields**
on the native f09 grid (192×288) over the stratospheric band (**1–150 hPa ⇒ 24
model levels**, 1,327,104 cells). The band is set by `P_LO_HPA` / `P_HI_HPA`.

## Method / assumptions
1. **Grid & pressure.** Native CESM grid; level pressures use the reference-`PS`
   convention from `fct.py` (`p = hyam·P0 + hybm·PS_REF`), constant in time.
   Everything (winds, T, MAM4) is on the same `h1` 70-level grid, so no
   wind/tracer level matching is needed (unlike `fct.py`).
2. **Initialization from MAM4.** CESM modal aerosol `num_a{1,2,3}` (#/kg) and
   `so4_a{1,2,3}` (kg/kg) at the start hour are binned onto the 40-bin grid via
   a per-mode dry log-normal (mode σ_g = 1.8 / 1.6 / 1.8 for accum/Aitken/coarse).
   Per-mode arithmetic mean particle mass = `so4/num`; number is distributed by
   the log-normal CDF in log-mass; per-bin mass = number × bin geometric-mean
   mass (single fixed density). Number is conserved exactly; mass to ~2%.
   *Coarse-mode dust/sea-salt mass is not carried separately — only sulfate;
   all mass is lumped as one dry-mass moment.*
3. **Transport.** All 82 tracers advect together each step with the Lin-Rood
   flux-form scheme (`fct_lr.py`; the PPM + Zalesak-FCT `fct` core is the
   lineage), winds linearly interpolated across the step, CFL-limited vertical
   sub-stepping. Polar caps (|lat|>80°) are stirred to one well-mixed cell per
   level, mass-conservingly (`ADV_POLAR=zonal`, the default) rather than
   overwritten from MAM4 — the old reservoir overwrite discarded mass. Batched
   with `jax.vmap` over the tracer axis; shared winds, so the substep count is
   common to all and the batch is exact.
4. **Microphysics.** Brownian coagulation (`tomas_jax` forward-Euler + MNFIX),
   `vmap`ped over all grid cells in memory-bounded chunks. Mixing ratios are
   converted to per-m³ concentrations with local air density `ρ = p/(Rd·T)`
   before coagulating, then back. The unmodified TOMAS kernel needs a 44-wide
   `Mk` only for density, so total mass rides the SO4 slot ⇒ **constant
   ~1770 kg/m³ (pure sulfate) density**, a single fixed mass↔size relation.
5. **Two-moment consistency.** Because `num` and `mas` advect independently, a
   bin can arrive with `Nk>0, mass≈0`. Before coagulating, each bin's mean
   particle mass is clipped into `[xk_k, xk_{k+1}]` (empty bins → 0), keeping
   `Dp` finite and well posed. The clip is *not* mass-conserving, so its
   activity is monitored: the hourly log prints `clipM/M0 +add/-remove` for
   the current hour, and the timeseries saves cumulative totals
   (`clipMadd_cum`/`clipMrem_cum`, burden-weighted; `M0` included for
   normalization). These should stay ≪ M0 — growth means the limiter is
   decoupling the two moments faster than the physics justifies.
6. **Open vertical boundaries** (same rationale as `../advection/fct_openbc.py`).
   The top `N_BC_TOP` and bottom `N_BC_BOT` band levels are reset every hour to
   hourly CESM MAM4 binned onto the TOMAS grid. These Dirichlet reservoirs carry
   the net effect of all physics outside the band (emissions, wet removal, and
   everything below the tropopause), making the band a flux-through system
   rather than a sealed one. Number and mass are always pinned as a
   consistent `(Nk, Mk)` pair from one binning — never rescaled separately.
   **No global mass fixer**: with open boundaries the burden legitimately
   changes, and forcing `M0` would cancel the flux-through.

## SAI extension (sources & sinks, 2026-07-16)
The model now carries the full **SO2 → H2SO4 → SO4** chain and a
**gravitational settling** sink (see the `coupling.py` docstring for details):
- **+2 gas tracers** (`so2`, `h2so4` mass mixing ratios ⇒ 82 advected fields),
  IC/open-BC/polar-refreshed from CESM's own `SO2`/`H2SO4` h1 fields.
- **Injection**: continuous SO2 release (`INJ_SO2_TG_YR` at
  `INJ_LAT`/`INJ_LON`/`INJ_HPA`; `INJ_ZONAL=1` spreads it around the lat ring).
- **Full microphysics** (`MICRO=full`, default): per cell, the tomas-jax chain
  SO2 chemistry → nucleation → **adaptive** coagulation (`euler_step`, the
  FORTRAN-equivalent integrator — the fixed-substep coag in `make_step` is
  unstable after nucleation bursts at coupled-model dt) → condensation,
  forced by CESM `OH` (mol/mol → molec/cm³) and `RELHUM`. Stable at any
  `MICRO_SUBSTEPS`; plume-cell *number* only converges near dt_sub ≈ 60–120 s
  (accuracy dial, not a stability cliff). `MICRO=coag` reproduces the legacy
  coagulation-only model exactly.
- **Settling** (`settling.py`): per-bin slip-corrected Stokes velocity,
  implicit upwind column sweep, **open bottom face** — mass crossing the
  lowest level exits the model (the aerosol's one true sink). The staged
  budget gains a `settle` stage equal to −(bottom outflow).
- **CESM-free boundary plan** for emulator-driven multi-year runs:
  `BOUNDARY_CONDITIONS.md`.

## Run
Production runs go through `../src/run_prod.sh` (which execs `driver_fast.py`), not
these commands — see MANIFEST.md. Direct `coupling.py` invocation is the
standalone/dev path:

```bash
# short validation (default 2 days, hourly), single GPU
N_DAYS=2 OUT_TAG=2day CUDA_VISIBLE_DEVICES=0 python3 src/coupling.py
python3 scripts/utils/plot_run.py 2day

# quick smoke test
N_HOURS=2 OUT_TAG=smoke python3 src/coupling.py

# scale up
N_DAYS=365 OUT_TAG=1yr python3 src/coupling.py
```

### Environment knobs

> These are the **`coupling.py` module defaults** — what you get running
> `python3 src/coupling.py` directly. They are NOT what a production run uses:
> `run_prod.sh` overrides or hard-sets many of them (`OUT_TAG`, `DEBUG`,
> `PROFILE`, `FAST_CELL_CAP`, …). For the effective production values, and for
> the knobs this table does not list, see
> [CONFIGURATION.md](./CONFIGURATION.md), which is authoritative.

| Var | Default | Meaning |
|-----|---------|---------|
| `N_DAYS` / `N_HOURS` | 2 / — | run length (`N_HOURS` overrides) |
| `N_LEV` | 0 (full band) | sub-sample the band to ~N_LEV native levels (e.g. `17` to mimic PARADIS) |
| `H0` | 0 | start hour index into the h1 series (1996-01-01 00Z) |
| `N_COAG_SUBSTEPS` | 3 | forward-Euler substeps per hour (legacy `MICRO=coag` path) |
| `CELL_CHUNK` | 300000 | cells per microphysics vmap batch (GPU memory vs speed) |
| `N_BC_TOP` | 1 | top band levels pinned to hourly MAM4 (open BC) |
| `N_BC_BOT` | 1 | bottom band levels pinned to hourly MAM4 (open BC) |
| `PROBE_HPA` | 50 | level [hPa] used for diagnostics, frames, mean-Dp |
| `LOG_EVERY` | 1 | print a progress line every N simulated hours |
| `FRAME_EVERY` | 24 | save a probe-level size-bin snapshot every N hours |
| `OUT_TAG` | `<N_DAYS>day` | output filename tag |
| `PROFILE` | — | print per-hour phase timing (read / advect / micro / bc+polar) |
| `DEBUG` | — | print finiteness of num/mas after advect & micro (hour 0) |
| `MICRO` | `full` | `full` = SO2 chem + nucleation + coag + condensation; `coag` = legacy coag-only; `off` = none (transport-only benchmark, not a physics config) |
| `MICRO_SUBSTEPS` | 6 | full-micro substeps per coupling step (6 ⇒ 1 h pieces at `STEP_HOURS=6`) |
| `INJ_SO2_TG_YR` | **0** | SAI SO2 injection rate [Tg/yr]. Default dropped from 10 to 0 on 2026-08-03 so a forgotten flag gives an obviously unforced baseline; pass `=10` to reproduce prod90d/prod1yr |
| `INJ_LAT` / `INJ_LON` / `INJ_HPA` | 0 / 180 / 55 | injection cell (nearest grid point / level) |
| `INJ_ZONAL` | **1** | 1 = spread the release over the full latitude ring (was 0 until 2026-08-03) |
| `SETTLE` | 1 | gravitational settling on/off |
| `ALPHA_COND` | 1.0 | H2SO4 accommodation coefficient |
| `NUC_ORG` / `NUC_NH3` / `NUC_FION` | 1e7 / 1e9 / 3.0 | nucleation precursors (SAI box-model defaults) |

## Outputs
- `coupled_final_<tag>.npz` — final `num`, `mas`, grid metadata.
- `coupled_timeseries_<tag>.npz` — burdens, mean Dp, sub-step counts vs time.
- `coupled_frames_<tag>.npz` — daily probe-level (~50 hPa) size-bin snapshots.
- `coupled_state_<tag>_ckpt.npz` — full 3-D state + cumulative counters, written
  last in each checkpoint block; this is what `RESUME=1` reads.

## Environment note
Runs in system `python3`: GPU JAX 0.6.2 + xarray + `tomas_jax`, with
`diffrax`/`equinox`/`lineax`/`optimistix`/`jaxtyping`/`wadler_lindig` installed
to `--user` (needed to import `tomas_jax.solvers.diffrax`; coagulation uses the
forward-Euler path, not diffrax itself).

## Performance
Microphysics dominates cost (**~94%** of a 90-day run, itself ~33 h on one
H100): it solves each cell independently (levels × 192 × 288). Two levers:
- **`N_LEV`** sub-samples the vertical (cost is linear in cell count). 17 levels
  ≈ 1.4× fewer cells than the full 24-level band — and matches the intended
  PARADIS vertical grid, so the level set is directly swappable later.
- **Multi-GPU:** microphysics is `pmap`-sharded across all *visible* GPUs
  automatically (per-cell independent), so `CUDA_VISIBLE_DEVICES` selects the
  scale. `run_prod.sh` deliberately pins a **single** card (`GPU=0`) — on a
  shared machine, exporting more than one is a decision to make explicitly, not
  a default to inherit.

I/O and advection are minor by comparison. See `../MANIFEST.md` for measured
timings of the current engine.

## Not included (the one-way scope)
- **Aerosol → circulation feedback.** The winds come from CESM and are never
  modified, so there is no self-lofting and no dynamical response. Radiative
  heating *does* feed back on the temperature the microphysics and the next
  radiation call see, so that loop is closed — the circulation one is not.
- **Wet removal and dry deposition.** Gravitational settling out of the band
  bottom is the only true aerosol sink; material crossing that face is treated
  as removed by the (unmodelled) tropospheric wet scavenging below it.
- **Per-species aerosol chemistry.** All aerosol mass is one dry-SO4 moment per
  bin; no dust, sea salt or organics are carried separately.

Condensation, nucleation and SO2+OH chemistry **are** included — they are the
`MICRO=full` chain, which is what the production driver runs and requires.
