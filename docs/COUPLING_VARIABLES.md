# CESM→TOMAS Coupling — Complete Variable Reference

Every variable the coupled model consumes or produces, start to finish. Compiled
from `coupling.py`, `radiation.py`, and the fast driver `driver_fast.py`.

Grid: FWHIST f09 (192 lat × 288 lon), stratospheric band levels selected by
pressure (`P_LO_HPA`..`P_HI_HPA`). Coupling step = `STEP_HOURS` (6 h).

---

## 1. Input fields read from CESM FWHIST `*.cam.h1.*` (hourly)

> All 14 non-radiation datasets below are **opened at startup regardless of
> configuration**, so a missing file stops even a run that never reads the field.
> A transport-only run (`MICRO=off`, `AER_SRC=fixed`) reads `U V OMEGA` and `T`
> every step, `RELHUM` too while `WET_SETTLING=1`, and `SO2`/`H2SO4` for the two
> gas tracers' IC and open-BC — those just have no chemistry to drive, so they
> ride along inert. `OH` and the MAM4 fields are read by nothing; the 7 radiation
> variables are not even opened when `RAD=0`.

### Dynamics / thermodynamics (always)
| var | units (file) | role |
|-----|------|------|
| `U`      | m/s     | zonal wind → advection |
| `V`      | m/s     | meridional wind → advection |
| `OMEGA`  | Pa/s    | vertical (pressure) velocity → advection |
| `T`      | K       | temperature → micro kernels, air density (ρ=p/RₐT), radiation |

### Gas phase (MICRO=full)
| var | units (file) | role |
|-----|------|------|
| `SO2`    | mol/mol (vmr) → kg/kg | gas tracer IC+BC; oxidized to H₂SO₄ |
| `H2SO4`  | mol/mol → kg/kg | gas tracer IC+BC; feeds nucleation + condensation |
| `OH`     | mol/mol → molec/cm³ | oxidant driving SO₂+OH→H₂SO₄ |
| `RELHUM` | percent → fraction | condensation, nucleation, water equilibrium |

### Radiation (RAD=1) — read by `radiation.py`
| var | units | role |
|-----|------|------|
| `T`    | K     | atmospheric temperature profile |
| `Q`    | kg/kg | specific humidity (water vapor) |
| `O3`   | mol/mol | ozone (SW/LW absorber) |
| `CH4`  | mol/mol | methane (LW absorber) |
| `FLDS` | W/m²  | surface downwelling longwave (LW boundary) |
| `FLNS` | W/m²  | surface net longwave |
| `FSDS` | W/m²  | surface downwelling shortwave |
| `FSNS` | W/m²  | surface net shortwave → surface albedo = 1 − FSNS/FSDS |

### MAM4 modal aerosol (used if `AER_SRC=mam4`, or as MAM4 IC/BC reservoir)
| var | units | role |
|-----|------|------|
| `num_a1`,`num_a2`,`num_a3` | #/kg | modal number (Aitken/accum/coarse) |
| `so4_a1`,`so4_a2`,`so4_a3` | kg/kg | modal sulfate mass |
| `dgnumwet1/2/3` | m | modal wet diameter — **only if `INIT_BIN=dgnum`** |

## 2. CARMA initial-condition file (used if `AER_SRC=carma`)
File: `cesm2.2_CARMA16node_freerun_1wk_19910601_1deg...nc` (`CARMA_FILE`), frame `CARMA_FRAME`.
| var | units | role |
|-----|------|------|
| `PRSUL01`..`PRSUL20` | kg/kg | pure-sulfate bins (fine/nucleation mode, Dp 0.69 nm–2.59 µm) |
| `MXAER01`..`MXAER20` | kg/kg | mixed-group sulfate bins (accum/coarse, Dp 100 nm–17.4 µm) |

Projected onto TOMAS bins via mp = (4/3)πr³·`CARMA_RHO`.

## 3. Coordinate / structural variables (from the h1 files)
`lat`, `lon`, `lev`, `time`, and the hybrid-sigma coefficients `hyam`, `hybm`, `P0`
→ reference-pressure levels `plev = hyam·P0 + hybm·PS_REF`.

---

## 4. Prognostic state (carried/evolved across steps)
| var | shape | units | notes |
|-----|-------|-------|-------|
| `num` | (NBINS, nlev, nlat, nlon) | #/kg | per-bin number mixing ratio (advected tracer) |
| `mas` | (NBINS, nlev, nlat, nlon) | kg/kg | per-bin **dry sulfate** mass mixing ratio (advected tracer) |
| `so2` | (nlev, nlat, nlon) | kg/kg | SO₂ gas |
| `h2so4` | (nlev, nlat, nlon) | kg/kg | H₂SO₄ vapor |
| `dT_rad` | (nlev, nlat, nlon) | K | accumulated radiative temperature increment (aerosol→heating feedback) |

Advected tracer count = 2·NBINS + 2 (num+mas per bin, plus SO₂, H₂SO₄) = **82 at 40 bins**.

## 5. Derived / setup variables (computed, not read)
| var | meaning |
|-----|---------|
| `klevs`, `band` | native level indices of the stratospheric band |
| `PLEV_PA`, `pres3d` | band pressures [Pa] |
| `DP` | Δp per level [Pa] (vertical advection + burden weighting) |
| `XK` | bin-boundary masses [kg] (`make_grid(NBINS, XK0, 2.0)`) |
| `MMID` | bin geometric-mean mass [kg] |
| `DP_BIN` | per-bin representative diameter [nm] (from MMID, RHO_AER) — used by radiation + meanDp |
| `rho` | air density p/(RD·T) [kg/m³] — per step; converts mixing ratio ↔ per-box concentration |
| `A` / `wgt3d` | burden weight (area×Δp/g) for global integrals |

---

## 6. Configuration knobs (environment variables)

### Run control
`N_DAYS` / `N_HOURS` (length), `H0` (start hour index), `STEP_HOURS` (6),
`OUT_TAG`, `LOG_EVERY`, `FRAME_EVERY`, `DEBUG`, `PROFILE`.

### Domain / grid
`P_LO_HPA`, `P_HI_HPA` (band top/bottom), `N_LEV` (subsample levels; 0=all),
`PROBE_HPA` (diagnostic level), `N_BC_TOP`, `N_BC_BOT` (pinned boundary levels),
`LAT_FREEZE` (=80°, polar-cap latitude, constant).

### Aerosol source & bin grid
`AER_SRC` (`carma`|`mam4`|`fixed`), `N_BINS` (0→native 40), `CARMA_FILE`,
`CARMA_FRAME`, `CARMA_RHO` (1923), `INIT_BIN` (`so4`, the default | `dgnum`,
legacy), `INIT_SIGMA`.

`AER_SRC=fixed` replaces the CESM aerosol entirely with a prescribed uniform,
time-invariant PSD — the advection-only experiment — and adds `FIXED_PSD`
(`lognormal`|`flat`), `FIXED_N` (1e8 #/kg), `FIXED_DG_NM` (200), `FIXED_SIGMA`
(1.6), `FIXED_P_LO_HPA` / `FIXED_P_HI_HPA` (the pressure window, default = whole
band) and `FIXED_LAT_MAX_DEG` (91 = every row; set it for a tagged pulse). None of
the MAM4 or CARMA fields below are read in that mode, and all seven values are
stamped into every output `.npz`. Defaults and rationale:
[CONFIGURATION.md](./CONFIGURATION.md), [../MANIFEST.md](../MANIFEST.md).

### Microphysics
`MICRO` (`full`|`coag`|`off`; `off` = advect+settle only, and the only mode
`driver_fast.py` refuses), `MICRO_SUBSTEPS` (6), `N_COAG_SUBSTEPS`,
`COAG_MAX_SUBSTEPS` (256; physical-path substep cap), `ALPHA_COND` (1.0),
`CELL_CHUNK`, `TRACER_CHUNK`, `SETTLE`.

### Nucleation (physical/ternary path)
`NUC_ORG`, `NUC_NH3`, `NUC_FION`, `NUC_FN_MAX`.
⚠ `NUC_NH3=1e9 pptv`, `NUC_ORG=1e7` are boundary-layer values — inappropriate for
the stratosphere; source of the physical run's over-nucleation. See fast (binary) path.

### OH diurnal
`OH_SZA` (**default 1** = SZA curve, per-substep diurnal; 0 = constant CESM OH),
`OH_PEAK` (2.3e6 molec/cm³ noon peak), `OH_SUBSTEPS` (samples/step; each driver
sets its own, see below).
The curve is a **parabola in μ = cos(SZA) with no constant term**, `OH = a·μ² + b·μ`,
least-squares fitted at import to `OH_SZA_KNOTS` in `coupling.py` — digitized off
Hanisco et al. 2001 Fig. 1: 2.3 / 2.0 / 1.5 / 0.7 / 0 ×10⁶ molec/cm³ at SZA
0 / 30 / 45 / 60 / 90°. Edit those points to refit; nothing else changes.
Dropping the constant term is what forces OH → 0 at the terminator.
Being least-squares it does **not** pass through the knots: +4.0 / −4.8 / −7.2 /
+19.2 % at SZA 0 / 30 / 45 / 60, exact 0 at 90. So peak OH is 2.393e6, 4% above
`OH_PEAK`. It was `OH_PEAK*max(0,1-(θ/90)²)` before 2026-07-29 — a parabola in θ
rather than μ, which decayed far too slowly (1.22e6 at 60° vs the paper's 0.7e6).
⚠ `μ` is clipped to [0,1] and the **output must not be clipped instead**: `a·μ²+b·μ`
has its second root at μ = −0.652, so it turns positive again past SZA 130.7° and
would emit OH at local midnight. A plot over SZA ∈ [−90, 90] cannot reveal this.
⚠ The curve *replaces* CESM OH, so magnitude changes as well as time resolution.
Measured over the 13–88 hPa band: 24 h-mean OH 0.50× CESM's, tropical SO2 lifetime
18.7 d → 36.3 d. No vertical structure (CESM has a 3.4× gradient across the band),
and polar-night OH is exactly 0.
Both drivers resolve the cycle at their own inner step: `coupling.py`'s per-cell
chain at `MICRO_SUBSTEPS`, the fast driver at 6 h/`FAST_DT` = 60 samples/step fed to
`run_fast` as a per-inner-step `(n_steps, ncell)` profile.

### SAI injection source
`INJ_SO2_TG_YR`, `INJ_H2SO4_TG_YR` (rates), `INJ_HPA` (altitude), `INJ_LAT`,
`INJ_LON`, `INJ_ZONAL`, `INJ_MIRROR`. Both **rates** at 0 = no-injection control
(the default); the geometry knobs are then inert.

### Radiation
`RAD` (on/off), `RAD_MODE` (`anomaly`), `RAD_EVERY`, `CO2_PPM` (380), `N2O_PPB` (319).

### Fast reduced model (`driver_fast.py`, tomas_jax.fast)
`FAST_DT` (360 s inner step), `FAST_CELL_CAP` (module 250000 cells/chunk;
`run_prod.sh` hard-sets **50000**, which is faster — see MANIFEST),
`FAST_FN_SCALE` (nucleation scale), `FAST_COAG_SUB_CAP` (256),
`FAST_COND_SUB_CAP` (40), `FAST_COAG_CMAX` (0.05), `FAST_SORT` (stiffness-sort).

### Advection (fast driver)
`ADV_CFL` (0.5), `ADV_F32` (float32 transport).

---

## 7. Physical constants
| const | value | meaning |
|-------|-------|---------|
| `RD` | 287.05 J/kg/K | dry-air gas constant |
| `GRAV` | 9.80665 m/s² | gravity |
| `BOXVOL` | 1.0e6 cm³ | box volume (=1 m³); per-m³ concentrations |
| `PS_REF` | 1.0e5 Pa | reference surface pressure (level construction) |
| `MW_AIR` | 28.9644 g/mol | air molar mass (vmr→MMR) |
| `RHO_AER` | 1770 kg/m³ | sulfate density (mass↔size for TOMAS) |
| `CARMA_RHO` | 1923 kg/m³ | CARMA-file sulfate density (projection) |
| `NEPS_N` | 1e-10 | empty-bin number threshold |
| From tomas_jax config | | `NBINS` (40), `ICOMP`, `XK0`, `KB`, `MW_H2SO4`, `MW_SO2`, `SRTSO4/SRTSO2/SRTH2O` |
| Radiation | | `TSI_DEFAULT` (solar irradiance), `SFC_EMIS` (surface emissivity) |

---

## 8. Outputs / diagnostics produced

The authoritative list is the `ts = {k: [] for k in (...)}` literal in
`coupling.py` — check it there if this section looks short, since new
diagnostics are added by extending that tuple.

**Timeseries** (`coupled_timeseries_<TAG>.npz`), one value per coupling step:

| group | keys |
|---|---|
| normalizers | `N0`, `M0` (scalars, not series) |
| time | `hours` |
| burdens | `Nburden`, `Mburden`, `SO2burden`, `H2SO4burden` |
| size, probe level | `meanDp_nm`, `meanDp_num_nm`, `meanDp_mass_nm`, `reff_nm` |
| size, **whole band** (air-mass weighted; separate keys, not a redefinition) | `reff_dom_nm`, `meanDp_num_dom_nm`, `meanDp_mass_dom_nm` |
| budget (cumulative, fractions of M0) | `B_adv_np`, `B_adv_pol`, `B_floor`, `B_micro`, `B_bc`, `B_settle`, `B_vf_in`, `B_vf_out` |
| radiation | `dT_min`, `dT_max`, `dT_rms`, `arf_toa`, `arf_toa_avg`, `aod550` |
| accounting | `nsub`, `Nmin`, `Nmax`, `Nfloor_cum`, `clipMadd_cum`, `clipMrem_cum`, `injSO2_cum`, `settleM_cum` |
| **resolved drain**, cumulative — one **vector** per record, so the saved arrays are `(nsteps, nlat)` / `(nsteps, NBINS)` | `D_setN_lat`, `D_setM_lat`, `D_setN_bin`, `D_setM_bin` (settling) and `D_vfN_lat`, `D_vfM_lat`, `D_vfN_bin`, `D_vfM_bin` (advective flux through the same face) |

The drain vectors are what `plot_run.py`'s `<TAG>_drain.png` is drawn from — the
figure of the advection-only comparison — and they close against the globally
summed counters: `D_setM_lat.sum() == settleM_cum`, `D_vfM_lat.sum() ==
-B_vf_out*M0`. The two channels are kept apart on purpose: settling depends on the
model's T and nothing else, the advective flux *is* the residual circulation.

**Frames** (`coupled_frames_<TAG>.npz`), snapshots every `FRAME_EVERY` hours, in
three reductions of the same fields — probe level (`frames_*`), vertical column
integral (`frames_col_*`, f32) and zonal-mean lat–height cross-section
(`frames_zm_*`):

| group | keys |
|---|---|
| probe level (`PROBE_HPA`) | `frames_num`, `frames_mas`, `frames_dT`, `frames_so2`, `frames_h2so4` |
| column | `frames_col_num`, `frames_col_mas`, `frames_col_dT`, `frames_col_so2`, `frames_col_h2so4` |
| zonal-mean cross-section (bin-summed) | `frames_zm_num`, `frames_zm_mas`, `frames_zm_dT`, `frames_zm_so2`, `frames_zm_h2so4` |
| vertically averaged PSD | `frames_psd_num`, `frames_psd_mas` |
| grid, so the plots need only this file | `frame_hours`, `probe_hpa`, `xk`, `plev_hpa`, `dp_pa`, `col_kgm2`, `lat`, `lon` |

A single-level map cannot show vertical transport at all — aerosol that sinks out
of the probe level simply vanishes from the figure — which is why the column and
cross-section reductions exist and why they are the ones the filmstrip and GIFs
prefer. Frames are the big output: ~50 MB per frame at 40 bins, ~3.5 MB at
`N_BINS=1`. **The whole history is rewritten at every frame**, so cost grows as
(frames)²; raise `FRAME_EVERY` for long runs.

**Both files also carry the run's configuration stamp** — `inj_cfg` /
`inj_cfg_keys` (the injection scenario) and `phys_cfg` / `phys_cfg_keys` (the
physics-mode flags). `RESUME` compares these: an `INJ_*` mismatch is refused
outright, a physics-mode mismatch only warns. Both arrays are append-only, so
adding a field must not lock out older checkpoints.

Older runs may lack the newer keys (`prod90d`, for instance, predates
`reff_nm`); `plot_run.py` degrades rather than failing when one is absent.
