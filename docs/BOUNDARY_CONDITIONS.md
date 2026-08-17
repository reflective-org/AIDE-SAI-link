# Boundary conditions without CESM output files

Goal: multi-year SAI runs driven by a circulation emulator + microphysics
emulator + (physical) radiation, with **no CESM history files at runtime**.
This note inventories where the model currently leans on CESM, and what to
replace each dependence with.

## Where CESM files enter today

| dependence | consumer | fields |
|---|---|---|
| initial condition | `bin_mam4(t=0)`, `read_gases(t=0)` | num/so4_a{1,2,3}, SO2, H2SO4 |
| open-BC slabs (per step) | top/bottom band levels | same as IC |
| polar-cap refresh (per step) | `qfroz` rows \|lat\|>80 | same as IC |
| meteorology (per step) | advection + micro | U, V, OMEGA, T |
| chemistry/micro forcing (per step) | `run_microphysics_full` | OH, RELHUM |
| radiation reference (per step) | anomaly mode | MAM4 bins + rad driver inputs |

## The key change of regime

The N2O proof-of-concept was a *tracer relaxation* problem: everything the
tracer did came from the boundaries, so the boundaries had to carry the full
CESM signal. With the SO2 -> H2SO4 -> SO4 chain and settling in the model, the
SAI aerosol is **internally generated and internally removed**: source =
injection, sink = settling through the band bottom. The boundaries only need
to supply a plausible *background* state, not the signal. That downgrades the
BC problem from "assimilate CESM" to "carry a climatology", which is exactly
what makes CESM-free long runs feasible.

## Recommended design: a one-time cyclic climatology file

Run a preprocessing pass over the 19-year CESM archive ONCE, and save a
day-of-year-cyclic climatology of exactly the fields the BCs consume:

* **boundary slabs**: binned (num, mas) on the TOMAS grid + SO2/H2SO4 [kg/kg]
  at the top `N_BC_TOP` and bottom `N_BC_BOT` levels — monthly means are
  plenty (slabs are reservoirs, not weather). Tiny: (12, 2*NBINS+2, nslab,
  nlat, nlon) or zonal-mean (…, nlat).
* **polar caps**: same state, all band levels, |lat|>LAT_FREEZE rows only.
* **OH**: monthly **zonal-mean** (12, nlev, nlat) plus the diurnal shape
  applied at runtime via cos(SZA) — `tomas_jax.physics.so2_chemistry` already
  has `calc_solar_zenith_angle` + `calc_oh_concentration` for exactly this.
  OH is the one field with a hard diurnal cycle; zonal-mean × diurnal factor
  captures it far better than an hourly snapshot ever did.
* **RELHUM**: monthly zonal-mean (stratosphere is dry and zonally smooth).
* **IC**: any January from the same climatology. A multi-year run forgets its
  IC in months, so this costs nothing.

Runtime then linearly interpolates in day-of-year and cycles forever. One
`.npz` of a few MB replaces the entire `$CESM_DIR` archive dependence.

**One row of that table is already gone.** `AER_SRC=fixed` (2026-08-13) replaces
the aerosol IC, the boundary slabs and the polar-cap reservoir with a prescribed
uniform PSD carrying no CESM information at all, and `BC_TOP_AER=0 BC_BOT_AER=0`
switches the face inflow off outright. A transport-only run therefore already
depends on CESM for nothing but U/V/OMEGA/T (+ RELHUM for the wet size) — which is
exactly the set an emulator replaces by construction. It is not a climatology, so
it does not solve the coupled model's BC problem; it does mean the advection-only
comparison can be pointed at a new wind source with no aerosol boundary work.

## Per-boundary physics

**Bottom (~150 hPa, the production `P_HI_HPA`).** Two distinct roles, now separable:
* *outflow* — settling already exits through an open bottom face
  (`settling.py`), independent of any reservoir. Advective export into the
  pinned bottom slab is likewise absorbed by the reset. Sink: solved.
* *inflow* — tropical upwelling carries (nearly aerosol-free, SO2-bearing)
  tropospheric air into the band. A climatology-clamped bottom slab
  represents this fine. Later upgrade if wanted: a true flux BC (inflow at
  climatological concentration where OMEGA<0, free outflow where OMEGA>0),
  but the clamped slab is defensible indefinitely.

**Top (~1 hPa).** Effectively aerosol-free; climatology or zeros. Nothing
interesting crosses it except number in the smallest bins.

> **Zeros are not free, though — where the lid sits matters.** Measured on the
> advection-only runs: at a 1 hPa lid, 6.9% of the domain's air descends through
> the top face per 90 days, and with `BC_TOP_AER=0` it carries `q = 0`, diluting
> the band *without appearing in any mass term* (the face term counts only aerosol
> leaving). Tropical zonal-mean ascent turns poleward at ~2.1 hPa in this forcing,
> so a 1 hPa lid is inside the circulation, not above it. Raising the domain top to
> 0.03 hPa (33 levels) cuts the descent to ~0.8%. A climatological top slab has the
> same geometry problem: the fix is the lid height, not the slab's contents. See
> `../MANIFEST.md`.

**Poles (|lat|>80).** The freeze is *numerical* (lon-sweep CFL ~ 1/cos(lat)
blows up), not physical — but pinning polar cells to a *background*
climatology has a real cost in SAI runs: the Brewer-Dobson circulation
converges injected aerosol poleward, and a background-pinned cap silently
deletes that burden every step (visible today as the `adv_pol` budget term).
Options, in increasing effort:
1. **Relax, don't pin** (recommended next step): replace the hard overwrite
   with relaxation of the caps toward the *zonal mean of the adjacent
   resolved rows* (e.g. 78–80°) on a few-day timescale. Self-contained (no
   external data at all), keeps the caps numerically tame, and lets SAI
   aerosol accumulate over the poles instead of vanishing. Cheap to
   implement inside `advect_hour_batch`'s existing `qfroz` mechanism —
   build `qfroz` polar rows from the model state itself.
2. **Polar filter**: damp high zonal wavenumbers poleward of ~75° (FFT along
   lon is cheap in JAX) so the CFL constraint relaxes and LAT_FREEZE can move
   to 88–89°. Standard GCM practice; a real fix.
3. **Grid change** (cubed sphere / reduced grid): only worth it if the
   circulation emulator dictates a new grid anyway.

**Meteorology.** The circulation emulator replaces U/V/OMEGA/T wholesale —
that dependence disappears by construction. Two things to watch:
* the advection scheme needs *mass-consistent* winds; emulator winds won't
  satisfy continuity exactly, so expect to need a divergence-consistency fix
  (or accept the `floor`/budget noise growing).
* T drives the coag kernel, settling viscosity, and the SO2+OH rate — all
  smooth functions of T, so emulator-grade T is fine.

**Radiation.** Stays physical (rrtmgp + Mie) because the NN radiation
emulator is aerosol-blind, and therefore unusable for SAI forcing. Its
non-aerosol inputs (trace gases, surface albedo/T) come from the same
climatology file. Anomaly mode's *reference aerosol* should switch from
"MAM4 binned at this hour" to "climatological binned background", which is
also more physically meaningful: the anomaly is then "SAI vs unperturbed
climatological stratosphere" rather than "vs whatever CESM weather did".

## What stays open

Once emulator winds respond to the aerosol's radiative heating (two-way
coupling), the poles/bottom questions get re-asked inside the emulator's own
dynamics; the climatology file remains valid as the *chemical/background*
boundary either way.

## Suggested implementation order

1. `make_bc_climatology.py`: preprocessing pass over the CESM archive →
   `bc_climatology.npz` (slabs, caps, OH, RH, IC). Pure I/O, no model change.
2. `BC_SOURCE=cesm|clim` switch in `coupling.py` reading the same code paths
   from the npz instead of xarray. Validate: clim-BC run vs CESM-BC run over
   a few weeks — with injection on, the SAI signal should be nearly
   identical (boundaries are background).
3. Polar relax-to-neighbor-zonal-mean (drops the polar CESM dependence and
   fixes the polar SAI-burden deletion in one move).
4. Flux-form bottom BC (optional, later).
