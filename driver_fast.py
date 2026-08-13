"""Fully-linked coupled run using the tomas-jax GPU-FAST reduced model.

Runs coupling.py with ONE difference: the microphysics engine is the
natively-batched reduced model `tomas_jax.fast` (gpu-fast branch) instead of the
hand-assembled per-cell chain (coupling.run_microphysics_full). Advection is the
same fct_lr swap coupling.py already uses, so the two agree on transport.

IC/BC source is coupling.py's default, AER_SRC=mam4 (per-step dynamic from the
CESM hourly series). Pass AER_SRC=carma for the static CARMA reservoir instead.

Why a separate driver (not just an env flag):
  * `tomas_jax.fast` is a DIFFERENT engine with its own batched API
    (FastState / run_fast) -- we monkeypatch coupling.run_microphysics_full with
    a shim that maps coupling's per-cell state <-> FastState and calls run_fast.
  * The fast model is HARD-FIXED at 40 bins, aerosol SO4+H2O (ICOMP=2), gases
    H2SO4+SO2. So this driver runs at NATIVE 40 bins and does NOT set N_BINS at
    all -- it exits if you do. At 40 bins coupling's XK == the fast model's grid exactly
    (both make_grid(40, XK0, 2.0)), so num/mas are already on the right grid.
  * Reduced physics differs from the 20-bin physical reference: Dunne-2016
    neutral-binary nucleation (no org/NH3/ion ternary terms), so nucleation and
    the resulting number are NOT directly comparable to the physical run. This is
    a first look at behaviour + speed of the fast engine, not an A/B match.

Run: use ../run_prod.sh, which sets the whole production environment (GPU pin,
libcuda path, memory policy) and execs this file. Direct invocation for a short
test, once tomas-jax and jax-rrtmgp are importable -- clone them beside this repo
or set TOMAS_JAX_PATH / RRTMGP_PATH; coupling.py resolves both:

  INJ_SO2_TG_YR=0 N_HOURS=18 DEBUG=1 PROFILE=1 \
      FAST_CELL_CAP=250000 OUT_TAG=smoke python3 driver_fast.py

Do NOT activate the tomas-jax .venv: it carries CPU-only jaxlib and no xarray,
so the run silently loses the GPU. System python3 with GPU jax is the combo.
"""
import os
import sys
import time
import functools

import numpy as np

# --- env defaults for this driver (N_BINS is deliberately NOT set: see below) ---
os.environ.setdefault('RAD', '1')
os.environ.setdefault('RAD_MODE', 'anomaly')
os.environ.setdefault('RAD_EVERY', '1')
os.environ.setdefault('MICRO', 'full')            # main() takes the full-chain path
os.environ.setdefault('STEP_HOURS', '6')
# domain: 1-150 hPa (changed 2026-07-29, was 12-100). 24 native levels =
# 1,327,104 cells, 2.18x the 11-level 12-100 band, so expect ~2.2x the step cost
# (micro is ~94% of it): ~170 -> ~370 s/step, a 90-day run ~17 h -> ~37 h.
# Rationale for each edge:
#   1 hPa top  -- MAM4 so4 there is 3.2e-17 kg/kg, 7 orders below the 13 hPa value,
#                 so the top face is effectively an aerosol-free inflow (correct for
#                 mesospheric subsidence) without needing a knob to zero it.
#   150 hPa bot -- MAM4 at 143 hPa is 0.6x the 88 hPa value, and crucially its
#                 LATITUDE structure is right: 7.5e-11 in the tropics (the TTL,
#                 already convectively scavenged, and where upwelling enters the
#                 band) vs 1.26e-9 at 60N (where STE brings real air in). The old
#                 BC_BOT_AER=0 hack zeroed both, right for the tropics and wrong
#                 for the extratropics.
# CAVEAT: dropping the floor from ~88 to 143 hPa DELAYS settling, which is the
# model's only true aerosol sink -- expect a longer effective lifetime and higher
# steady-state burden. There is still no wet removal anywhere, so SAI aerosol that
# settles into the 100-150 hPa layer lingers there.
os.environ.setdefault('P_LO_HPA', '1.0')
os.environ.setdefault('P_HI_HPA', '150.0')
# AER_SRC deliberately NOT set here (changed 2026-07-29): this driver used to force
# 'carma', which silently overrode coupling.py's own 'mam4' default, so every run
# through this file got a STATIC aerosol reservoir unless AER_SRC was passed
# explicitly. MAM4 is the right default -- it is per-step dynamic (bin_mam4 indexes
# the hourly h1 series; verified 100% of cells change over 24 h) and it is the only
# source that exists past the 1-week CARMA output, which multi-year runs need.
# CARMA is now explicit opt-in: AER_SRC=carma. NOTE all runs before this date
# (zonal90d, zonal4wk, carma_tomas_*) were CARMA/static -- not comparable.
os.environ.setdefault('OUT_TAG', 'tomas_fast')
# NOTE: deliberately NOT setting N_BINS -> coupling runs at native NBINS=40,
# the grid the fast model requires.

# ---- fast-model knobs -------------------------------------------------------
FAST_DT        = float(os.environ.get('FAST_DT', '360'))        # inner outer-step [s]
FAST_CELL_CAP  = int(os.environ.get('FAST_CELL_CAP', '250000')) # max cells per chunk
FAST_FN_SCALE  = float(os.environ.get('FAST_FN_SCALE', '1.0'))  # nucleation rate scale
FAST_COAG_CAP  = int(os.environ.get('FAST_COAG_SUB_CAP', '256'))
FAST_COND_CAP  = int(os.environ.get('FAST_COND_SUB_CAP', '40'))
FAST_COAG_CMAX = float(os.environ.get('FAST_COAG_CMAX', '0.05'))
FAST_SORT      = os.environ.get('FAST_SORT', '1') != '0'        # stiffness-sort chunks

if os.environ.get('STEP_HOURS', '6') != '6':
    sys.exit("driver_fast: STEP_HOURS must be 6 (parity with 6h step)")
if os.environ.get('MICRO', 'full') != 'full':
    sys.exit("driver_fast: MICRO must be 'full'")
if os.environ.get('N_BINS', '0') not in ('0', '40'):
    sys.exit(f"driver_fast: fast model is 40-bin -- unset N_BINS or set "
             f"40; got N_BINS={os.environ.get('N_BINS')}")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'fast_advection'))
sys.path.insert(0, HERE)

import jax.numpy as jnp

# NB the compat shim that used to live here (aliasing coupling.py's imported
# calc_oh_parabola to tomas-jax's calc_oh_concentration) is GONE and must not come
# back: the two take their arguments in the opposite order and
# calc_oh_concentration defaults to constant mode, so the alias made
# calc_oh_parabola(cos_sza, OH_PEAK) return cos(SZA) -- OH ~ 1 molec/cm3, i.e. no
# SO2 oxidation at all -- in every OH_SZA=1 run. coupling.py now owns the fit.

import coupling as C
from fct_fast import advect_hour_batch as _fast_advect
try:
    from tomas_jax.fast import FastState, run_fast
    from tomas_jax.fast.config import NBINS as FAST_NBINS, SRTSO4, GH2SO4, GSO2
except ImportError as _e:
    # coupling.py imported fine (it uses the per-cell chain, which is on every
    # branch), so tomas-jax is present but on the wrong one.
    raise SystemExit(
        f"driver_fast.py: cannot import tomas_jax.fast ({_e}).\n"
        "  The batched engine exists only on the tomas-jax `gpu-fast` branch;\n"
        "  `main` has the per-cell chain only. In your tomas-jax clone:\n"
        "      git checkout gpu-fast\n"
        "  See the README Installation section.")

# native grid must be the fast model's 40-bin grid
if C.NBINS != FAST_NBINS:
    sys.exit(f"driver_fast: coupling NBINS={C.NBINS} != fast NBINS="
             f"{FAST_NBINS}; do not set N_BINS for this driver")

# ---- the advection swap (same scheme coupling.py binds for itself) ----------
_ADV_CFL = float(os.environ.get('ADV_CFL', '0.5'))
_ADV_DTYPE = jnp.float32 if os.environ.get('ADV_F32', '1') != '0' else jnp.float64
# ADV_SCHEME selects the transport form:
#   lr   (default) fast_advection/fct_lr.py -- Lin-Rood FLUX form with air-mass
#        tracking. Tracer mass is conserved to ROUNDOFF (~1e-15/day) for any wind
#        field, divergent or not, so it does not depend on the winds satisfying
#        continuity. Costs ~5x more substeps (no integer-shift FFSL in flux form).
#   fast fast_advection/fct_fast.py -- advective form with the 2026-07-25 fixes
#        (true grid spacing, cos(phi) metric, continuity omega, conserving polar
#        caps). Residual ~3e-4/day; cheaper. Kept as the validated fallback.
_ADV_SCHEME = os.environ.get('ADV_SCHEME', 'lr').lower()
if _ADV_SCHEME == 'lr':
    from fct_lr import advect_hour_batch as _adv_fn
else:
    _adv_fn = _fast_advect
C.advect_hour_batch = functools.partial(_adv_fn, cfl=_ADV_CFL, dtype=_ADV_DTYPE)

# number of inner fast steps that fill one coupling step (6 h / FAST_DT)
_N_FAST_STEPS = int(round(C.DT_MICRO / FAST_DT))

# Sample the diurnal OH fit at THIS driver's inner step, not at the physical
# chain's MICRO_SUBSTEPS: coupling.oh_sza reads C.OH_SUBSTEPS at call time, so
# with OH_SZA=1 the parabola is evaluated at each inner step's local solar zenith
# angle and run_microphysics_full_fast hands run_fast one OH per (inner step,
# cell). Respect an explicit OH_SUBSTEPS from the environment (A/B testing the
# sampling rate); _oh_to_substeps then resamples onto _N_FAST_STEPS.
if 'OH_SUBSTEPS' not in os.environ:
    C.OH_SUBSTEPS = _N_FAST_STEPS


def run_microphysics_full_fast(num, mas, so2, h2so4, temp3d, pres3d, rh3d, oh3d,
                               wgt3d):
    """Drop-in for coupling.run_microphysics_full backed by tomas_jax.fast.

    Same signature/return contract and the SAME mixing-ratio<->per-box
    conversion + two-moment consistency clip + clip diagnostic as the original,
    but the per-cell chain is replaced by one run_fast() over all cells.
    """
    nbin, nlev, nlat, nlon = num.shape
    assert nbin == FAST_NBINS, f"expected {FAST_NBINS} bins, got {nbin}"
    ncell = nlev * nlat * nlon

    rho = np.asarray(pres3d) / (C.RD * np.asarray(temp3d))      # kg/m3
    rho_f = rho.reshape(ncell)
    inv_rho = 1.0 / np.maximum(rho_f, 1e-30)
    T_f  = np.asarray(temp3d).reshape(ncell)
    P_f  = np.asarray(pres3d).reshape(ncell)
    RH_f = np.asarray(rh3d).reshape(ncell)
    # OH: run_fast takes a per-inner-step profile (tomas-jax gpu-fast 5ca1d73), so
    # a diurnal field is fed in at the resolution the chemistry runs at instead of
    # being averaged away. C.OH_SUBSTEPS = _N_FAST_STEPS (set below) makes
    # coupling.oh_sza sample the SZA parabola at THIS driver's inner step, so the
    # 4D field arrives with exactly _N_FAST_STEPS rows and _oh_to_substeps is a
    # transpose; a (nlev,nlat,nlon) CESM field (OH_SZA=0) still broadcasts to a
    # constant profile, and any other row count resamples rather than failing.
    #   shape (_N_FAST_STEPS, ncell), row t = OH seen by inner step t
    # 60 x 608256 float64 is ~292 MB and gets re-uploaded once per cell chunk;
    # that is ~10% of the coag kernel's own footprint, and milliseconds of PCIe
    # against a multi-minute micro step. A time-constant field stays (ncell,) so
    # run_fast closes over it and never materializes the time axis at all.
    oh_arr = np.asarray(oh3d)
    OH_f = (C._oh_to_substeps(oh_arr, ncell, nsub_out=_N_FAST_STEPS).T
            if oh_arr.ndim == 4 else oh_arr.reshape(ncell))

    # mixing ratio -> per-box (1 m3) concentration
    Nk_all = np.asarray(num).reshape(nbin, ncell).T * rho_f[:, None]   # (ncell,nbin)
    Mt_all = np.asarray(mas).reshape(nbin, ncell).T * rho_f[:, None]
    S_all  = np.asarray(so2).reshape(ncell)   * rho_f     # kg SO2 per box
    H_all  = np.asarray(h2so4).reshape(ncell) * rho_f     # kg H2SO4 per box

    # ---- two-moment consistency clip (advected Nk/Mt arrive independent) ----
    # fast's create() does NOT clip, so apply it here exactly as _micro_cell does
    # and report the burden-weighted add/remove for the mass budget.
    XK_NP = C.XK_NP
    Nk_eff = np.where(Nk_all > C.NEPS_N, Nk_all, 0.0)
    Mt_clip = np.clip(Mt_all, Nk_eff * XK_NP[None, :-1], Nk_eff * XK_NP[None, 1:])
    dM_mr = (Mt_clip - Mt_all) * inv_rho[:, None]
    w = np.asarray(wgt3d).reshape(ncell)[:, None]
    clip_add = float((np.maximum(dM_mr, 0.0) * w).sum())
    clip_rem = float((np.minimum(dM_mr, 0.0) * w).sum())

    # ---- build the batched FastState ----
    Mk = np.zeros((ncell, nbin, 2), dtype=np.float64)
    Mk[:, :, SRTSO4] = Mt_clip                       # SO4 dry mass; H2O col = 0
    Gc = np.zeros((ncell, 2), dtype=np.float64)
    Gc[:, GH2SO4] = H_all
    Gc[:, GSO2]   = S_all
    state = FastState.create(Nk=Nk_eff, Mk=Mk, Gc=Gc, xk=np.asarray(C.XK),
                             temp=T_f, pres=P_f, boxvol=C.BOXVOL, rh=RH_f)

    ncc = max(1, -(-ncell // FAST_CELL_CAP))          # ceil(ncell / cap)
    t0 = time.time()
    state2, diags = run_fast(
        state, _N_FAST_STEPS, dt=FAST_DT, oh_conc=OH_f,
        n_cell_chunks=ncc, sort_by_coag_cost=FAST_SORT,
        alpha=C.ALPHA_COND, fn_scale=FAST_FN_SCALE,
        coag_sub_cap=FAST_COAG_CAP, cond_sub_cap=FAST_COND_CAP,
        coag_c_max=FAST_COAG_CMAX)
    dt_fast = time.time() - t0

    out_N = np.asarray(state2.Nk)                     # (ncell, nbin)
    out_M = np.asarray(state2.Mk[:, :, SRTSO4])       # dry SO4
    out_S = np.asarray(state2.Gc[:, GSO2])
    out_H = np.asarray(state2.Gc[:, GH2SO4])

    if os.environ.get('DEBUG') or os.environ.get('PROFILE'):
        cc = int(np.asarray(diags['coag_cap_hit']).sum())
        dc = int(np.asarray(diags['cond_cap_hit']).sum())
        print(f"  [fast] run_fast: {ncell} cells x {_N_FAST_STEPS} steps "
              f"(dt={FAST_DT:.0f}s, {ncc} chunk(s), sort={FAST_SORT}) in "
              f"{dt_fast:.1f}s | coag_cap_hit_steps={cc} cond_cap_hit_steps={dc}",
              flush=True)

    # per-box -> mixing ratio, restore (nbin,nlev,nlat,nlon)/(nlev,nlat,nlon)
    num2   = (out_N * inv_rho[:, None]).T.reshape(nbin, nlev, nlat, nlon)
    mas2   = (out_M * inv_rho[:, None]).T.reshape(nbin, nlev, nlat, nlon)
    so2_2  = (out_S * inv_rho).reshape(nlev, nlat, nlon)
    h2so4_2 = (out_H * inv_rho).reshape(nlev, nlat, nlon)
    return num2, mas2, so2_2, h2so4_2, clip_add, clip_rem


# ---- the ONE micro swap; advection already swapped above --------------------
C.run_microphysics_full = run_microphysics_full_fast

print("=" * 60, flush=True)
print(f"COUPLING: tomas_jax.fast (GPU-fast reduced model) + "
      f"{C.AER_SRC.upper()} IC/BC", flush=True)
print(f"  micro    : tomas_jax.fast.run_fast  ({_N_FAST_STEPS} steps x "
      f"{FAST_DT:.0f}s, cap {FAST_CELL_CAP} cells/chunk, sort={FAST_SORT})",
      flush=True)
print(f"  advection: fast_advection/fct_{_ADV_SCHEME}  (cfl={_ADV_CFL}, "
      f"{'f32' if _ADV_DTYPE is jnp.float32 else 'f64'})", flush=True)
print(f"  OH       : " + (f"SZA curve sampled {C.OH_SUBSTEPS}x/step -> "
                          f"per-inner-step profile into run_fast" if C.OH_SZA
                          else "CESM field, constant over step"), flush=True)
      # read from coupling (not os.environ): AER_SRC is no longer forced here, so
      # the env key may be absent entirely and coupling.py owns the 'mam4' default
print(f"  IC/BC    : AER_SRC={C.AER_SRC} "
      + ("(per-step dynamic from CESM h1)" if C.AER_SRC == 'mam4'
         else "(STATIC reservoir, built once)")
      + f"   bins={C.NBINS} (native)", flush=True)
print("=" * 60, flush=True)

if __name__ == '__main__':
    C.main()
