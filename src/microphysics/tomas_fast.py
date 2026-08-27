"""The tomas_jax.fast adapter: coupling.py's per-cell state <-> batched FastState.

Extracted from driver_fast.py 2026-08-27 so that every coupled process has a
home under src/. This is the microphysics side of the coupling; the physics
itself lives in the tomas-jax submodule.

`run_microphysics_full_fast` is a drop-in for `coupling.run_microphysics_full`:
same signature, same return contract, same mixing-ratio <-> per-box conversion
and two-moment consistency clip. The difference is that coupling's per-cell
chain is replaced by ONE batched run_fast() over all cells.

IMPORT ORDER MATTERS. This module imports coupling, and coupling reads the
environment at import time, so whatever imports THIS module must have applied
its os.environ.setdefault block first. driver_fast.py does exactly that, and its
import of this module is deliberately placed below that block -- do not let an
"organize imports" pass hoist it. (One of the reasons ruff's I001 is off.)
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _paths                      # noqa: E402 -- puts src/ subdirs on sys.path

import coupling as C               # noqa: E402 -- see IMPORT ORDER above

try:
    from tomas_jax.fast import FastState, run_fast
    from tomas_jax.fast.config import NBINS as FAST_NBINS, SRTSO4, GH2SO4, GSO2
except ImportError as _e:
    # coupling.py imported fine (it uses the per-cell chain, which is on every
    # branch), so tomas-jax is present but on the wrong one.
    raise SystemExit(
        f"tomas_fast.py: cannot import tomas_jax.fast ({_e}).\n"
        "  The batched engine exists only on the tomas-jax `gpu-fast` branch;\n"
        "  `main` has the per-cell chain only. In your tomas-jax submodule:\n"
        "      git -C models/tomas-jax checkout gpu-fast\n"
        "  See the README Installation section.")

# native grid must be the fast model's 40-bin grid
if C.NBINS != FAST_NBINS:
    sys.exit(f"tomas_fast: coupling NBINS={C.NBINS} != fast NBINS="
             f"{FAST_NBINS}; do not set N_BINS for this driver")

# ---- fast-model knobs -------------------------------------------------------
FAST_DT        = float(os.environ.get('FAST_DT', '360'))        # inner outer-step [s]
FAST_CELL_CAP  = int(os.environ.get('FAST_CELL_CAP', '250000')) # max cells per chunk
FAST_FN_SCALE  = float(os.environ.get('FAST_FN_SCALE', '1.0'))  # nucleation rate scale
FAST_COAG_CAP  = int(os.environ.get('FAST_COAG_SUB_CAP', '256'))
FAST_COND_CAP  = int(os.environ.get('FAST_COND_SUB_CAP', '40'))
FAST_COAG_CMAX = float(os.environ.get('FAST_COAG_CMAX', '0.05'))
FAST_SORT      = os.environ.get('FAST_SORT', '1') != '0'        # stiffness-sort chunks

# number of inner fast steps that fill one coupling step (6 h / FAST_DT)
N_FAST_STEPS = int(round(C.DT_MICRO / FAST_DT))

# Sample the diurnal OH fit at THIS driver's inner step, not at the physical
# chain's MICRO_SUBSTEPS: coupling.oh_sza reads C.OH_SUBSTEPS at call time, so
# with OH_SZA=1 the parabola is evaluated at each inner step's local solar zenith
# angle and run_microphysics_full_fast hands run_fast one OH per (inner step,
# cell). Respect an explicit OH_SUBSTEPS from the environment (A/B testing the
# sampling rate); _oh_to_substeps then resamples onto N_FAST_STEPS.
if 'OH_SUBSTEPS' not in os.environ:
    C.OH_SUBSTEPS = N_FAST_STEPS


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
    # being averaged away. C.OH_SUBSTEPS = N_FAST_STEPS (set below) makes
    # coupling.oh_sza sample the SZA parabola at THIS driver's inner step, so the
    # 4D field arrives with exactly N_FAST_STEPS rows and _oh_to_substeps is a
    # transpose; a (nlev,nlat,nlon) CESM field (OH_SZA=0) still broadcasts to a
    # constant profile, and any other row count resamples rather than failing.
    #   shape (N_FAST_STEPS, ncell), row t = OH seen by inner step t
    # 60 x 608256 float64 is ~292 MB and gets re-uploaded once per cell chunk;
    # that is ~10% of the coag kernel's own footprint, and milliseconds of PCIe
    # against a multi-minute micro step. A time-constant field stays (ncell,) so
    # run_fast closes over it and never materializes the time axis at all.
    oh_arr = np.asarray(oh3d)
    OH_f = (C._oh_to_substeps(oh_arr, ncell, nsub_out=N_FAST_STEPS).T
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
        state, N_FAST_STEPS, dt=FAST_DT, oh_conc=OH_f,
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
        print(f"  [fast] run_fast: {ncell} cells x {N_FAST_STEPS} steps "
              f"(dt={FAST_DT:.0f}s, {ncc} chunk(s), sort={FAST_SORT}) in "
              f"{dt_fast:.1f}s | coag_cap_hit_steps={cc} cond_cap_hit_steps={dc}",
              flush=True)

    # per-box -> mixing ratio, restore (nbin,nlev,nlat,nlon)/(nlev,nlat,nlon)
    num2   = (out_N * inv_rho[:, None]).T.reshape(nbin, nlev, nlat, nlon)
    mas2   = (out_M * inv_rho[:, None]).T.reshape(nbin, nlev, nlat, nlon)
    so2_2  = (out_S * inv_rho).reshape(nlev, nlat, nlon)
    h2so4_2 = (out_H * inv_rho).reshape(nlev, nlat, nlon)
    return num2, mas2, so2_2, h2so4_2, clip_add, clip_rem


