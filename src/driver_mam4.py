"""Coupled run with the MAM4-JAX MODAL microphysics ("run 3" of the SAI campaign).

driver_fast.py's pattern, duplicated on purpose rather than parameterised:
import coupling.py, swap the advection (same ADV_SCHEME selection), swap the
microphysics seam -- and, because the prognostic aerosol state is 4 modes x
(number, SO4) = 8 rows instead of 40 bins x 2 moments = 80, rebind the three
aerosol-representation hooks coupling.py exposes for exactly this purpose:

    C.AER_NROWS   = 4                       rows per moment in the tracer stack
    C.AER_FILL_FN = mam4_adapter.modal_fill IC / open-BC slabs / polar refresh
                                            from the archive's num_a*/so4_a*
    C.aer_to_bins = mam4_adapter.aer_to_bins modes -> TOMAS bins for the Mie
                                            optics and the size diagnostics
    C.aer_settle  = mam4_adapter.settle_modal bins -> settle_step -> refit
    C.run_microphysics_full = mam4_adapter.run_microphysics_full_mam4

Everything else in the step (transport of the 10-row stack, budgets, the
staged budget, frames, checkpoints, the radiation reference from the archive)
is coupling.py's own code; the radiation reference aerosol stays binned
(coupling's aer_fill_rad), only the PROGNOSTIC state is modal.

Physics (src/microphysics/mam4_adapter.py): MAM4-JAX condensation,
coagulation, calcsize/rename; Dunne 2016 nucleation and Tabazadeh 1997 water
(the tomas_jax.fast functions) swapped in; SO2 chemistry outside the core.
MAM4_PHYS=stock runs the core as shipped for an A/B.

Needs the hooks in coupling.py (runs/mam4_dev/STATUS.md has the patch); the
driver refuses to start without them rather than silently running TOMAS.

Run (from a directory under runs/):
  LD_LIBRARY_PATH=/run/nvidia/driver/usr/lib/x86_64-linux-gnu CUDA_VISIBLE_DEVICES=1 \\
  N_HOURS=24 INJ_SO2_TG_YR=10 INJ_LAT=30 INJ_MIRROR=1 H0=8766 ADV_SCHEME=sl ADV_SL_K=12 \\
  OUT_TAG=mam4_smoke DEBUG=1 PROFILE=1 \\
  /data/AIDE-SAI-link/runs/mam4_dev/venv/bin/python /data/AIDE-SAI-link/src/driver_mam4.py
"""
import os
import sys
import functools

# --- env defaults, identical to driver_fast.py ----------------------------------
os.environ.setdefault('RAD', '1')
os.environ.setdefault('RAD_MODE', 'anomaly')
os.environ.setdefault('RAD_EVERY', '1')
os.environ.setdefault('MICRO', 'full')
os.environ.setdefault('STEP_HOURS', '6')
os.environ.setdefault('P_LO_HPA', '1.0')
os.environ.setdefault('P_HI_HPA', '150.0')
os.environ.setdefault('OUT_TAG', 'mam4')

if os.environ.get('STEP_HOURS', '6') != '6':
    sys.exit("driver_mam4: STEP_HOURS must be 6 (parity with the TOMAS driver)")
if os.environ.get('MICRO', 'full') not in ('full', 'off'):
    sys.exit("driver_mam4: MICRO must be 'full' (or 'off' for a transport-only run)")
if os.environ.get('AER_SRC', 'mam4').lower() != 'mam4':
    sys.exit("driver_mam4: the modal state is initialised from the archive's MAM4 "
             "modes; AER_SRC=carma has no modal form")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _paths                      # noqa: E402

import jax.numpy as jnp            # noqa: E402

import coupling as C               # noqa: E402

# The hooks must exist before anything else is bound: without them coupling's
# main loop would slice the modal stack with NBINS=40 and settle modes as bins.
for _h in ('AER_NROWS', 'AER_FILL_FN', 'aer_to_bins', 'aer_settle'):
    if not hasattr(C, _h):
        sys.exit(f"driver_mam4: coupling.py has no '{_h}' hook -- apply the patch in "
                 f"runs/mam4_dev/STATUS.md (aerosol-representation seam) first")

# The adapter: below the setdefault block, as for tomas_fast (it imports coupling).
import mam4_adapter as M           # noqa: E402

# ---- the advection swap (same selection as driver_fast.py) ------------------
_ADV_CFL = float(os.environ.get('ADV_CFL', '0.5'))
_ADV_DTYPE = jnp.float32 if os.environ.get('ADV_F32', '1') != '0' else jnp.float64
_ADV_SCHEME = os.environ.get('ADV_SCHEME', 'ffsl').lower()
if _ADV_SCHEME == 'lr':
    from fct_lr import advect_step_batch as _adv_fn
elif _ADV_SCHEME == 'sl':
    import sl_lr
    _adv_fn = sl_lr.advect_step_batch
elif _ADV_SCHEME == 'ffsl':
    import ffsl_lr
    _adv_fn = ffsl_lr.advect_step_batch
else:
    from fct_fast import advect_step_batch as _adv_fn
C.advect_step_batch = functools.partial(_adv_fn, cfl=_ADV_CFL, dtype=_ADV_DTYPE)

# ---- the aerosol-representation hooks + the micro swap ----------------------
C.AER_NROWS = M.NMODE
C.AER_FILL_FN = M.modal_fill
C.aer_to_bins = M.aer_to_bins
C.aer_settle = M.settle_modal
C.run_microphysics_full = M.run_microphysics_full_mam4
if 'OH_SUBSTEPS' not in os.environ:
    C.OH_SUBSTEPS = 60

print("=" * 60, flush=True)
print(f"COUPLING: MAM4-JAX modal microphysics + {C.AER_SRC.upper()} IC/BC", flush=True)
print(f"  micro    : {M.describe()}", flush=True)
print(f"  state    : {M.NMODE} modes x (num, so4) = {2*M.NMODE} aerosol rows + SO2 + H2SO4 "
      f"= {2*M.NMODE+2} advected tracers; bins only for optics/settling/diagnostics "
      f"(modes_to_bins, {C.NBINS} bins)", flush=True)
print(f"  advection: {_ADV_SCHEME}  ({'f32' if _ADV_DTYPE is jnp.float32 else 'f64'})", flush=True)
print(f"  OH       : " + (f"SZA curve sampled {C.OH_SUBSTEPS}x/step, block-mean onto {M.N_INNER} core steps"
                          if C.OH_SZA else "CESM field, constant over step"), flush=True)
print("=" * 60, flush=True)

if __name__ == '__main__':
    C.main()
