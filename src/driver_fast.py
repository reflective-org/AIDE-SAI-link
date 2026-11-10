"""Fully-linked coupled run using the tomas-jax GPU-FAST reduced model.

Runs coupling.py with ONE difference: the microphysics engine is the
natively-batched reduced model `tomas_jax.fast` (gpu-fast branch) instead of the
hand-assembled per-cell chain (coupling.run_microphysics_full). Advection is the
same fct_lr swap coupling.py already uses, so the two agree on transport.

Since 2026-10-09 the seam takes a second engine: MICRO_ENGINE=emulator swaps in
the learned 6 h operator from /data/tomas-emulator instead of run_fast, and
MICRO_REF=<other engine> runs that one too, on the same input, for a per-step
A/B (docs/EMULATOR.md). The fast physics remains the default and is unchanged.

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

Run: use run_prod.py beside this file, which sets the whole production
environment (GPU pin,
libcuda path, memory policy) and execs this file. Direct invocation for a short
test, once tomas-jax and jax-rrtmgp are importable -- `git submodule update
--init`, or set TOMAS_JAX_PATH / RRTMGP_PATH; coupling.py resolves both:

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

if os.environ.get('STEP_HOURS', '6') != '6':
    sys.exit("driver_fast: STEP_HOURS must be 6 (parity with 6h step)")
# MICRO=off (transport only, for timing runs) is allowed through: it simply never
# calls the fast engine, so the 40-bin/6h parity constraints below do not apply to
# it, and a benchmark then measures advection through the PRODUCTION entry point
# rather than through a second launcher that can drift from it. The legacy 'coag'
# path is still refused -- this driver exists to swap the fast engine into the full
# chain, and pairing it with the pre-SAI model would silently be neither.
if os.environ.get('MICRO', 'full') not in ('full', 'off'):
    sys.exit("driver_fast: MICRO must be 'full' (or 'off' for a transport-only run)")
if os.environ.get('N_BINS', '0') not in ('0', '40'):
    sys.exit(f"driver_fast: fast model is 40-bin -- unset N_BINS or set "
             f"40; got N_BINS={os.environ.get('N_BINS')}")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _paths                      # noqa: E402 -- puts src/ subdirs on sys.path

import jax.numpy as jnp

# NB the compat shim that used to live here (aliasing coupling.py's imported
# calc_oh_parabola to tomas-jax's calc_oh_concentration) is GONE and must not come
# back: the two take their arguments in the opposite order and
# calc_oh_concentration defaults to constant mode, so the alias made
# calc_oh_parabola(cos_sza, OH_PEAK) return cos(SZA) -- OH ~ 1 molec/cm3, i.e. no
# SO2 oxidation at all -- in every OH_SZA=1 run. coupling.py now owns the fit.

import coupling as C
from fct_fast import advect_step_batch as _fast_advect
# The tomas_jax.fast adapter. MUST be imported here, BELOW the
# os.environ.setdefault block above and not at the top of the file: it imports
# coupling.py, and coupling.py reads the environment at import time, so hoisting
# this line changes the configuration the run uses. (This is one of the reasons
# ruff's import-sorting rule I001 is disabled -- see docs/REPO_LAYOUT.md.)
#
# MICRO_ENGINE (2026-10-09) selects WHICH batched engine fills the seam:
#   fast      (default) tomas_jax.fast.run_fast, 60 x 360 s inner steps -- the
#             production physics, unchanged.
#   emulator  the learned 6 h operator from /data/tomas-emulator (a bin-token
#             transformer trained on tomas_jax.fast trajectories; see
#             docs/EMULATOR.md and microphysics/tomas_emulator.py). ONE forward
#             pass per cell per step. Requires torch in the run environment.
# MICRO_REF (optional) names the OTHER engine, to be run on the SAME input every
# MICRO_REF_EVERY steps purely for comparison: the run advances with
# MICRO_ENGINE, the reference output is compared cell-by-cell and discarded, and
# the per-step error + the timing of BOTH engines go to micro_bench_<TAG>.npz
# (microphysics/micro_bench.py). This is how the emulator is benchmarked against
# the fast physics as ground truth without a second harness.
MICRO_ENGINE = os.environ.get('MICRO_ENGINE', 'fast').lower()
MICRO_REF = os.environ.get('MICRO_REF', '').lower()
MICRO_REF_EVERY = int(os.environ.get('MICRO_REF_EVERY', '1'))
_ENGINES = ('fast', 'emulator')
if MICRO_ENGINE not in _ENGINES:
    sys.exit(f"driver_fast: MICRO_ENGINE must be one of {_ENGINES}; got {MICRO_ENGINE!r}")
if MICRO_REF and MICRO_REF not in _ENGINES:
    sys.exit(f"driver_fast: MICRO_REF must be empty or one of {_ENGINES}; got {MICRO_REF!r}")
if MICRO_REF == MICRO_ENGINE:
    sys.exit("driver_fast: MICRO_REF names the same engine as MICRO_ENGINE -- "
             "the comparison would be a no-op")
_need = {MICRO_ENGINE} | ({MICRO_REF} if MICRO_REF else set())

# Both adapters import coupling.py, so they too must stay below the setdefault
# block above (see the note on tomas_fast). Each one is imported only when its
# engine is used: the emulator adapter needs torch, the fast adapter needs the
# tomas-jax gpu-fast branch, and a run that uses one should not require the
# other's dependencies.
_ENGINE_FN = {}
if 'fast' in _need:
    from tomas_fast import (run_microphysics_full_fast, N_FAST_STEPS, FAST_DT,
                            FAST_CELL_CAP, FAST_SORT)
    _ENGINE_FN['fast'] = run_microphysics_full_fast
if 'emulator' in _need:
    import tomas_emulator
    _ENGINE_FN['emulator'] = tomas_emulator.run_microphysics_full_emulator
# OH sampling: the SZA parabola is evaluated at C.OH_SUBSTEPS substep centers
# per step. tomas_fast sets it to its 60 inner steps at import; the emulator
# interpolates whatever it is given onto its 13 endpoints (tomas_emulator.
# _oh_to_endpoints), so 60 serves both and keeps the emulator's OH input
# identical to the physics' whatever the engine mix. An explicit OH_SUBSTEPS in
# the environment still wins (both adapters resample from any row count).
if 'OH_SUBSTEPS' not in os.environ:
    C.OH_SUBSTEPS = 60

# ---- the advection swap (same scheme coupling.py binds for itself) ----------
_ADV_CFL = float(os.environ.get('ADV_CFL', '0.5'))
_ADV_DTYPE = jnp.float32 if os.environ.get('ADV_F32', '1') != '0' else jnp.float64
# ADV_SCHEME selects the transport form:
#   ffsl (default since 2026-10-15) advection/ffsl_lr.py -- fct_lr at large
#        Courant number (Lin & Rood's flux-form semi-Lagrangian extension): the
#        same flux-form sweeps, faces, budgets and limiter at ADV_FFSL_K=24
#        sub-steps instead of 70-250. advect 1.5 s/step vs 7.0; 30 days within
#        0.2 % of lr on the budgets (docs/PROCESSES.md 1.9). 2026-10-14.
#   lr   (the default until 2026-10-15) advection/fct_lr.py -- Lin-Rood FLUX form with air-mass
#        tracking. Tracer mass is conserved to ROUNDOFF (~1e-15/day) for any wind
#        field, divergent or not, so it does not depend on the winds satisfying
#        continuity. Costs ~5x more substeps (no integer-shift FFSL in flux form).
#   fast advection/fct_fast.py -- advective form with the 2026-07-25 fixes
#        (true grid spacing, cos(phi) metric, continuity omega, conserving polar
#        caps). Residual ~3e-4/day; cheaper. Kept as the validated fallback.
#   sl   advection/sl_lr.py -- semi-Lagrangian (dinosaur's operator, vendored in
#        sl_core.py): no CFL limit, ADV_SL_K sub-steps on the hourly winds, cubic
#        gather + quasi-monotone limiter, closed column + per-tracer mass fixer
#        (ADV_SL_FIXER), Dirichlet edge levels (coupling.py sets ADV_WCONT=0).
#        ~49 ms/step vs 6.6 s. docs/TRANSPORT_SL_HANDOFF.md. 2026-10-13.
_ADV_SCHEME = os.environ.get('ADV_SCHEME', 'ffsl').lower()
if _ADV_SCHEME == 'lr':
    from fct_lr import advect_step_batch as _adv_fn
elif _ADV_SCHEME == 'sl':
    import sl_lr
    _adv_fn = sl_lr.advect_step_batch
elif _ADV_SCHEME == 'ffsl':
    import ffsl_lr                 # fct_lr at large Courant number (2026-10-14)
    _adv_fn = ffsl_lr.advect_step_batch
else:
    _adv_fn = _fast_advect
C.advect_step_batch = functools.partial(_adv_fn, cfl=_ADV_CFL, dtype=_ADV_DTYPE)

# ---- the ONE micro swap; advection already swapped above --------------------
_micro_fn = _ENGINE_FN[MICRO_ENGINE]
if MICRO_REF:
    from micro_bench import make_benchmarked
    _micro_fn = make_benchmarked(_micro_fn, MICRO_ENGINE, _ENGINE_FN[MICRO_REF],
                                 MICRO_REF, every=MICRO_REF_EVERY,
                                 tag=os.environ.get('OUT_TAG', 'tomas_fast'))
# MICRO_SAMPLE_EVERY=N (2026-10-13): every N steps write a stratified sample of
# the cells the engine is about to see (microphysics/micro_sampler.py) -- the
# emulator's training data from this model's own states. Outermost wrapper, so
# what is sampled is exactly the engine input. Needs the emulator adapter for
# its triage test, i.e. torch, even under MICRO_ENGINE=fast.
MICRO_SAMPLE_EVERY = int(os.environ.get('MICRO_SAMPLE_EVERY', '0'))
if MICRO_SAMPLE_EVERY > 0:
    import tomas_emulator  # noqa: F811 -- the sampler uses its triage test
    from micro_sampler import make_sampled
    _micro_fn = make_sampled(_micro_fn, every=MICRO_SAMPLE_EVERY,
                             n=int(os.environ.get('MICRO_SAMPLE_N', '5000')),
                             tag=os.environ.get('OUT_TAG', 'tomas_fast'))
C.run_microphysics_full = _micro_fn


def _describe(engine):
    if engine == 'fast':
        return (f"tomas_jax.fast.run_fast  ({N_FAST_STEPS} steps x {FAST_DT:.0f}s, "
                f"cap {FAST_CELL_CAP} cells/chunk, sort={FAST_SORT})")
    return tomas_emulator.describe()


print("=" * 60, flush=True)
print(f"COUPLING: {'tomas_jax.fast (GPU-fast reduced model)' if MICRO_ENGINE == 'fast' else 'TOMAS EMULATOR (learned 6 h operator)'} + "
      f"{C.AER_SRC.upper()} IC/BC", flush=True)
print(f"  micro    : {_describe(MICRO_ENGINE)}", flush=True)
if MICRO_SAMPLE_EVERY > 0:
    print(f"  SAMPLING : every {MICRO_SAMPLE_EVERY} step(s), {os.environ.get('MICRO_SAMPLE_N', '5000')} cells "
          f"-> micro_samples_{os.environ.get('OUT_TAG', 'tomas_fast')}_s*.npz", flush=True)
if MICRO_REF:
    print(f"  REFERENCE: {_describe(MICRO_REF)}\n"
          f"             run on the same input every {MICRO_REF_EVERY} step(s) for "
          f"comparison only -> micro_bench_{os.environ.get('OUT_TAG', 'tomas_fast')}.npz",
          flush=True)
print((f"  advection: advection/sl_lr  {sl_lr.describe()}  "
       f"({'f32' if _ADV_DTYPE is jnp.float32 else 'f64'} gathers)")
      if _ADV_SCHEME == 'sl' else
      (f"  advection: advection/ffsl_lr  {ffsl_lr.describe()}  "
       f"({'f32' if _ADV_DTYPE is jnp.float32 else 'f64'})")
      if _ADV_SCHEME == 'ffsl' else
      f"  advection: advection/fct_{_ADV_SCHEME}  (cfl={_ADV_CFL}, "
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
