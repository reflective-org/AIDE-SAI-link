#!/usr/bin/env python3
"""Physics-labelled 30-minute chains from sampled coupled-model states.

Input: `micro_samples_<TAG>_s<step>.npz` files written by
src/microphysics/micro_sampler.py (MICRO_SAMPLE_EVERY) -- per-box states and
forcing of cells the microphysics was about to integrate. Output: for each
input file, `<out_dir>/aide_chains_<TAG>_s<step>.npz` holding

    S  (n, T, 82) float32   log per-box state on the 30-min grid, T = 2*hours+1
    C  (n, T, 6)  float32   [log P, T, RH, log OH, log SO2-source, log k_dil]
    stratum (n,) int8, lat/lon/plev_pa/hour_abs (n,)

in exactly the emulator's dataset_v8 conventions (frontier_common / the
kchain trainer read it as (S, C) pairs), with the SAME input floors the
production adapter applies (tomas_emulator: Nk 1e-20, Mk/H2SO4 1e-30, SO2
1e-17 kg/box, OH 100 cm-3, RH in [0.01, 1]) so that what the network is
trained on is what it will be fed.

The label is `tomas_jax.fast.run_fast` -- the production physics, with the
production knobs (FAST_DT, ALPHA_COND, FAST_FN_SCALE, caps) -- run as a box
model from the sampled state: T, P, RH held at the sampled values, OH the
coupled model's own diurnal SZA parabola continued past the step at the
cell's latitude, longitude and absolute hour, no in-step source, no dilution.
States are recorded every 30 min (every FAST_DT*5 = 1800 s of inner steps).

    CUDA_VISIBLE_DEVICES=6 LD_LIBRARY_PATH=/run/nvidia/driver/usr/lib/x86_64-linux-gnu \
      /data/aide_emu_venv/bin/python scripts/emulator/label_chains.py \
      'runs/bench_phys30d_v2/micro_samples_*.npz' --hours 48 --out /data/tomas-jax-output/aide_chains
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('pattern', help='glob of micro_samples_*.npz (quote it)')
ap.add_argument('--hours', type=float, default=48.0)
ap.add_argument('--out', required=True)
ap.add_argument('--batch', type=int, default=40000, help='cells per run_fast batch')
ap.add_argument('--chunks', type=int, default=1, help='n_cell_chunks per batch')
ap.add_argument('--max-per-file', type=int, default=0)
ap.add_argument('--overwrite', action='store_true')
args = ap.parse_args()

os.environ.setdefault('STEP_HOURS', '6')
os.environ.setdefault('MICRO', 'full')
os.environ.setdefault('RAD', '0')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'src')
sys.path.insert(0, SRC); sys.path.insert(0, os.path.join(SRC, 'microphysics'))
import coupling as C                                   # noqa: E402
import tomas_fast as TF                                # noqa: E402  (production knobs)
import jax                                             # noqa: E402
import jax.numpy as jnp                                # noqa: E402
from tomas_jax.fast import FastState, run_fast         # noqa: E402
from tomas_jax.fast.config import SRTSO4, GH2SO4, GSO2  # noqa: E402

# the adapter's floors, restated (importing tomas_emulator would load the network)
N_FLOOR, M_FLOOR, S_FLOOR = 1e-20, 1e-30, float(os.environ.get('EMU_SO2_FLOOR', '1e-17'))
OH_FLOOR, RH_MIN = float(os.environ.get('EMU_OH_FLOOR', '100')), 0.01
LOG_ZERO = float(np.log(1e-30))

DT = TF.FAST_DT                                        # 360 s
STEPS_PER_HALF_HOUR = int(round(1800.0 / DT))          # 5
T_GRID = int(round(args.hours * 2)) + 1
N_INNER = (T_GRID - 1) * STEPS_PER_HALF_HOUR
print(f"chains of {args.hours:g} h = {T_GRID} points, {N_INNER} inner steps of {DT:.0f} s; "
      f"physics knobs: alpha {C.ALPHA_COND}, fn_scale {TF.FAST_FN_SCALE}, caps {TF.FAST_COAG_CAP}/{TF.FAST_COND_CAP}, cmax {TF.FAST_COAG_CMAX}",
      flush=True)
os.makedirs(args.out, exist_ok=True)


def oh_series(lat, lon, hour_abs, n_inner):
    """(n_inner, n) OH at inner-step CENTERS and (T_GRID, n) at the 30-min
    endpoints, from coupling's SZA parabola (the coupled model's OH source)."""
    n = lat.shape[0]
    lat_j = jnp.asarray(lat); lon_j = jnp.asarray(lon)

    def at(hr):
        doy = np.floor(hr // 24) % 365 + 1
        hour_utc = hr % 24.0
        cs = C.calc_solar_zenith_angle(lat_j, jnp.asarray(doy), jnp.asarray(hour_utc), lon_j)
        return np.asarray(C.calc_oh_parabola(cs, jnp.asarray(C.OH_PEAK)))

    inner = np.empty((n_inner, n)); ends = np.empty((T_GRID, n))
    for m in range(n_inner):
        inner[m] = at(hour_abs + (m + 0.5) * DT / 3600.0)
    for k in range(T_GRID):
        ends[k] = at(hour_abs + k * 0.5)
    return inner, ends


files = sorted(glob.glob(args.pattern))
if not files:
    sys.exit(f"no files match {args.pattern}")
for fn in files:
    tag = os.path.basename(fn).replace('micro_samples_', '').replace('.npz', '')
    out_fn = os.path.join(args.out, f'aide_chains_{tag}.npz')
    if os.path.exists(out_fn) and not args.overwrite:
        print(f"skip {out_fn} (exists)", flush=True); continue
    z = np.load(fn)
    n_all = z['Nk'].shape[0] if not args.max_per_file else min(args.max_per_file, z['Nk'].shape[0])
    S_out = np.empty((n_all, T_GRID, 82), np.float32); C_out = np.empty((n_all, T_GRID, 6), np.float32)
    t_file = time.time()
    for a in range(0, n_all, args.batch):
        b = min(a + args.batch, n_all); n = b - a
        Nk = z['Nk'][a:b].astype(np.float64); Mk = z['Mk'][a:b].astype(np.float64)
        H = z['H'][a:b].astype(np.float64); S = z['S'][a:b].astype(np.float64)
        T = z['T'][a:b].astype(np.float64); P = z['P'][a:b].astype(np.float64); RH = z['RH'][a:b].astype(np.float64)
        hour_abs = float(z['hour_abs'][a])
        oh_in, oh_end = oh_series(z['lat'][a:b].astype(np.float64), z['lon'][a:b].astype(np.float64), hour_abs, N_INNER)
        # forcing in dataset_v8 conventions, with the adapter's floors
        C_out[a:b, :, 0] = np.log(P)[None, :].T
        C_out[a:b, :, 1] = T[None, :].T
        C_out[a:b, :, 2] = np.clip(RH, RH_MIN, 1.0)[None, :].T
        C_out[a:b, :, 3] = np.log(np.maximum(oh_end, OH_FLOOR)).T
        C_out[a:b, :, 4] = LOG_ZERO; C_out[a:b, :, 5] = LOG_ZERO
        Mk3 = np.zeros((n, 40, 2)); Mk3[:, :, SRTSO4] = Mk
        Gc = np.zeros((n, 2)); Gc[:, GH2SO4] = H; Gc[:, GSO2] = S
        state = FastState.create(Nk=Nk, Mk=Mk3, Gc=Gc, xk=np.asarray(C.XK), temp=T, pres=P,
                                 boxvol=C.BOXVOL, rh=RH)

        def record(k, st):
            N_ = np.asarray(st.Nk); M_ = np.asarray(st.Mk[:, :, SRTSO4]); G_ = np.asarray(st.Gc)
            S_out[a:b, k, 0:40] = np.log(np.maximum(N_, N_FLOOR))
            S_out[a:b, k, 40:80] = np.log(np.maximum(M_, M_FLOOR))
            S_out[a:b, k, 80] = np.log(np.maximum(G_[:, GH2SO4], M_FLOOR))
            S_out[a:b, k, 81] = np.log(np.maximum(G_[:, GSO2], S_FLOOR))

        record(0, state)
        t0 = time.time()
        for k in range(1, T_GRID):
            sl = slice((k - 1) * STEPS_PER_HALF_HOUR, k * STEPS_PER_HALF_HOUR)
            state, _ = run_fast(state, STEPS_PER_HALF_HOUR, dt=DT, oh_conc=oh_in[sl],
                                n_cell_chunks=args.chunks, sort_by_coag_cost=(args.chunks > 1),
                                alpha=C.ALPHA_COND, fn_scale=TF.FAST_FN_SCALE,
                                coag_sub_cap=TF.FAST_COAG_CAP, cond_sub_cap=TF.FAST_COND_CAP,
                                coag_c_max=TF.FAST_COAG_CMAX)
            record(k, state)
        print(f"  {tag}: cells {a}-{b} of {n_all} labelled in {time.time() - t0:.0f}s "
              f"({(time.time() - t0) / n / (T_GRID - 1) * 1e6:.1f} us/cell/half-hour)", flush=True)
    ok = np.isfinite(S_out).all(axis=(1, 2))
    if not ok.all():
        print(f"  WARNING {tag}: {(~ok).sum()} chains non-finite, dropped", flush=True)
    np.savez(out_fn, S=S_out[ok], C=C_out[ok], stratum=z['stratum'][:n_all][ok],
             lat=z['lat'][:n_all][ok], lon=z['lon'][:n_all][ok], plev_pa=z['plev_pa'][:n_all][ok],
             hour_abs=z['hour_abs'][:n_all][ok], lev=z['lev'][:n_all][ok], source=fn,
             hours=args.hours, dt=DT)
    print(f"wrote {out_fn}: {int(ok.sum())} chains x {T_GRID} points in {time.time() - t_file:.0f}s", flush=True)
print("DONE", flush=True)
