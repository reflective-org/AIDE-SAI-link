#!/usr/bin/env python3
"""Integration test for src/microphysics/tomas_emulator.py, GPU required.

Two checks, neither of which needs the CESM archive:

  1. _oh_to_endpoints: substep-CENTER samples of a known parabola are
     interpolated onto the 13 endpoints to second order (24 samples -> every
     endpoint is the mean of its two 7.5-min neighbours).

  2. The adapter reproduces the handoff's reference accuracy on the emulator's
     own cached test cells (EMULATOR_HANDOFF.md section 4): v9_sai test, one
     6 h step, bintf_big_rebal fp16 compiled -> totN p50/p90 ~ 3.1/16 %, totM
     ~ 2.6/13 %. The cached arrays are log-space per-box states; this test
     turns them into LINEAR per-box quantities and feeds them through
     `emulate_cells`, i.e. through the exact log/floor/normalize/pad/denorm
     path the coupled run uses. If a floor convention, the channel order, or
     the batching were wrong, the numbers here would be far from the handoff's.
     The test passes if p50 is within 2 points and p90 within 6 points of the
     reference (the subset here is the first N cells, not the full set).

Run from anywhere (the adapter imports coupling.py, which reads its
configuration from the environment at import, so STEP_HOURS etc. are set here
to the production values first):

    CUDA_VISIBLE_DEVICES=4 LD_LIBRARY_PATH=/run/nvidia/driver/usr/lib/x86_64-linux-gnu \
        /data/aide_emu_venv/bin/python scripts/validation/test_emulator_adapter.py [N_CELLS]
"""
import os
import sys
import time

import numpy as np

os.environ.setdefault('STEP_HOURS', '6')
os.environ.setdefault('MICRO', 'full')
os.environ.setdefault('RAD', '0')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'src')
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, 'microphysics'))
import tomas_emulator as E          # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
CACHE = f"{E.EMU_ROOT}/cache"
BOX = 1.0e6
ok = True

# ---- 1. OH endpoint interpolation ------------------------------------------
nsub = 24
t_c = (np.arange(nsub) + 0.5) / nsub
f = lambda t: 1.0 + 3.0 * t - 2.0 * t ** 2            # a parabola in step fraction
oh4 = np.broadcast_to(f(t_c)[:, None, None, None], (nsub, 2, 3, 4)).copy()
got = E._oh_to_endpoints(oh4, 24)[0]
t_e = np.arange(13) / 12
want = f(t_e)
err = np.abs(got - want).max()
# interior: mean of neighbours 7.5 min apart -> error = curvature term 2*(1/48)^2;
# the two ends are linearly extrapolated over half a substep -> 3x that
print(f"[1] _oh_to_endpoints: max |interp - exact| = {err:.2e} (expect <= ~{6 * (1 / 48) ** 2:.1e}, "
      f"the parabola's curvature over 7.5 min)")
ok &= err < 5e-3

# ---- 2. cached test cells through the adapter -------------------------------
S = np.load(f"{CACHE}/v9_sai_test_S.npy", mmap_mode='r')
C = np.load(f"{CACHE}/v9_sai_test_C.npy", mmap_mode='r')
S0 = np.asarray(S[:N, 0], dtype=np.float64); S12 = np.asarray(S[:N, 12], dtype=np.float64)
C13 = np.asarray(C[:N, :13], dtype=np.float64)
# the adapter has no in-step source or dilution; keep only cells where the
# cached forcing agrees (channel 4 and 5 at their "none" value log(1e-30))
none = np.isclose(C13[:, :, 4], np.log(1e-30)).all(1) & np.isclose(C13[:, :, 5], np.log(1e-30)).all(1)
print(f"[2] v9_sai test: {N} cells, {int(none.sum())} with no in-step source/dilution (used)")
S0, S12, C13 = S0[none], S12[none], C13[none]
Nk = np.exp(S0[:, :40]); Mk = np.exp(S0[:, 40:80]); H = np.exp(S0[:, 80]); So2 = np.exp(S0[:, 81])
P = np.exp(C13[:, 0, 0]); T = C13[:, 0, 1]; RH = C13[:, 0, 2]; OH13 = np.exp(C13[:, :, 3])
# T/P/RH vary slightly along the cached 30-min grid; the adapter holds the
# step-start value, as coupling.py does. Check how much that matters below.
tv = np.abs(C13[:, :, 1] - C13[:, :1, 1]).max()
print(f"    (cached T varies by up to {tv:.2f} K within the window; the adapter uses t0 values)")

t0 = time.time()
Nk2, Mk2, H2, S2 = E.emulate_cells(Nk, Mk, H, So2, T, P, RH, OH13)
dt = time.time() - t0
t0 = time.time()
Nk2, Mk2, H2, S2 = E.emulate_cells(Nk, Mk, H, So2, T, P, RH, OH13)
dt2 = time.time() - t0
print(f"    emulate_cells: first call {dt:.1f}s (compile), second {dt2:.2f}s = {dt2 / Nk.shape[0] * 1e6:.2f} us/cell")

tN_r = np.exp(S12[:, :40]).sum(1); tM_r = np.exp(S12[:, 40:80]).sum(1)
rel = lambda p, r: np.abs(p - r) / np.maximum(r, 1e-25)
active = np.exp(S0[:, :40]).sum(1) / BOX > 1.0
eN = rel(Nk2.sum(1), tN_r)[active] * 100; eM = rel(Mk2.sum(1), tM_r)[active] * 100
eS = rel(S2, np.exp(S12[:, 81]))[active] * 100; eH = rel(H2, np.exp(S12[:, 80]))[active] * 100
pN = np.percentile(eN, (50, 90)); pM = np.percentile(eM, (50, 90))
pS = np.percentile(eS, (50, 90)); pH = np.percentile(eH, (50, 90))
print(f"    6 h totals rel.err p50/p90 % on {int(active.sum())} active cells: "
      f"totN {pN[0]:.1f}/{pN[1]:.1f}  totM {pM[0]:.1f}/{pM[1]:.1f}  SO2 {pS[0]:.1f}/{pS[1]:.1f}  H2SO4 {pH[0]:.1f}/{pH[1]:.1f}")
print(f"    handoff reference (all v9_sai test cells, bintf_big_rebal fp16 compiled): totN 3.1/16  totM 2.6/13")
ok &= abs(pN[0] - 3.1) < 2.0 and abs(pN[1] - 16) < 6.0 and abs(pM[0] - 2.6) < 2.0 and abs(pM[1] - 13) < 6.0
# a deliberately WRONG floor must make it worse -- guards the convention
Nk2w, Mk2w, _, _ = E.emulate_cells(np.maximum(Nk, 1e-30), np.maximum(Mk, 1e-20), H, So2, T, P, RH, OH13)
pNw = np.percentile(rel(Nk2w.sum(1), tN_r)[active] * 100, 50)
print(f"    control: swapped floors (N 1e-30, M 1e-20) -> totN p50 {pNw:.1f} % (should be clearly worse)")

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
