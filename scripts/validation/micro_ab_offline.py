#!/usr/bin/env python3
"""Offline microphysics A/B on a dumped pre-micro state (no CESM reads, no run loop).

Input: the npz written by DUMP_PREMICRO=1 [DUMP_PREMICRO_STEP=n] -- the exact
post-advection state and forcing the step loop hands to the microphysics
(num, mas, so2, h2so4, T3d, pres3d, rh3d, oh3d, lat, lon, plev). Both engines
are run on it through their production adapters (same mixing-ratio <->
per-box conversion, same two-moment clip), compared with micro_bench.compare,
and the WORST cells are printed with their inputs, which is what the in-loop
benchmark cannot afford to keep. Deterministic, so a fix to an adapter can be
checked against the same state in a minute instead of a run.

    CUDA_VISIBLE_DEVICES=6 LD_LIBRARY_PATH=/run/nvidia/driver/usr/lib/x86_64-linux-gnu \
      /data/aide_emu_venv/bin/python scripts/validation/micro_ab_offline.py \
      runs/emu_dumps/debug_nucleation/premicro_state.npz --out runs/emu_dumps/ab_step0

Writes <out>.npz (per-cell totals of both engines + inputs) and <out>.txt.
"""
import argparse
import os
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('dump')
ap.add_argument('--out', default=None)
ap.add_argument('--primary', default='emulator')
ap.add_argument('--reference', default='fast')
ap.add_argument('--top', type=int, default=20)
ap.add_argument('--rank', default='N', choices=('N', 'M'),
                help='rank the worst cells by burden-weighted number (N) or mass (M) difference')
args = ap.parse_args()

os.environ.setdefault('STEP_HOURS', '6')
os.environ.setdefault('MICRO', 'full')
os.environ.setdefault('RAD', '0')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('FAST_CELL_CAP', '50000')
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'src')
sys.path.insert(0, SRC); sys.path.insert(0, os.path.join(SRC, 'microphysics'))
import coupling as C                 # noqa: E402
import micro_bench                   # noqa: E402

ENG = {}
if 'emulator' in (args.primary, args.reference):
    import tomas_emulator            # noqa: E402
    ENG['emulator'] = tomas_emulator.run_microphysics_full_emulator
if 'fast' in (args.primary, args.reference):
    import tomas_fast                # noqa: E402
    ENG['fast'] = tomas_fast.run_microphysics_full_fast
if 'OH_SUBSTEPS' not in os.environ:
    C.OH_SUBSTEPS = 60

out = args.out or os.path.splitext(args.dump)[0] + '_ab'
z = np.load(args.dump)
num, mas, so2, h2so4 = z['num'], z['mas'], z['so2'], z['h2so4']
T3d, P3d, RH3d, OH = z['T3d'], z['pres3d'], z['rh3d'], z['oh3d']
lat, lon, plev = z['lat'], z['lon'], z['plev']
nbin, nlev, nlat, nlon = num.shape
ncell = nlev * nlat * nlon
# burden weight as coupling builds it: dp * (sin-difference latitude weight)
dp = np.abs(np.gradient(plev))
_e = np.deg2rad(np.concatenate([[-90.0], 0.5 * (lat[1:] + lat[:-1]), [90.0]]))
wlat = (np.sin(_e[1:]) - np.sin(_e[:-1])) / np.deg2rad(lat[1] - lat[0])
A = np.broadcast_to((dp[:, None] * wlat[None, :])[:, :, None], (nlev, nlat, nlon)).copy()

lines = []
def say(s=''):
    print(s, flush=True); lines.append(s)

say(f"dump {args.dump}: {nbin} bins x {nlev} lev x {nlat} x {nlon} = {ncell} cells; OH {OH.shape}")
res = {}
for name in (args.primary, args.reference):
    fn = ENG[name]
    # warm-up on a slab so timing is steady-state (JIT / torch.compile)
    t0 = time.time(); fn(num[:, :2], mas[:, :2], so2[:2], h2so4[:2], T3d[:2], P3d[:2], RH3d[:2],
                         OH[:, :2] if OH.ndim == 4 else OH[:2], A[:2]); tw = time.time() - t0
    t0 = time.time(); r = fn(num, mas, so2, h2so4, T3d, P3d, RH3d, OH, A); dt = time.time() - t0
    res[name] = (r, dt)
    say(f"{name:9s}: warm-up {tw:.1f}s, full grid {dt:.2f}s = {dt / ncell * 1e6:.2f} us/cell")

rho_f = (P3d / (C.RD * T3d)).reshape(ncell)
d = micro_bench.compare(res[args.primary][0][:4], res[args.reference][0][:4], rho_f,
                        A.reshape(ncell), (nlev, nlat, nlon), nbin)
P_, R_ = args.primary, args.reference
say(f"\none-step |{P_} - {R_}| / {R_}, rel. error p50/p90/p99 [%]:")
for k in ('totN', 'totM', 'so2', 'h2so4'):
    say(f"  {k:6s} all {d[k+'_all'][0]*100:7.2f}/{d[k+'_all'][1]*100:7.2f}/{d[k+'_all'][2]*100:8.1f}   "
        f"active({d['n_active']}) {d[k+'_active'][0]*100:7.2f}/{d[k+'_active'][1]*100:7.2f}/{d[k+'_active'][2]*100:8.1f}   "
        f"plume({d['n_plume']}) {d[k+'_plume'][0]*100:7.2f}/{d[k+'_plume'][1]*100:7.2f}/{d[k+'_plume'][2]*100:8.1f}   "
        f"global burden-wtd {d[k+'_global']*100:+9.2f}")
say(f"  burst cells {d['n_burst']}: per-bin N {d['burst_p'][0]*100:.0f}/{d['burst_p'][1]*100:.0f}/{d['burst_p'][2]*100:.0f} %;"
    f"  W1(bins) active {d['W1_active'][0]:.2f}/{d['W1_active'][1]:.2f}")
say("  per-level totN p50/p90 % (active cells)  |  signed global N err contributed by level [%]")
Np = (res[P_][0][0].reshape(nbin, ncell).T * rho_f[:, None]).sum(1)
Nr = (res[R_][0][0].reshape(nbin, ncell).T * rho_f[:, None]).sum(1)
Mp = (res[P_][0][1].reshape(nbin, ncell).T * rho_f[:, None]).sum(1)
Mr = (res[R_][0][1].reshape(nbin, ncell).T * rho_f[:, None]).sum(1)
w = A.reshape(ncell)
den = (Nr * w).sum()
for k in range(nlev):
    sl = slice(k * nlat * nlon, (k + 1) * nlat * nlon)
    contrib = ((Np[sl] - Nr[sl]) * w[sl]).sum() / den * 100
    say(f"    lev {k:2d} {plev[k]/100:7.2f} hPa  {d['lev_totN_p50'][k]*100:6.2f}/{d['lev_totN_p90'][k]*100:6.2f}   |  {contrib:+9.2f}")

# ---- sulfur conservation per cell, both engines -------------------------------
# a closed box with no in-step source conserves total sulfur exactly; the
# physics should hold it to roundoff, and how far the emulator is from it is
# the cleanest single number for "is it making mass up"
N0 = (num.reshape(nbin, ncell).T * rho_f[:, None]); M0 = (mas.reshape(nbin, ncell).T * rho_f[:, None])
S0 = so2.reshape(ncell) * rho_f; H0 = h2so4.reshape(ncell) * rho_f
def sulfur(Mtot, S, H):
    return Mtot * (32.0 / 96.0) + S * (32.0 / 64.0) + H * (32.0 / 98.0)
S_in = sulfur(M0.sum(1), S0, H0)
say(f"\nsulfur conservation |S_out - S_in| / S_in per cell, p50/p90/p99/max [%]:")
for name, (Mx, Sx, Hx) in ((P_, (Mp, res[P_][0][2].reshape(ncell) * rho_f, res[P_][0][3].reshape(ncell) * rho_f)),
                           (R_, (Mr, res[R_][0][2].reshape(ncell) * rho_f, res[R_][0][3].reshape(ncell) * rho_f))):
    dS = np.abs(sulfur(Mx, Sx, Hx) - S_in) / np.maximum(S_in, 1e-30)
    q = np.percentile(dS, (50, 90, 99)) * 100
    say(f"  {name:9s} {q[0]:8.3f}/{q[1]:8.3f}/{q[2]:8.2f}/{dS.max()*100:8.1f}   "
        f"(burden-weighted signed: {((sulfur(Mx, Sx, Hx) - S_in) * w).sum() / (S_in * w).sum() * 100:+.3f} %)")

# ---- worst cells by burden-weighted signed number (or mass) difference ---------
OHm = OH.reshape(OH.shape[0], ncell).mean(0) if OH.ndim == 4 else OH.reshape(ncell)
NA = 6.02214076e23
to_cm3 = lambda kg, mw: kg / (mw / 1000.0) * NA / 1e6     # kg/box -> molec/cm3 (box = 1 m3)
if args.rank == 'M':
    contrib = (Mp - Mr) * w; den = (Mr * w).sum()
else:
    contrib = (Np - Nr) * w
Q = args.rank
say(f"\nworst {args.top} cells by w*({Q}_{P_} - {Q}_{R_}) (share of the global signed {Q} error):")
order = np.argsort(-contrib)
tot = contrib.sum()
say(f"  (total signed {Q} error {tot / den * 100:+.1f} %; top {args.top} cells carry "
    f"{contrib[order[:args.top]].sum() / tot * 100:.1f} % of it, top 1000: "
    f"{contrib[order[:1000]].sum() / tot * 100:.1f} %, top 1 % of cells: "
    f"{contrib[order[:ncell // 100]].sum() / tot * 100:.1f} %)")
say("  idx      lev(hPa)  lat    lon   T[K]  RH    OH[cm-3]  SO2[cm-3] H2SO4[cm-3] N0[cm-3]  M0[kg/m3]  "
    f"N_{R_}[cm-3] N_{P_}[cm-3]  M_{R_}    M_{P_}   | bins>1cm-3 in {P_} (first..last) | share%")
for i in order[:args.top]:
    kl, jl, il = np.unravel_index(i, (nlev, nlat, nlon))
    occ = np.where(res[P_][0][0].reshape(nbin, ncell)[:, i] * rho_f[i] / 1e6 > 1.0)[0]
    say(f"  {i:7d} {plev[kl]/100:8.2f} {lat[jl]:6.1f} {lon[il]:6.1f} {T3d.reshape(ncell)[i]:6.1f} "
        f"{RH3d.reshape(ncell)[i]:5.2f} {OHm[i]:9.2e} {to_cm3(S0[i], 64):9.2e} {to_cm3(H0[i], 98):10.2e} "
        f"{N0[i].sum()/1e6:8.2e} {M0[i].sum():9.2e}  {Nr[i]/1e6:10.2e} {Np[i]/1e6:10.2e} {Mr[i]:9.2e} {Mp[i]:9.2e} "
        f"| {occ[0] if occ.size else '-'}..{occ[-1] if occ.size else '-'} ({occ.size}) | {contrib[i] / tot * 100:.2f}")
# the input size distribution of the worst cell, for the eye
i = order[0]
XK = C.XK_NP
occ0 = N0[i] / 1e6 > 1e-3
say(f"  worst cell input: N[cm-3] by bin " + " ".join(f"{k}:{N0[i, k]/1e6:.1e}" for k in np.where(occ0)[0]))
say(f"                    (M/N)/xk_lo    " + " ".join(f"{k}:{M0[i, k]/max(N0[i, k],1e-300)/XK[k]:.2f}" for k in np.where(occ0)[0]))
Npk = res[P_][0][0].reshape(nbin, ncell)[:, i] * rho_f[i]; Nrk = res[R_][0][0].reshape(nbin, ncell)[:, i] * rho_f[i]
say(f"  {P_} output N[cm-3]: " + " ".join(f"{k}:{Npk[k]/1e6:.1e}" for k in range(nbin) if Npk[k] / 1e6 > 1e-3))
say(f"  {R_} output N[cm-3]: " + " ".join(f"{k}:{Nrk[k]/1e6:.1e}" for k in range(nbin) if Nrk[k] / 1e6 > 1e-3))

# training-envelope flags (handoff section 3)
so2_cm3 = to_cm3(S0, 64); ohmax = OH.reshape(OH.shape[0], ncell).max(0) if OH.ndim == 4 else OHm
flag = (so2_cm3 > 2.9e15) | (so2_cm3 * ohmax > 2.2e21) | (ohmax > 2.5e7)
say(f"\ncells outside the handoff's training envelope (SO2>2.9e15 or SO2*OH>2.2e21 or OH>2.5e7): "
    f"{int(flag.sum())} ({flag.mean()*100:.3f} %); they carry {contrib[flag].sum() / tot * 100:.1f} % of the signed N error")
say(f"input ranges: T {T3d.min():.1f}-{T3d.max():.1f} K, P {P3d.min():.1f}-{P3d.max():.0f} Pa, RH {RH3d.min():.3f}-{RH3d.max():.3f}, "
    f"OH max {ohmax.max():.2e}, SO2 max {so2_cm3.max():.2e} cm-3, H2SO4 max {to_cm3(H0, 98).max():.2e} cm-3")

np.savez(out + '.npz', N_pred=Np.astype(np.float32), N_ref=Nr.astype(np.float32),
         M_pred=Mp.astype(np.float32), M_ref=Mr.astype(np.float32),
         S_pred=(res[P_][0][2].reshape(ncell) * rho_f).astype(np.float32),
         S_ref=(res[R_][0][2].reshape(ncell) * rho_f).astype(np.float32),
         H_pred=(res[P_][0][3].reshape(ncell) * rho_f).astype(np.float32),
         H_ref=(res[R_][0][3].reshape(ncell) * rho_f).astype(np.float32),
         N0=N0.sum(1).astype(np.float32), M0=M0.sum(1).astype(np.float32),
         S0=S0.astype(np.float32), H0=H0.astype(np.float32), OHmean=OHm.astype(np.float32),
         T=T3d.reshape(ncell).astype(np.float32), P=P3d.reshape(ncell).astype(np.float32),
         RH=RH3d.reshape(ncell).astype(np.float32), w=w.astype(np.float32),
         shape=np.array([nlev, nlat, nlon]), plev=plev, lat=lat, lon=lon,
         primary=P_, reference=R_, t_primary=res[P_][1], t_reference=res[R_][1],
         **{f'cmp_{k}': np.asarray(v) for k, v in d.items()})
open(out + '.txt', 'w').write('\n'.join(lines) + '\n')
say(f"\nwrote {out}.npz and {out}.txt")
