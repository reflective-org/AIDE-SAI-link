#!/usr/bin/env python3
"""Validate the ADV_VPOS vertical positivity limiter.

A  POSITIVITY   : real 40-bin number state, one 6h step -> no negatives left.
B  CONSERVATION : telescoping identity sum(dp*q') == sum(dp*q) + F_top - F_bot
                  must hold EXACTLY for both limited and unlimited.
C  INACTIVITY   : on a SMOOTH positive field the limiter must not fire at all
                  -> bit-identical to unlimited. This is what protects the
                  validated LR/N2O accuracy result.
D  MULTI-STEP   : 40 consecutive steps (10 days) of pure advection on the real
                  number field: does the limiter stay stable and keep the floor
                  at zero, and how much does the burden trajectory differ?
"""
import os, sys, importlib, numpy as np
import os as _os
# _REPO/_os were REFERENCED but never DEFINED here between the 2026-07-29 move
# into validation/ and 2026-07-30, so this harness died on its 2nd line -- the
# prologue that floor_anatomy.py got was not copied across. Repo root carries
# coupling/settling/radiation; fast_advection/ carries fct_lr and fct_fast.
# three dirnames: scripts/validation/<this file> -> scripts/validation ->
# scripts -> the repo root. Was two until the 2026-08-27 move into scripts/.
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))))
sys.path.insert(0, _os.path.join(_REPO, 'src'))
import _paths                      # noqa: E402 -- puts src/ subdirs on sys.path
import jax, jax.numpy as jnp
import xarray as xr

PS_REF = 1.0e5
LAT_FREEZE = 80.0
STEP_SEC = 6 * 3600.0
DEG = np.pi / 180.0
HDIR = ('/data/cesm2.1.5_output/histSST/'
        'f.e21.FWHIST.f09_f09_mg17.atmos-scale_fixedSST_1996-2014.001/'
        'archive/atm/proc/tseries')
PREFIX = 'f.e21.FWHIST.f09_f09_mg17.atmos-scale_fixedSST_1996-2014.001.cam'
H1 = f'{HDIR}/hour_1/{PREFIX}.h1'
SUF = '.1996010100-2014123100.nc'

# Checkpoint from the CURRENT DIRECTORY, like every other script here. This used
# to default to a path inside the repo, which stopped resolving once run outputs
# moved to a runs directory outside the tree (see README) -- run this from there.
STATE = _os.environ.get('STATE', 'coupled_state_prod90d_ckpt.npz')
if not _os.path.exists(STATE):
    raise SystemExit(
        f"validate_vpos_f32: no checkpoint at {STATE!r}.\n"
        f"  Run this from the directory holding your run output, or set\n"
        f"  STATE=/path/to/coupled_state_<TAG>_ckpt.npz")
st = np.load(STATE)
num = np.asarray(st['num'])
NBINS, nlev, nlat, nlon = num.shape
h0 = int(st['s_done']) * int(st['step_hours'])

dU = xr.open_dataset(f'{H1}.U{SUF}', decode_times=False)
dV = xr.open_dataset(f'{H1}.V{SUF}', decode_times=False)
dW = xr.open_dataset(f'{H1}.OMEGA{SUF}', decode_times=False)
lat = dU['lat'].values
plev = dU['hyam'].values * dU['P0'].values + dU['hybm'].values * PS_REF
band = np.where((plev >= 100.0) & (plev <= 15000.0))[0]
klevs = list(range(band.min(), band.max() + 1))
PLEV_PA = plev[klevs]; DP = np.gradient(PLEV_PA)
assert len(klevs) == nlev, (len(klevs), nlev)

W_LAT = np.cos(lat * DEG)
A3 = (DP[:, None] * W_LAT[None, :])[None, :, :, None]
burden = lambda x: float((x * A3).sum())

def winds(t):
    return (dU['U'].isel(time=t, lev=klevs).values,
            dV['V'].isel(time=t, lev=klevs).values,
            dW['OMEGA'].isel(time=t, lev=klevs).values)

from coupling import DP_BIN
DPB = np.asarray(DP_BIN)[:NBINS]


def run_step(mod, q, t):
    u0, v0, w0 = winds(t); u1, v1, w1 = winds(t + 6)
    out = mod.advect_step_batch(jnp.asarray(q),
                               jnp.asarray(u0), jnp.asarray(v0), jnp.asarray(w0),
                               jnp.asarray(u1), jnp.asarray(v1), jnp.asarray(w1),
                               lat=lat, dp=DP, qfrozb=jnp.asarray(q),
                               lat_freeze=LAT_FREEZE, dt_total=STEP_SEC,
                               cfl=float(os.environ.get('ADV_CFL','0.5')),
                               dtype=(jnp.float32 if os.environ.get('ADV_F32','1')!='0'
                                      else jnp.float64),
                               return_vflux=True)
    return np.asarray(out[0]), np.asarray(out[2])


def load(vpos):
    os.environ['ADV_VPOS'] = '1' if vpos else '0'
    import fct_lr
    importlib.reload(fct_lr)
    assert fct_lr.VPOS == vpos
    return fct_lr


print('=' * 74)
print('A/B  POSITIVITY + CONSERVATION, real 40-bin number state, one 6h step')
print('=' * 74)
res = {}
for vpos in (False, True):
    m = load(vpos)
    adv, vfl = run_step(m, num, h0)
    Nin = burden(num); Nout = burden(adv)
    fl = burden(np.maximum(adv, 0.0)) - Nout
    # telescoping identity: interior change must equal the two boundary fluxes
    AF = (W_LAT[:, None] * np.ones((nlat, nlon)))[None, :, :]
    ftop = float((vfl[:, 0] * AF).sum()); fbot = float((vfl[:, 1] * AF).sum())
    ident = (Nout - Nin) - (ftop - fbot)
    tag = 'VPOS=1' if vpos else 'VPOS=0'
    res[vpos] = adv
    print(f'  {tag}:')
    print(f'    negative cells      {int((adv<0).sum()):>10d} of {adv.size}')
    print(f'    most negative value {adv.min():.4e}')
    print(f'    floor would add     {fl:.4e}  ({100*fl/Nin:+.5f}% of burden)')
    print(f'    telescoping residual {ident:+.4e}  -> relative {ident/Nin:+.3e}')

d = res[True] - res[False]
print('\n  where the limiter changed the answer (per bin, burden units):')
dpb = (np.abs(d) * A3).sum(axis=(1, 2, 3))
npb = (num * A3).sum(axis=(1, 2, 3))
tot = dpb.sum()
for k in range(NBINS):
    if dpb[k] / max(tot, 1e-300) < 1e-4:
        continue
    print(f'    bin {k:2d} Dp={DPB[k]:7.1f}nm  |delta| {dpb[k]:.3e}  '
          f'= {100*dpb[k]/max(npb[k],1e-300):8.4f}% of that bin  '
          f'({100*dpb[k]/tot:5.2f}% of all change)')
opt = (DPB >= 150) & (DPB <= 1200)
print(f'    optically active bins (150-1200nm): {100*dpb[opt].sum()/tot:.4f}% '
      f'of the change, = {100*dpb[opt].sum()/max(npb[opt].sum(),1e-300):.6f}% of their burden')

print('\n' + '=' * 74)
print('C  INACTIVITY ON A SMOOTH FIELD (must be bit-identical)')
print('=' * 74)
# a smooth, strictly positive tracer: no steep gradients -> limiter must not fire
lon = dU['lon'].values
LL, LO = np.meshgrid(lat, lon, indexing='ij')
prof = np.exp(-PLEV_PA / 5000.0)[:, None, None]
smooth = (1.0 + 0.5 * np.cos(LL * DEG) * np.sin(LO * DEG))[None, :, :] * prof
smooth = np.broadcast_to(smooth, (2, nlev, nlat, nlon)).copy()
outs = {}
for vpos in (False, True):
    m = load(vpos)
    outs[vpos], _ = run_step(m, smooth, h0)
diff = np.abs(outs[True] - outs[False]).max()
print(f'  max|VPOS=1 - VPOS=0| on the smooth field: {diff:.3e}')
print(f'  bit-identical: {diff == 0.0}')
print(f'  (negatives produced on the smooth field, VPOS=0: '
      f'{int((outs[False]<0).sum())})')

# also: the real MASS field, which is much smoother than number
mas = np.asarray(st['mas'])
outs2 = {}
for vpos in (False, True):
    m = load(vpos)
    outs2[vpos], _ = run_step(m, mas, h0)
Mrel = np.abs(outs2[True] - outs2[False]).sum() / max(np.abs(outs2[False]).sum(), 1e-300)
print(f'  real MASS field: relative change from limiting = {Mrel:.3e}  '
      f'(negatives VPOS=0: {int((outs2[False]<0).sum())})')

print('\n' + '=' * 74)
print('D  MULTI-STEP: 40 steps (10 days) pure advection on the real number field')
print('=' * 74)
NSTEP = int(os.environ.get('NSTEP', '40'))
traj = {}
for vpos in (False, True):
    m = load(vpos)
    q = num.copy(); N0 = burden(q); fl_cum = 0.0; negmax = 0.0
    for i in range(NSTEP):
        adv, _ = run_step(m, q, h0 + 6 * i)
        negmax = min(negmax, adv.min())
        pre = burden(adv)
        q = np.maximum(adv, 0.0)                  # the model's floor
        fl_cum += burden(q) - pre
        if not np.isfinite(q).all():
            print('    NON-FINITE at step', i); break
    traj[vpos] = (burden(q) / N0, fl_cum / N0, negmax)
    tag = 'VPOS=1' if vpos else 'VPOS=0'
    print(f'  {tag}: N/N0 after {NSTEP} steps {traj[vpos][0]:.6f}   '
          f'cumulative floor/N0 {traj[vpos][1]:+.4e}   worst negative {negmax:.3e}')
f0, f1 = traj[False][1], traj[True][1]
print(f'\n  floor reduction: {f0:.4e} -> {f1:.4e}  '
      f'= {(1 - abs(f1)/max(abs(f0),1e-300))*100:.3f}% removed')
