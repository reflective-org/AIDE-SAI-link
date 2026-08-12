#!/usr/bin/env python3
"""Where does the NUMBER FLOOR come from, and is it fatal for multi-year runs?

Reproduces ONE exact advection step (same scheme, winds, grid, cfl, dt as
coupling.py) on a real saved 3-D state, captures the field BEFORE the floor, and
answers:

  Q1  Which BINS go negative, and how much number does the floor add to each?
  Q2  Is it a LOCAL DIPOLE (undershoot paired with a neighbouring overshoot ->
      redistribution, self-limiting) or a NET SOURCE?
  Q3  WHERE (level / latitude) does it happen -> is it the injection ring, the
      domain edges, or everywhere?
  Q4  Which SWEEP produces it -- horizontal (Zalesak-limited) or the vertical
      remap (unlimited)?
  Q5  What does the per-bin floor rate imply for a MULTI-YEAR run?

Read-only w.r.t. the production run. Runs on its own GPU.
"""
import os, sys, numpy as np
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))


sys.path.insert(0, _REPO)
sys.path.insert(0, _os.path.join(_REPO, 'fast_advection'))

import jax, jax.numpy as jnp
import xarray as xr

# Checkpoint from the CURRENT DIRECTORY, like every other script here. This used
# to default to a path inside the repo, which stopped resolving once run outputs
# moved to a runs directory outside the tree (see README) -- run this from there.
STATE = sys.argv[1] if len(sys.argv) > 1 else \
    _os.environ.get('STATE', 'coupled_state_prod90d_ckpt.npz')
if not _os.path.exists(STATE):
    raise SystemExit(
        f"floor_anatomy: no checkpoint at {STATE!r}.\n"
        f"  Run this from the directory holding your run output, pass one as\n"
        f"  argv[1], or set STATE=/path/to/coupled_state_<TAG>_ckpt.npz")
HOUR = int(sys.argv[2]) if len(sys.argv) > 2 else None

# ---- grid, exactly as coupling.py builds it -------------------------------
HDIR = ('/data/cesm2.1.5_output/histSST/'
        'f.e21.FWHIST.f09_f09_mg17.atmos-scale_fixedSST_1996-2014.001/'
        'archive/atm/proc/tseries')
PREFIX = 'f.e21.FWHIST.f09_f09_mg17.atmos-scale_fixedSST_1996-2014.001.cam'
H1 = f'{HDIR}/hour_1/{PREFIX}.h1'
SUF = '.1996010100-2014123100.nc'
PS_REF = 1.0e5
P_LO_HPA = float(os.environ.get('P_LO_HPA', '1.0'))
P_HI_HPA = float(os.environ.get('P_HI_HPA', '150.0'))
LAT_FREEZE = 80.0
STEP_SEC = 6 * 3600.0

dU = xr.open_dataset(f'{H1}.U{SUF}', decode_times=False)
dV = xr.open_dataset(f'{H1}.V{SUF}', decode_times=False)
dW = xr.open_dataset(f'{H1}.OMEGA{SUF}', decode_times=False)
lat = dU['lat'].values
plev = dU['hyam'].values * dU['P0'].values + dU['hybm'].values * PS_REF
band = np.where((plev >= P_LO_HPA * 100) & (plev <= P_HI_HPA * 100))[0]
klevs = list(range(band.min(), band.max() + 1))
PLEV_PA = plev[klevs]
DP = np.gradient(PLEV_PA)
nlev = len(klevs)

st = np.load(STATE)
num = np.asarray(st['num'])                      # (NBINS, nlev, nlat, nlon)
NBINS, _, nlat, nlon = num.shape
h0 = int(st['s_done']) * int(st['step_hours']) if HOUR is None else HOUR
print(f'state   : {os.path.basename(STATE)}  s_done={int(st["s_done"])} -> hour {h0}')
print(f'grid    : {NBINS} bins x {nlev} lev x {nlat} lat x {nlon} lon '
      f'({PLEV_PA[0]/100:.1f}..{PLEV_PA[-1]/100:.1f} hPa)')
print(f'negatives already in the SAVED state (post-floor, must be 0): '
      f'{(num < 0).sum()}')

# bin diameters, to name the sizes
from coupling import DP_BIN                      # nm, bin mid diameters
DPB = np.asarray(DP_BIN)[:NBINS]

def winds(t):
    u = dU['U'].isel(time=t, lev=klevs).values
    v = dV['V'].isel(time=t, lev=klevs).values
    w = dW['OMEGA'].isel(time=t, lev=klevs).values
    return u, v, w

u0, v0, w0 = winds(h0)
u1, v1, w1 = winds(h0 + 6)

# ---- area / dp weight, matching coupling.py's burden metric ---------------
DEG = np.pi / 180.0
W_LAT = np.cos(lat * DEG)
A = DP[:, None] * W_LAT[None, :]                 # (nlev, nlat) burden weight
A3 = A[None, :, :, None]                         # broadcast over bin, lon


def burden(x):
    """dp*cos(lat)-weighted burden, same metric as coupling.py's Nbur."""
    return float((x * A3).sum())


def burden_per_bin(x):
    return (x * A3).sum(axis=(1, 2, 3))


def burden_per_lev(x):
    return (x * A3).sum(axis=(0, 2, 3))


import fct_lr

# =========================================================================
# Q4 first: isolate the VERTICAL remap from the horizontal sweeps.
# =========================================================================
print('\n' + '=' * 72)
print('Q4  WHICH SWEEP CREATES THE NEGATIVES?')
print('=' * 72)

# --- vertical remap alone, on the real number field ---
w_face_fn = fct_lr._omega_continuity
_mf, _ac = fct_lr.grid_metric(lat, LAT_FREEZE)
mf = jnp.asarray(_mf); ac = jnp.asarray(_ac)
polar = jnp.asarray(np.abs(lat) > LAT_FREEZE)
dlam = 2.0 * np.pi / nlon
dphi = float(lat[1] - lat[0]) * DEG
dp_j = jnp.asarray(DP)

# substep count exactly as advect_hour_batch computes it
keep = np.abs(lat) <= LAT_FREEZE
umax = max(float(np.abs(u0[:, keep]).max()), float(np.abs(u1[:, keep]).max()))
vmax = max(float(np.abs(v0).max()), float(np.abs(v1).max()))
dx_min = float((fct_lr.RAD * np.cos(lat[keep] * DEG) * dlam).min())
cfl = 0.2
dt_x = cfl * dx_min / max(umax, 1e-6)
dt_y = cfl * (fct_lr.RAD * dphi) / max(vmax, 1e-6)
dt_sub = min(dt_x, dt_y)
nsub = int(np.ceil(STEP_SEC / dt_sub)); dt_sub = STEP_SEC / nsub
print(f'  substeps for this 6h step: nsub={nsub}  dt_sub={dt_sub:.1f}s')

w_face = w_face_fn(jnp.asarray(u0), jnp.asarray(v0), jnp.asarray(w0),
                   jnp.asarray(lat), dp_j, polar, mf, ac, dlam, dphi)
w_face = np.asarray(w_face)

# ONE vertical remap substep on every bin, rho=1 (isolates the remap operator)
rho1 = jnp.ones((nlev, nlat * nlon))
neg_vert = np.zeros(NBINS)
for k in range(NBINS):
    q = jnp.asarray(num[k].reshape(nlev, -1))
    _, rq, _, _ = fct_lr._lr_vert(rho1.T, q.T, dp_j, dt_sub,
                                  jnp.asarray(w_face.reshape(nlev + 1, -1)).T,
                                  bc_top=q.T[:, 0], bc_bot=q.T[:, -1])
    out = np.asarray(rq).T.reshape(nlev, nlat, nlon)
    neg_vert[k] = -(np.minimum(out, 0.0) * A[:, :, None]).sum()
print(f'  VERTICAL remap, one substep: floor-add = {neg_vert.sum():.4e} '
      f'(bins with negatives: {(neg_vert > 0).sum()}/{NBINS})')

# ONE horizontal x+y sweep pair, same substep
neg_horz = np.zeros(NBINS)
for k in range(NBINS):
    rho = jnp.ones((nlev, nlat, nlon))
    rhoq = jnp.asarray(num[k])
    uf = 0.5 * (jnp.asarray(u0) + jnp.roll(jnp.asarray(u0), -1, 2))
    cx = uf * dt_sub / (fct_lr.RAD * jnp.cos(jnp.asarray(lat) * DEG) * dlam)[None, :, None]
    cx = jnp.where(polar[None, :, None], 0.0, cx)
    sh = lambda X: X.reshape(nlev * nlat, nlon)
    r2, t2 = fct_lr._lr_sweep(sh(rho), sh(rhoq), sh(cx), periodic=True)
    rho = r2.reshape(nlev, nlat, nlon); rhoq = t2.reshape(nlev, nlat, nlon)
    vf = 0.5 * (jnp.asarray(v0) + jnp.roll(jnp.asarray(v0), -1, 1))
    cy = vf * dt_sub / (fct_lr.RAD * dphi)
    tr = lambda X: X.transpose(0, 2, 1).reshape(nlev * nlon, nlat)
    r2, t2 = fct_lr._lr_sweep(tr(rho), tr(rhoq), tr(cy), periodic=False, mf=mf, ac=ac)
    back = lambda X: X.reshape(nlev, nlon, nlat).transpose(0, 2, 1)
    out = np.asarray(back(t2))
    neg_horz[k] = -(np.minimum(out, 0.0) * A[:, :, None]).sum()
print(f'  HORIZONTAL x+y sweeps, one substep: floor-add = {neg_horz.sum():.4e} '
      f'(bins with negatives: {(neg_horz > 0).sum()}/{NBINS})')
tot = neg_vert.sum() + neg_horz.sum()
if tot > 0:
    print(f'  => vertical accounts for {100*neg_vert.sum()/tot:.1f}% of the '
          f'negative number created in one substep')

# =========================================================================
# Q1-Q3: the FULL step, as the model actually runs it
# =========================================================================
print('\n' + '=' * 72)
print('Q1-Q3  FULL 6h STEP: per-bin, per-level anatomy of the floor')
print('=' * 72)
qb = jnp.asarray(num)
qfroz = jnp.asarray(num)
out = fct_lr.advect_hour_batch(qb, jnp.asarray(u0), jnp.asarray(v0), jnp.asarray(w0),
                               jnp.asarray(u1), jnp.asarray(v1), jnp.asarray(w1),
                               lat=lat, dp=DP, qfrozb=qfroz,
                               lat_freeze=LAT_FREEZE, dt_total=STEP_SEC,
                               return_vflux=True)
adv = np.asarray(out[0])
print(f'  advect_hour_batch returned, nsub={out[1]}')

N_pre = burden(adv)
N_post = burden(np.maximum(adv, 0.0))
floor_add = N_post - N_pre
N_in = burden(num)
print(f'  N burden  in {N_in:.6e}   post-advect(pre-floor) {N_pre:.6e}   '
      f'post-floor {N_post:.6e}')
print(f'  floor adds {floor_add:.4e}  = {100*floor_add/N_in:.3f}% of the '
      f'standing burden, in ONE 6h step')

neg = np.minimum(adv, 0.0)
pos_mask = adv > 0
fpb = -(neg * A3).sum(axis=(1, 2, 3))            # floor add per bin
npb_in = burden_per_bin(num)                      # standing burden per bin
print('\n  --- Q1: per-bin floor (only bins that get any) ---')
print('   bin   Dp(nm)   floor_add      standing N     floor/standing   %of total floor')
order = np.argsort(-fpb)
for k in order:
    if fpb[k] <= 0:
        continue
    frac = fpb[k] / max(npb_in[k], 1e-300)
    print(f'  {k:4d} {DPB[k]:8.1f}   {fpb[k]:.4e}   {npb_in[k]:.4e}   '
          f'{frac:12.3e}   {100*fpb[k]/max(floor_add,1e-300):7.2f}%')

print('\n  --- Q3: per-level floor ---')
fpl = -(neg * A3).sum(axis=(0, 2, 3))
nplev = burden_per_lev(num)
print('   lev    hPa     floor_add      standing N     floor/standing')
for z in range(nlev):
    if fpl[z] <= 0:
        continue
    print(f'  {z:4d} {PLEV_PA[z]/100:7.1f}   {fpl[z]:.4e}   {nplev[z]:.4e}   '
          f'{fpl[z]/max(nplev[z],1e-300):12.3e}')

print('\n  --- Q3b: floor by latitude band ---')
fplat = -(neg * A3).sum(axis=(0, 1, 3))
for lo, hi, name in [(-90, -60, 'SH polar'), (-60, -30, 'SH mid'),
                     (-30, -10, 'SH subtrop'), (-10, 10, 'TROPICS (inj ring)'),
                     (10, 30, 'NH subtrop'), (30, 60, 'NH mid'), (60, 90, 'NH polar')]:
    m = (lat >= lo) & (lat < hi)
    print(f'  {name:20s} {fplat[m].sum():.4e}   '
          f'{100*fplat[m].sum()/max(floor_add,1e-300):6.2f}% of floor')

# =========================================================================
# Q2: dipole or net source?
# =========================================================================
print('\n' + '=' * 72)
print('Q2  DIPOLE OR NET SOURCE?')
print('=' * 72)
# A conservative undershoot pairs -d in one cell with +d in a neighbour. Test by
# comparing the floor add against the CONSERVATION ERROR of the same step: LR is
# exactly conservative, so if the negatives were pure redistribution the total
# would be unchanged and the floor would be the ONLY thing breaking conservation.
cons_err = (N_pre - N_in) / N_in
print(f'  conservation error of the raw step (pre-floor): {cons_err:+.3e}')
print(f'    (LR is exactly conservative apart from the open vertical faces, so a')
print(f'     small value here means the negatives are REDISTRIBUTION, i.e. every')
print(f'     -d has a matching +d somewhere. The floor then keeps the +d and')
print(f'     DELETES the -d, which is what makes it a one-sided SOURCE.)')
print(f'  floor add / |total| = {floor_add/N_in:+.3e}')

# vertical-neighbour test: is the cell above/below a negative cell anomalously high?
zneg, yneg, xneg = np.where(adv[0] < 0) if (adv[0] < 0).any() else (np.array([]),)*3
kbig = int(np.argmax(fpb))
sl = adv[kbig]
mneg = sl < 0
print(f'\n  bin {kbig} (Dp={DPB[kbig]:.1f} nm) carries '
      f'{100*fpb[kbig]/max(floor_add,1e-300):.1f}% of the floor; '
      f'{mneg.sum()} of {mneg.size} cells negative ({100*mneg.mean():.3f}%)')
if mneg.any():
    zz = np.where(mneg.any(axis=(1, 2)))[0]
    print(f'    negative cells occur at levels: '
          f'{[f"{PLEV_PA[z]/100:.1f}hPa" for z in zz]}')
    # magnitude of the negative vs the local positive field
    print(f'    most negative value {sl.min():.3e} vs field max {sl.max():.3e} '
          f'-> |neg|/max = {abs(sl.min())/max(sl.max(),1e-300):.3e}')

# =========================================================================
# Q5: multi-year implication
# =========================================================================
print('\n' + '=' * 72)
print('Q5  MULTI-YEAR IMPLICATION')
print('=' * 72)
per_step = floor_add / N_in
print(f'  floor adds {per_step:.4e} of the standing burden per 6h step')
for yrs in (0.25, 1, 5, 10):
    steps = yrs * 365 * 4
    print(f'   {yrs:5.2f} yr = {steps:6.0f} steps -> cumulative floor = '
          f'{per_step*steps:8.2f} x the CURRENT standing N '
          f'(if the rate and N both held constant)')
print('\n  NB this is a SOURCE rate, not a standing contamination. The standing')
print('  contamination is rate x (number lifetime of the created particles).')
print('  A cumulative figure >1 does NOT mean >100% of particles are spurious --')
print('  it means the floor injects more than one standing population over that')
print('  time, which coagulation then has to remove.')
