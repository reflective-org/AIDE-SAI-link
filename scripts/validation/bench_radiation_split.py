#!/usr/bin/env python3
"""Where does one radiation call spend its time?  Builds the production
RadiationDriver exactly as coupling.main() does, feeds it the day-30 aerosol
state, and times the pieces of heating(): host prep, per-chunk aerosol optics
(Mie tables -> tau/ssa/g on 30 bands), the rrtmgp solve, the AOD."""
import os, sys, time, glob
import numpy as np
REPO = '/data/AIDE-SAI-link'
sys.path.insert(0, f'{REPO}/src')
import _paths
import coupling as C
import radiation as R
import jax, jax.numpy as jnp

dU = C.open_var('U')
lat = dU['lat'].values; lon = dU['lon'].values
plev = dU['hyam'].values * dU['P0'].values + dU['hybm'].values * C.PS_REF
zd = np.load(f'{REPO}/runs/chemdump/debug_nucleation/premicro_state_step120.npz')
# the band the production runs used: match the dump's levels to the native ones
klevs = [int(np.argmin(np.abs(plev - pp))) for pp in zd['plev']]
assert np.allclose(plev[klevs], zd['plev'], rtol=1e-6), 'band levels do not match the dump'
drv = R.RadiationDriver(C.open_var, klevs, lat, lon, C.DP_BIN * 1e-9, mmid_kg=C.MMID, rho_dry=C.RHO_AER)
print(f'nzc (native levels in the column) = {drv.nzc}, band levels = {len(klevs)}, columns = {lat.size*lon.size}, '
      f'RAD_LAT_CHUNK={R.RAD_LAT_CHUNK}, ndev={drv.ndev}, wet_optics={drv.wet_optics}', flush=True)
z = np.load(f'{REPO}/runs/chemdump/debug_nucleation/premicro_state_step120.npz')
num = jnp.asarray(z['num'])                       # (40, 24, 192, 288) #/kg
t = C.H0 + 720                                    # hour index of the day-30 state

def tm(f, n=3):
    f(); ts = []
    for _ in range(n):
        t0 = time.time(); f(); ts.append(time.time() - t0)
    return float(np.median(ts))

# 1. whole call
hr, dg = drv.heating(t, num); jax.block_until_ready(hr)
t_full = tm(lambda: jax.block_until_ready(drv.heating(t, num)[0]))
print(f'heating() full call         : {t_full:6.2f} s', flush=True)

# 2. host prep: reads + xyz transposes + wt field
def prep():
    T = drv._read3d(drv.dT, 'T', t).astype(np.float64); q = drv._read3d(drv.dQ, 'Q', t)
    o3 = drv._read3d(drv.dO3, 'O3', t); ch4 = drv._read3d(drv.dCH4, 'CH4', t)
    ts_, alb = drv.surface(t); zen = R.solar_zenith(drv.time[t], drv.lat, drv.lon)
    Tx = drv._to_xyz(T); qx = drv._to_xyz(np.maximum(q, 1e-12), floor=1e-12)
    o3x = drv._to_xyz(np.maximum(o3, 0.0), floor=0.0); ch4x = drv._to_xyz(np.maximum(ch4, 0.0), floor=0.0)
    ww = R.wt_weights(drv._wt_field(T[drv.band_pos], q[drv.band_pos])) if drv.wet_optics else None
    return T, Tx, qx, o3x, ch4x, ts_, alb, zen, ww
t_prep = tm(prep); print(f'host prep (reads, xyz, wt%) : {t_prep:6.2f} s', flush=True)
T, Tx, qx, o3x, ch4x, ts_, alb, zen, ww = prep()

# 3. aerosol optics per chunk, 4. rrtmgp per chunk, 5. aod
Cc = R.RAD_LAT_CHUNK; nchunk = drv.ndev * int(np.ceil(drv.nlat / (drv.ndev * Cc))); Cc = int(np.ceil(drv.nlat / nchunk))
k = 0; dp_d = drv._dp_band_dev[k]; msw = drv._mie_sw_dev[k]; mlw = drv._mie_lw_dev[k]
zenj = jnp.asarray(zen)[:, :, None]
chunks = [(a, min(a + Cc, drv.nlat)) for a in range(0, drv.nlat, Cc)]
def optics():
    outs = []
    for a, b in chunks:
        nc = num[:, :, a:b]; w = jnp.asarray(ww[:, :, a:b])
        s = R._aerosol_props_wet(nc, dp_d, msw['sigma_ext'], msw['ssa'], msw['g'], w)
        l = R._aerosol_props_wet(nc, dp_d, mlw['sigma_ext'], mlw['ssa'], mlw['g'], w)
        outs.append((tuple(drv._band_to_xyz(v) for v in s), tuple(drv._band_to_xyz(v) for v in l)))
    jax.block_until_ready(outs); return outs
t_opt = tm(optics); print(f'aerosol optics (Mie->bands) : {t_opt:6.2f} s  ({len(chunks)} chunks)', flush=True)
opt = optics()
def solve():
    outs = []
    for (a, b), (aer_sw, aer_lw) in zip(chunks, opt):
        hr, dg = drv._jit_hr(jnp.asarray(Tx[a:b]), jnp.asarray(qx[a:b]), jnp.asarray(o3x[a:b]), jnp.asarray(ch4x[a:b]),
                             jnp.asarray(ts_[a:b]), jnp.asarray(alb[a:b]), zenj[a:b], aer_sw, aer_lw)
        outs.append(hr)
    jax.block_until_ready(outs); return outs
t_solve = tm(solve); print(f'rrtmgp solve (SW+LW)        : {t_solve:6.2f} s', flush=True)
def aod():
    outs = [R._aod550_wet(num[:, :, a:b], dp_d, drv._sig550_dev[k], jnp.asarray(ww[:, :, a:b])) for a, b in chunks]
    jax.block_until_ready(outs); return outs
t_aod = tm(aod); print(f'AOD 550                     : {t_aod:6.2f} s', flush=True)
# 6. the same solve with zero aerosol -> is the cost aerosol-dependent?  (it should not be)
print(f'sum of parts                : {t_prep + t_opt + t_solve + t_aod:6.2f} s  vs full {t_full:.2f} s', flush=True)
hrn = np.asarray(hr); print('hr band [K/day] p50 |hr| %.3f  max |hr| %.3f' % (np.median(np.abs(hrn)) * 86400, np.abs(hrn).max() * 86400))
print('RAD_TIMING_DONE', flush=True)
