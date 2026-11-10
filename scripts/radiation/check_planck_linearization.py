#!/usr/bin/env python3
"""Is hr_clear(T + dT) ~ hr_clear(T) + hr_dT * dT good enough?  (docs/
RADIATION_EMULATOR_SCOPE.md §4.2). hr_dT is the cached response to a UNIFORM
+1 K on the band levels (a Jacobian row sum); the emulator applies it to the
LOCAL dT. Checked here on the day-30 dT_rad of the physics run (0.8 K max)
against the exact clear-sky solve at T + dT, and at 2 dT. Also tests the TOA
fluxes the same way, and whether reverse-mode AD through the solve gives the
exact per-level TOA Jacobian in one pass. GPU + archive, ~2 min.
"""
import os, sys, time
import numpy as np
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, 'src'))
import _paths                      # noqa: E402
import coupling as C
import radiation as R
import rad_cache_build
import jax, jax.numpy as jnp

st = np.load(f'{_REPO}/runs/bench_phys30d/coupled_state_bench_phys30d_ckpt.npz')
dT = np.asarray(st['dT_rad'])                  # (24, 192, 288) K, day 30
hour = 720
dU = C.open_var('U'); lat = dU['lat'].values; lon = dU['lon'].values
plev = dU['hyam'].values * dU['P0'].values + dU['hybm'].values * C.PS_REF
zd = np.load(f'{_REPO}/runs/chemdump/debug_nucleation/premicro_state_step120.npz')
klevs = [int(np.argmin(np.abs(plev - pp))) for pp in zd['plev']]
drv = R.RadiationDriver(C.open_var, klevs, lat, lon, C.DP_BIN * 1e-9, mmid_kg=C.MMID, rho_dry=C.RHO_AER)
T_band = drv._read3d(drv.dT, 'T', hour)[drv.band_pos]
print(f'dT_rad day 30: max {dT.max():.3f} min {dT.min():.3f} K, |dT| p50 {np.median(np.abs(dT)):.4f}; '
      f'zonal-mean max {np.abs(dT.mean(-1)).max():.3f} K', flush=True)

t0 = time.time()
cl = rad_cache_build.clear_fields(drv, hour)
print(f'clear fields {time.time()-t0:.1f}s', flush=True)
S = cl['hr_dT'].astype(np.float64); hc = cl['hr_clear'].astype(np.float64)
for fac in (1.0, 2.0):
    hr_ex, d_ex = drv.heating(hour, None, T_band_override=T_band + fac * dT)
    hr_ex = np.asarray(hr_ex)
    exact = (hr_ex - hc) * 86400                        # K/day, the Planck term
    lin = S * (fac * dT) * 86400
    err = lin - exact
    # the anomaly the model integrates is ~0.1-1 K/day in the plume; report absolute errors
    print(f'[x{fac:.0f} dT] Planck term: |exact| max {np.abs(exact).max():.4f} K/d p99 {np.percentile(np.abs(exact),99):.4f}; '
          f'lin error max {np.abs(err).max():.5f} K/d p99 {np.percentile(np.abs(err),99):.5f} rms {np.sqrt((err**2).mean()):.6f}; '
          f'rel rms {np.sqrt((err**2).mean())/np.sqrt((exact**2).mean()):.3%}', flush=True)
    for k in (0, 5, 10, 15, 20, 23):
        print(f'      level {k} ({plev[klevs[k]]/100:6.1f} hPa): exact max {np.abs(exact[k]).max():.4f} err max {np.abs(err[k]).max():.5f} K/d', flush=True)
    dTm = dT.mean(0) * fac
    for n in ('olr', 'sw_up_toa'):
        ex = np.asarray(d_ex[n]) - cl[f'dc_{n}']
        li = cl[f'ddT_{n}'] * dTm
        print(f'      TOA {n}: exact max {np.abs(ex).max():.4f} W/m2, global-mean exact {ex.mean():.5f}; '
              f'lin(band-mean dT) error max {np.abs(li-ex).max():.4f} mean {np.mean(li-ex):.5f}', flush=True)

# --- reverse-mode Jacobian of OLR wrt T on one chunk -------------------------
print('AD test: vjp of OLR wrt the band-level T through the solve ...', flush=True)
try:
    nx = 48; a, b = 72, 72 + nx          # a tropical chunk
    P = rad_cache_build  # noqa
    import rad_features as F
    h = F.prep_hour(drv, hour)
    T = h['T'].copy()
    Tx = drv._to_xyz(T[:, a:b]); qx = drv._to_xyz(np.maximum(h['q'][:, a:b], 1e-12), floor=1e-12)
    o3x = drv._to_xyz(np.maximum(h['o3'][:, a:b], 0.0), floor=0.0)
    ch4x = drv._to_xyz(np.maximum(h['ch4'][:, a:b], 0.0), floor=0.0)
    tsj = jnp.asarray(h['ts'][a:b]); albj = jnp.asarray(h['alb'][a:b]); zenj = jnp.asarray(h['zen'][a:b])[:, :, None]
    zb = jnp.zeros((drv.nb_sw, nx, drv.nlon, drv.nzc + 2)); zl = jnp.zeros((drv.nb_lw, nx, drv.nlon, drv.nzc + 2))
    def olr_of_T(Txx):
        hr, dg = drv._hr_impl(Txx, qx, o3x, ch4x, tsj, albj, zenj, (zb, zb, zb), (zl, zl, zl))
        return dg[2].sum()          # sum over columns: columns are independent
    t1 = time.time()
    g = jax.jit(jax.grad(olr_of_T))(jnp.asarray(Tx)); jax.block_until_ready(g); t1 = time.time() - t1
    g = np.asarray(g)                               # (nx, nlon, nzc+2) d olr / d T  [W/m2/K]
    # map band levels (CESM order) -> xyz index: xyz = 1 + (nzc-1-krad_pos)
    pos = np.array([1 + (drv.nzc - 1 - p) for p in drv.band_pos])
    J = g[:, :, pos]                                # (nx, nlon, nzb)
    lin_J = np.einsum('xyk,kxy->xy', J, dT[:, a:b])
    hr_ex, d_ex = drv.heating(hour, None, T_band_override=T_band + dT)
    ex = (np.asarray(d_ex['olr']) - cl['dc_olr'])[a:b]
    print(f'  vjp OK in {t1:.1f}s (incl. jit); J row-sum vs FD ddT_olr: max |diff| {np.abs(J.sum(-1) - cl["ddT_olr"][a:b]).max():.4f} W/m2/K; '
          f'Jacobian-linearised OLR error max {np.abs(lin_J-ex).max():.5f} mean {np.mean(lin_J-ex):.6f} W/m2 '
          f'(exact max {np.abs(ex).max():.4f})', flush=True)
    print(f'  J per level (chunk mean, W/m2/K): ' + ' '.join(f'{v:.3f}' for v in J.mean((0, 1))), flush=True)
    t2 = time.time(); g2 = jax.jit(jax.grad(olr_of_T))(jnp.asarray(Tx)); jax.block_until_ready(g2); print(f'  vjp warm {time.time()-t2:.2f}s for {nx} rows', flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
    print(f'  AD through the solve FAILED: {e}', flush=True)
print('PLANCK_CHECK_DONE', flush=True)
