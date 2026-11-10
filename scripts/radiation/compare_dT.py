#!/usr/bin/env python3
"""Gate 1 of docs/RADIATION_EMULATOR_SCOPE.md §5.3: the prognostic radiative
temperature increment dT_rad at the end of a run, emulator vs physics.

    python3 scripts/radiation/compare_dT.py REF_TAG TAG [TAG ...]

Reads runs/<TAG>/coupled_final_<TAG>.npz (falls back to the _ckpt state) and
prints, per TAG: point-wise |ddT| max / p99 / rms over the band and at the
probe level, the zonal-mean |ddT| max, and the same numbers relative to the
physics dT itself. Also the TOA forcing and AOD of the timeseries at day 10-30.
"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def state(tag):
    for f in (f'{ROOT}/runs/{tag}/coupled_final_{tag}.npz', f'{ROOT}/runs/{tag}/coupled_state_{tag}_ckpt.npz'):
        if os.path.exists(f):
            z = np.load(f); return z, f
    sys.exit(f'no state for {tag}')
ref, fr = state(sys.argv[1]); dT0 = np.asarray(ref['dT_rad'])
plev = np.asarray(ref['plev_pa']) / 100 if 'plev_pa' in ref.files else None
kp = int(np.argmin(np.abs(plev - 51.7))) if plev is not None else 17
print(f'reference {sys.argv[1]}: dT_rad max {dT0.max():.3f} min {dT0.min():.3f} K; zonal-mean max {np.abs(dT0.mean(-1)).max():.3f}; probe level {kp} ({plev[kp] if plev is not None else "?"} hPa) max {np.abs(dT0[kp]).max():.3f}')
ts0 = np.load(f'{ROOT}/runs/{sys.argv[1]}/coupled_timeseries_{sys.argv[1]}.npz') if os.path.exists(f'{ROOT}/runs/{sys.argv[1]}/coupled_timeseries_{sys.argv[1]}.npz') else None
for tag in sys.argv[2:]:
    z, f = state(tag); dT = np.asarray(z['dT_rad']); d = dT - dT0
    zm = d.mean(-1)
    print(f'{tag} ({os.path.basename(f)}):')
    print(f'   point-wise |ddT|: max {np.abs(d).max():.4f} K, p99 {np.percentile(np.abs(d), 99):.4f}, rms {np.sqrt((d**2).mean()):.5f}  '
          f'(gate 0.05 K); rel to |dT| rms {np.sqrt((d**2).mean())/np.sqrt((dT0**2).mean()):.1%}')
    print(f'   probe level {kp}: max {np.abs(d[kp]).max():.4f} K, rms {np.sqrt((d[kp]**2).mean()):.5f}; zonal-mean |ddT| max {np.abs(zm).max():.4f} K (gate 0.02 K), at probe {np.abs(zm[kp]).max():.4f}')
    for k in range(0, d.shape[0], 4):
        print(f'      level {k:2d} ({plev[k]:6.1f} hPa): |ddT| max {np.abs(d[k]).max():.4f} rms {np.sqrt((d[k]**2).mean()):.5f}; |dT| max {np.abs(dT0[k]).max():.3f}')
    tsf = f'{ROOT}/runs/{tag}/coupled_timeseries_{tag}.npz'
    if ts0 is not None and os.path.exists(tsf):
        ts = np.load(tsf); n = min(len(ts['hours']), len(ts0['hours']))
        h = np.asarray(ts0['hours'][:n]); m = (h >= 240) & (h <= 720)
        a0 = np.asarray(ts0['arf_toa_avg'][:n]); a1 = np.asarray(ts['arf_toa_avg'][:n])
        print(f'   arf_toa_avg day 10-30: physics {np.nanmean(a0[m]):+.4f} emulator {np.nanmean(a1[m]):+.4f} W/m2 ({(np.nanmean(a1[m])/np.nanmean(a0[m])-1):+.1%}, gate 2 %); '
              f'aod550 day 30: {ts0["aod550"][n-1]:.5f} vs {ts["aod550"][n-1]:.5f} ({ts["aod550"][n-1]/ts0["aod550"][n-1]-1:+.2%}); '
              f'Nburden {ts["Nburden"][n-1]/ts0["Nburden"][n-1]-1:+.2%} Mburden {ts["Mburden"][n-1]/ts0["Mburden"][n-1]-1:+.2%}')
