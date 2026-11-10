"""Stack the 365 per-day caches into cache/daily_*.npz (time series / GIF inputs) and cache/agg.npz (annual, last-30, monthly)."""
import numpy as np, glob
from common import *
files = sorted(glob.glob(f'{CACHE}/day_*.npz'))
assert len(files) == 365, len(files)
z0 = np.load(files[0]); lat = np.load(f'{RUNS}/r2_tomas_fast/frames3d/step_0000.npz')['lat']
keys = [k for k in z0.files if k != 'day']
D = {k: [] for k in keys}
for f in files:
    z = np.load(f)
    for k in keys:
        D[k].append(z[k])
D = {k: np.stack(v) for k, v in D.items()}
w = np.cos(np.deg2rad(lat)); w2 = w[:, None] / (w.sum() * 288)
gm = lambda x: (x * w2).sum((-2, -1))
for key in ('emu', 'num'):
    # SW fixed (daily-mean insolation x insolation-weighted albedo change)
    S = D['swdn'].mean(2)                                         # (365,192) zonal-mean daily insolation, exact for 4 UTC frames x all lons
    dA = D[f'{key}_dswup_lit'] / np.where(D[f'{key}_swdn_lit'] > 0, D[f'{key}_swdn_lit'], np.nan)
    D[f'{key}_Fsw_fix'] = np.nan_to_num(-dA * S[:, :, None]).astype(np.float32)
    D[f'{key}_Fsw_raw'] = D[f'{key}_ctl_toa_sw_up'] - D[f'{key}_inj_toa_sw_up']
    D[f'{key}_Flw'] = D[f'{key}_ctl_toa_olr'] - D[f'{key}_inj_toa_olr']
    D[f'{key}_Fnet_fix'] = D[f'{key}_Fsw_fix'] + D[f'{key}_Flw']
    D[f'{key}_Fnet_raw'] = D[f'{key}_Fsw_raw'] + D[f'{key}_Flw']
    D[f'{key}_dM'] = D[f'{key}_inj_colM'] - D[f'{key}_ctl_colM']
    D[f'{key}_dN'] = D[f'{key}_inj_colN'] - D[f'{key}_ctl_colN']
    D[f'{key}_dAOD'] = D[f'{key}_inj_aod550'] - D[f'{key}_ctl_aod550']
ts = {}
for key in ('emu', 'num'):
    for v in ('Fsw_fix', 'Fsw_raw', 'Flw', 'Fnet_fix', 'Fnet_raw', 'dM', 'dN', 'dAOD'):
        ts[f'{key}_{v}'] = gm(D[f'{key}_{v}'])
    for v in ('inj_colM', 'ctl_colM', 'inj_colM_all', 'ctl_colM_all', 'inj_colN', 'ctl_colN', 'inj_colSO2', 'ctl_colSO2'):
        ts[f'{key}_{v}'] = gm(D[f'{key}_{v}'])
    for v in ('dg', 'sig', 'deff_dry', 'deff_wet', 'dmass'):
        ts[f'{key}_{v}'] = gm(D[f'{key}_{v}'])
ts['strat_frac'] = D['strat_frac_cells'].mean((1, 2)); ts['strat_frac_area'] = gm(D['strat_frac_cells'])
ts['nofound'] = D['nofound'].mean((1, 2))
np.savez(f'{CACHE}/timeseries.npz', lat=lat, **ts)
np.savez(f'{CACHE}/daily_gif.npz', emu_dM=D['emu_dM'], num_dM=D['num_dM'])
mon_edges = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
A = {}
skip3 = ('tband_wt',)
for k, v in D.items():
    if k in skip3:
        continue
    A[f'ann_{k}'] = np.nanmean(v, 0); A[f'l30_{k}'] = np.nanmean(v[-30:], 0)
for key in ('emu', 'num'):
    for h in ('dhr_kday', 'dhr_sw_kday'):
        zs = D[f'{key}_{h}_zsum']; zc = D['strat_zcount']
        A[f'ann_{key}_{h}_zm'] = zs.sum(0) / np.where(zc.sum(0) > 0, zc.sum(0), np.nan)
        A[f'mon_{key}_{h}_zm'] = np.stack([zs[a:b].sum(0) / np.where(zc[a:b].sum(0) > 0, zc[a:b].sum(0), np.nan)
                                           for a, b in zip(mon_edges[:-1], mon_edges[1:])])
A['zcount_ann'] = D['strat_zcount'].mean(0)
A['lat'] = lat
np.savez(f'{CACHE}/agg.npz', **A)
print('done', len(A))
