"""v2 aggregation: emulator + numerical + MAM4 per-day caches, plus the diurnal SW re-solves (cache/diurnal)."""
import numpy as np, glob, os
from common import *
D = {}
for pat in ('day_[0-9][0-9][0-9].npz', 'day_mam_[0-9][0-9][0-9].npz'):
    files = sorted(glob.glob(f'{CACHE}/{pat}')); assert len(files) == NDAYS, (pat, len(files))
    z0 = np.load(files[0])
    keys = [k for k in z0.files if k != 'day' and k != 'tband_wt' and (pat.startswith('day_[') or k.startswith('mam_'))]
    for k in keys:
        D[k] = np.stack([np.load(f)[k] for f in files]) if False else None
    acc = {k: [] for k in keys}
    for f in files:
        z = np.load(f)
        for k in keys:
            acc[k].append(z[k])
    D.update({k: np.stack(v) for k, v in acc.items()})
lat = np.load(f'{RUNS}/r2_tomas_fast/frames3d/step_0000.npz')['lat']
w = np.cos(np.deg2rad(lat)); w2 = w[:, None] / (w.sum() * 288)
gm = lambda x: (x * w2).sum((-2, -1))
K = ('emu', 'num', 'mam')
S = D['swdn'].mean(2)
for key in K:
    dA = D[f'{key}_dswup_lit'] / np.where(D[f'{key}_swdn_lit'] > 0, D[f'{key}_swdn_lit'], np.nan)
    D[f'{key}_Fsw_est'] = np.nan_to_num(-dA * S[:, :, None]).astype(np.float32)
    D[f'{key}_Fsw_raw'] = D[f'{key}_ctl_toa_sw_up'] - D[f'{key}_inj_toa_sw_up']
    D[f'{key}_Flw'] = D[f'{key}_ctl_toa_olr'] - D[f'{key}_inj_toa_olr']
    D[f'{key}_Fnet_raw'] = D[f'{key}_Fsw_raw'] + D[f'{key}_Flw']
    D[f'{key}_dM'] = D[f'{key}_inj_colM'] - D[f'{key}_ctl_colM']
    D[f'{key}_dN'] = D[f'{key}_inj_colN'] - D[f'{key}_ctl_colN']
    D[f'{key}_dAOD'] = D[f'{key}_inj_aod550'] - D[f'{key}_ctl_aod550']
ts = {}
for key in K:
    for v in ('Fsw_est', 'Fsw_raw', 'Flw', 'Fnet_raw', 'dM', 'dN', 'dAOD', 'inj_colM', 'ctl_colM', 'inj_colN', 'ctl_colN',
              'dg', 'sig', 'deff_dry', 'deff_wet', 'dmass'):
        ts[f'{key}_{v}'] = gm(D[f'{key}_{v}'])
ts['strat_frac'] = D['strat_frac_cells'].mean((1, 2)); ts['nofound'] = D['nofound'].mean((1, 2))
# diurnal re-solves
dd = sorted(glob.glob(f'{CACHE}/diurnal/*_[0-9][0-9][0-9].npz'))
days = sorted({int(os.path.basename(f)[-7:-4]) for f in dd})
days = [d for d in days if all(os.path.exists(f'{CACHE}/diurnal/{k}_{d:03d}.npz') for k in K)]
DI = {}
rep = {}
for key in K:
    zs = [np.load(f'{CACHE}/diurnal/{key}_{d:03d}.npz') for d in days]
    DI[f'{key}_Fsw16'] = np.stack([z['ctl_swup16'] - z['inj_swup16'] for z in zs])
    DI[f'{key}_Fsw8'] = np.stack([z['ctl_swup8'] - z['inj_swup8'] for z in zs])
    DI[f'{key}_Flw_d'] = D[f'{key}_Flw'][days]
    DI[f'{key}_Fnet16'] = DI[f'{key}_Fsw16'] + DI[f'{key}_Flw_d']
    DI[f'{key}_Fsw_raw_d'] = D[f'{key}_Fsw_raw'][days]; DI[f'{key}_Fsw_est_d'] = D[f'{key}_Fsw_est'][days]
    DI[f'{key}_Fnet_raw_d'] = D[f'{key}_Fnet_raw'][days]
    for r in ('inj', 'ctl'):
        rep[f'{key}_{r}_repro_maxabs'] = max(float(z[f'{r}_repro_maxabs']) for z in zs)
        rep[f'{key}_{r}_repro_bitfrac_min'] = min(float(z[f'{r}_repro_bitfrac']) for z in zs)
        rep[f'{key}_{r}_repro_gmdiff_maxabs'] = max(abs(float(z[f'{r}_repro_gmdiff'])) for z in zs)
    rep[f'{key}_swdn16_vs_swdn4_gm'] = float(np.mean([gm(z['inj_swdn16']) for z in zs]) - gm(D['swdn'][days].mean(0)))
days = np.array(days)
A = {}
for k, v in D.items():
    A[f'ann_{k}'] = np.nanmean(v, 0); A[f'l30_{k}'] = np.nanmean(v[-30:], 0)
l30 = days >= NDAYS - 30
for k, v in DI.items():
    A[f'dann_{k}'] = v.mean(0); A[f'dl30_{k}'] = v[l30].mean(0)
    ts[f'd_{k}'] = gm(v)
mon_edges = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
zc = D['strat_zcount']
for key in K:
    for h in ('dhr_kday', 'dhr_sw_kday'):
        zs_ = D[f'{key}_{h}_zsum']
        A[f'ann_{key}_{h}_zm'] = zs_.sum(0) / np.where(zc.sum(0) > 0, zc.sum(0), np.nan)
        _md = np.searchsorted(mon_edges, np.arange(zs_.shape[0]) % 365, side='right') - 1   # calendar month of each day (multi-year: same month pooled)
        A[f'mon_{key}_{h}_zm'] = np.stack([zs_[_md == m].sum(0) / np.where(zc[_md == m].sum(0) > 0, zc[_md == m].sum(0), np.nan)
                                           for m in range(12)])
A['lat'] = lat; A['ddays'] = days
np.savez(f'{CACHE}/agg_v2.npz', **A)
np.savez(f'{CACHE}/timeseries_v2.npz', lat=lat, ddays=days, **ts)
np.savez(f'{CACHE}/daily_gif_v2.npz', **{f'{k}_dM': D[f'{k}_dM'] for k in K})
# 2026-11-09: daily added-AOD maps + daily tropopause for the AOD animation and the 2-yr tropopause figure (plots_extra.py)
np.savez(f'{CACHE}/daily_aod_v2.npz', lat=lat, **{f'{k}_dAOD': D[f'{k}_dAOD'].astype(np.float32) for k in K},
         ptrop_zm=np.nanmean(D['ptrop'], 2).astype(np.float32), strat_frac_zm=D['strat_frac_cells'].mean(2).astype(np.float32))
import json; json.dump(rep, open(f'{CACHE}/repro.json', 'w'), indent=1)
print('days with diurnal:', len(days), rep)
