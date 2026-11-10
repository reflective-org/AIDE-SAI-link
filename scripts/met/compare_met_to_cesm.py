#!/usr/bin/env python3
"""Sanity statistics of a MET_SRC=file directory against the CESM archive on the band.

Not a skill score (the emulator's weather is its own); a check that the
regrid and the emulator are sane where it matters for transport: the
zonal-mean zonal wind, the temperature, the vertical velocity and the wind
extremes, per band level, over the file's span, versus the archive at the
same hours. Writes <out>.txt (tables) and <out>.png.

  python3 scripts/met/compare_met_to_cesm.py --met runs/<tag>/met --out runs/<tag>/met_vs_cesm [--every 4]
"""
import argparse
import json
import os
import sys

import numpy as np
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from emu_to_met import CESM_DIR, CESM_PREFIX, CESM_SUF  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--met', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--every', type=int, default=4, help='use every Nth frame (4 = daily)')
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.met, 'meta.json')))
    klevs = meta['klevs']; plev = np.array(meta['plev_pa']) / 100
    lat = np.array(meta['lat']); wl = np.cos(np.deg2rad(lat)); wl /= wl.sum()
    hour0 = meta['hour0']; step = meta['step_hours']; nf = meta['nframe']
    idx = np.arange(0, nf, a.every)
    hours = hour0 + step * idx
    met = {k: np.load(os.path.join(a.met, f'{k}.npy'), mmap_mode='r') for k in ('u', 'v', 'omega', 'T')}
    ces = {}
    for var, k in (('U', 'u'), ('V', 'v'), ('OMEGA', 'omega'), ('T', 'T')):
        ds = xr.open_dataset(f'{CESM_DIR}/hour_1/{CESM_PREFIX}.h1.{var}{CESM_SUF}')
        ces[k] = ds[var].isel(time=hours, lev=klevs).values
        ds.close()
    lines = []
    eq = np.abs(lat) < 5
    stats = {}
    for k in ('u', 'v', 'omega', 'T'):
        m = np.asarray(met[k][idx], np.float64); c = np.asarray(ces[k], np.float64)
        zm_m = m.mean(axis=3); zm_c = c.mean(axis=3)                      # (nt, nlev, nlat)
        gm_m = (zm_m * wl).sum(-1).mean(0); gm_c = (zm_c * wl).sum(-1).mean(0)   # (nlev,)
        sd_m = (zm_m.std(axis=0) * wl).sum(-1); sd_c = (zm_c.std(axis=0) * wl).sum(-1)
        eqm = zm_m[:, :, eq].mean(-1); eqc = zm_c[:, :, eq].mean(-1)       # (nt, nlev) equatorial zonal mean
        ed_m = (m - zm_m[..., None]).std(axis=(0, 3)); ed_c = (c - zm_c[..., None]).std(axis=(0, 3))
        edm = (ed_m * wl).sum(-1); edc = (ed_c * wl).sum(-1)
        mx_m = np.abs(m).max(axis=(0, 2, 3)); mx_c = np.abs(c).max(axis=(0, 2, 3))
        stats[k] = dict(gm_m=gm_m, gm_c=gm_c, sd_m=sd_m, sd_c=sd_c, eqm=eqm, eqc=eqc, edm=edm, edc=edc, mx_m=mx_m, mx_c=mx_c)
        lines.append(f'\n== {k}: time+area mean | zonal-mean time-std | eddy std | max|x|   (met / cesm) per level')
        for j in range(plev.size):
            lines.append(f'{plev[j]:8.2f} hPa  {gm_m[j]:9.3g} / {gm_c[j]:9.3g}   {sd_m[j]:8.3g} / {sd_c[j]:8.3g}   '
                         f'{edm[j]:8.3g} / {edc[j]:8.3g}   {mx_m[j]:8.3g} / {mx_c[j]:8.3g}')
    txt = '\n'.join(lines)
    with open(a.out + '.txt', 'w') as f:
        f.write(f'met {a.met} ({meta.get("source")}) vs archive, frames {idx.size} every {a.every}, hours {hours[0]}..{hours[-1]}\n' + txt)
    print(txt)
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 3, figsize=(15, 8))
        days = idx * step / 24.0
        for i, (k, lab) in enumerate((('u', 'u [m/s]'), ('T', 'T [K]'), ('omega', 'omega [Pa/s]'))):
            s = stats[k]
            ax[0, i].plot(s['gm_m'], plev, 'r-', label='met (emulator)'); ax[0, i].plot(s['gm_c'], plev, 'k--', label='CESM')
            ax[0, i].set_yscale('log'); ax[0, i].invert_yaxis(); ax[0, i].set_xlabel('global mean ' + lab); ax[0, i].set_ylabel('hPa'); ax[0, i].legend()
        for i, p in enumerate((10.0, 30.0, 50.0)):
            j = int(np.argmin(np.abs(plev - p)))
            ax[1, i].plot(days, stats['u']['eqm'][:, j], 'r-', label='met'); ax[1, i].plot(days, stats['u']['eqc'][:, j], 'k--', label='CESM')
            ax[1, i].set_title(f'equatorial zonal-mean u at {plev[j]:.1f} hPa'); ax[1, i].set_xlabel('day'); ax[1, i].legend()
        fig.suptitle(os.path.basename(os.path.dirname(a.met)) + ': met file vs CESM archive on the band')
        fig.tight_layout(); fig.savefig(a.out + '.png', dpi=110)
        print('wrote', a.out + '.png')
    except Exception as e:  # noqa: BLE001
        print('no figure:', e)


if __name__ == '__main__':
    main()
