#!/usr/bin/env python3
"""ESTIMATED stratospheric Delta T from the no-feedback runs (2026-11-09, PYC).
The campaign ran RAD_MODE=flux: the temperature never responds and only the aerosol heating rate Q (injection minus
control) exists. This integrates it with Newtonian radiative damping,  dDT/dt = Q - DT / tau(p),  DT(0) = 0, daily steps,
tau(p) linear in log p from 5 days at 1 hPa to 30 days at 100 hPa (clipped outside). That is an ASSUMED damping profile
(the order of published stratospheric radiative relaxation times), not something the model computed. Sensitivity of the
global mean to a uniform tau of 10 / 20 / 40 days is printed and drawn.
Inputs: cache/day_DDD.npz (num, emu) and cache/day_mam_DDD.npz (MAM4): zonal sums dhr_kday_zsum / strat_zcount (lat-p)
and the 51.7 hPa maps dhr_kday_50 (LW per cell + zonal-mean SW, as in heating_l30_map_ts.py, to avoid the fixed-UTC
day/night arcs)."""
import json, datetime
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import cartopy.crs as ccrs

C = '/data/AIDE-SAI-link/linking_plots/coupled'
meta = json.load(open('/data/AIDE-SAI-link/runs/emu_winds_P2s200_1997_1998/met/meta.json'))
plev = np.array(meta['plev_pa']) / 100; lat = np.array(meta['lat']); lon = np.arange(288) * 1.25
ND = 730; K = ('num', 'emu', 'mam'); LAB = {'emu': 'TOMAS emulator', 'num': 'TOMAS numerical', 'mam': 'MAM4'}
COL = {'emu': '#eb6834', 'num': '#2a78d6', 'mam': '#1baf7a'}
TAU = np.clip(5 + (30 - 5) * np.log(plev / 1) / np.log(100 / 1), 5, 30)    # days, per level
K50 = 17
print('tau(p):', ', '.join(f'{p:.1f} hPa {t:.1f} d' for p, t in zip(plev[::4], TAU[::4])))

Qzm = {k: np.zeros((ND, plev.size, lat.size)) for k in K}; cnt = np.zeros((ND, plev.size, lat.size))
Q50 = {k: np.zeros((ND, lat.size, lon.size), np.float32) for k in K}
for d in range(ND):
    a = np.load(f'{C}/cache/day_{d:03d}.npz'); b = np.load(f'{C}/cache/day_mam_{d:03d}.npz')
    cnt[d] = a['strat_zcount']
    for k in K:
        z = b if k == 'mam' else a
        Qzm[k][d] = z[f'{k}_dhr_kday_zsum']
        net, sw = z[f'{k}_dhr_kday_50'], z[f'{k}_dhr_sw_kday_50']
        swz = np.nanmean(sw, 1, keepdims=True)
        Q50[k][d] = net - sw + swz                                            # NaN where 51.7 hPa is not stratospheric
for k in K:
    Qzm[k] = np.where(cnt > 0.5, Qzm[k] / np.maximum(cnt, 1e-9), np.nan)    # K/day, zonal mean over stratospheric cells


def integrate(Q, tau):
    """Q (ND, ...) K/day, tau broadcastable (days): exact daily update with Q held over the day."""
    f = np.exp(-1.0 / tau); out = np.zeros_like(Q); x = np.zeros(Q.shape[1:])
    for d in range(Q.shape[0]):
        q = np.nan_to_num(Q[d]); x = x * f + q * tau * (1 - f); out[d] = np.where(np.isfinite(Q[d]), x, np.nan)
    return out


DTzm = {k: integrate(Qzm[k], TAU[:, None]) for k in K}
DT50 = {k: integrate(Q50[k].astype(np.float64), TAU[K50]) for k in K}
dp = np.gradient(plev); wlat = np.cos(np.deg2rad(lat))
def gmean(x):                                     # stratospheric mass-weighted global mean of a (..., lev, lat) field
    W = dp[:, None] * wlat[None, :] * (cnt[-1] > 0.5)
    return np.nansum(x * W, (-2, -1)) / np.nansum(np.where(np.isfinite(x), W, 0), (-2, -1))
GM = {k: gmean(DTzm[k]) for k in K}
SENS = {t: gmean(integrate(Qzm['num'], np.full((plev.size, 1), float(t)))) for t in (10, 20, 40)}
l30 = slice(ND - 30, ND)
w2 = wlat[:, None] * np.ones((1, 288))
for k in K:
    m50 = np.nanmean(DT50[k][l30], 0)
    print(f'{k}: last-30 DT global strat mean {np.nanmean(GM[k][l30]):.3f} K; 51.7 hPa map global mean '
          f'{np.nansum(m50 * w2) / np.nansum(np.where(np.isfinite(m50), w2, 0)):.3f} K, max {np.nanmax(m50):.2f} K; '
          f'zonal-mean max {np.nanmax(np.nanmean(DTzm[k][l30], 0)):.2f} K')
for t, s in SENS.items():
    print(f'num, uniform tau {t} d: last-30 DT global strat mean {np.nanmean(s[l30]):.3f} K')

plt.rcParams.update({'font.size': 9.5})
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.05], hspace=0.38, wspace=0.12, left=0.05, right=0.93, top=0.92, bottom=0.07)
M = {k: np.nanmean(DT50[k][l30], 0) for k in K}; Z = {k: np.nanmean(DTzm[k][l30], 0) for k in K}
lim_m = float(np.nanpercentile(np.abs(np.concatenate([M[k].ravel() for k in K])), 99.5))
lim_z = float(np.nanpercentile(np.abs(np.concatenate([Z[k].ravel() for k in K])), 99.5))
cm = plt.get_cmap('RdBu_r').copy(); cm.set_bad('0.85')
for j, k in enumerate(K):
    ax = fig.add_subplot(gs[0, j], projection=ccrs.PlateCarree(central_longitude=180)); ax.coastlines(lw=0.4, color='0.3'); ax.set_global()
    im1 = ax.pcolormesh(lon, lat, M[k], cmap=cm, norm=mcolors.Normalize(-lim_m, lim_m), shading='auto', transform=ccrs.PlateCarree(), rasterized=True)
    ax.plot([180, 180], [29.7, -29.7], 'k+', ms=7, mew=1.2, transform=ccrs.PlateCarree(), ls='none')
    ax.set_title(f'{LAB[k]}: ΔT at 51.7 hPa, last 30 days', fontsize=9.5)
    ax = fig.add_subplot(gs[1, j])
    im2 = ax.pcolormesh(lat, plev, Z[k], cmap=cm, norm=mcolors.Normalize(-lim_z, lim_z), shading='nearest', rasterized=True)
    ax.set_yscale('log'); ax.set_ylim(plev.max(), plev.min()); ax.set_yticks([100, 50, 20, 10, 5, 2]); ax.set_yticklabels(['100', '50', '20', '10', '5', '2'])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_xticks(range(-90, 91, 30)); ax.set_xlabel('latitude')
    if j == 0: ax.set_ylabel('hPa')
    ax.set_title(f'{LAB[k]}: zonal-mean ΔT, last 30 days', fontsize=9.5)
fig.colorbar(im1, ax=fig.axes[0:6:2], orientation='vertical', fraction=0.015, pad=0.01, extend='both').set_label('ΔT, K (estimate)')
fig.colorbar(im2, ax=fig.axes[1:6:2], orientation='vertical', fraction=0.015, pad=0.01, extend='both').set_label('ΔT, K (estimate)')
a = fig.add_subplot(gs[2, :]); xd = [datetime.date(1997, 1, 1) + datetime.timedelta(days=i) for i in range(ND)]
for k in K:
    a.plot(xd, GM[k], color=COL[k], lw=2, label=f'{LAB[k]} (τ(p) 5-30 d)')
for t, ls in zip(SENS, (':', '--', '-.')):
    a.plot(xd, SENS[t], color=COL['num'], lw=1, ls=ls, alpha=0.8, label=f'TOMAS numerical, uniform τ = {t} d')
a.axvspan(xd[-30], xd[-1], color='0.88', zorder=0); a.axhline(0, color='0.6', lw=0.6); a.grid(alpha=0.25)
a.legend(frameon=False, loc='upper left', ncol=2, fontsize=8.5); a.set_xlim(xd[0], xd[-1]); a.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
a.set_ylabel('ΔT, K (estimate)')
a.set_title('Global stratospheric mass-weighted mean ΔT through 1997-1998 (shaded: the last 30 days in the maps)', loc='left', fontsize=9.5)
fig.suptitle('ESTIMATED stratospheric temperature response ΔT (injection minus control): the aerosol heating rate integrated with assumed '
             'radiative damping\ndΔT/dt = Q − ΔT/τ(p), τ from 5 d at 1 hPa to 30 d at 100 hPa. The runs themselves have no temperature feedback; '
             'no dynamical response is included.', fontsize=10.5)
fig.text(0.5, 0.012, 'Maps at 51.7 hPa: longwave heating per cell + zonal-mean shortwave (the fixed-UTC frames otherwise draw day/night arcs); '
         'grey = 51.7 hPa below the local tropopause.', ha='center', fontsize=8.5, color='0.35')
out = f'{C}/extra/dT_estimate_l30.png'; fig.savefig(out, dpi=140); print('wrote', out)
np.savez(f'{C}/extra/dT_estimate.npz', tau=TAU, plev=plev, lat=lat, **{f'{k}_gm': GM[k] for k in K},
         **{f'{k}_zm_l30': Z[k] for k in K}, **{f'{k}_50_l30': M[k] for k in K}, **{f'num_gm_tau{t}': s for t, s in SENS.items()})
