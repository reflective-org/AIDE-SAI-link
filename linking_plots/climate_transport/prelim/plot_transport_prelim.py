#!/usr/bin/env python3
import os, numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from plotlib import latp, latp_axes, fig_to_img, save_gif, date_of_day, coastlines
OUT = '/data/AIDE-SAI-link/linking_plots/climate_transport/prelim'
d = np.load(f'{OUT}/data/aoa_run.npz')
lat, lon, p = d['lat'], d['lon'], d['plev'] / 100
nyr = int(d['years_done'])
aoa = d['aoa_zm']                      # (365*nyr, nlev, nlat) years
inj_col, inj_zm, bud = d['inj_col'], d['inj_zm'], d['budget']
cells = d['inj_cells']
notes = []

# =================== AGE OF AIR ===================
fin = aoa[-365:].mean(0)
prev = aoa[-730:-365].mean(0) if nyr >= 2 else None
amax = float(np.ceil(np.percentile(fin, 99.9) * 2) / 2)
fig, ax = plt.subplots(1, 2 if prev is not None else 1, figsize=(12 if prev is not None else 6.5, 4.6), squeeze=False)
latp(ax[0, 0], lat, p, fin, 'Blues', 0, amax, 'age of air [years]', fig, contours=np.arange(0.5, 20, 0.5))
ax[0, 0].set_title(f'Age of air, annual-mean zonal mean, model year {nyr} [years]')
if prev is not None:
    dtr = fin - prev
    lim = float(np.percentile(np.abs(dtr), 99.5))
    from plotlib import DIV
    latp(ax[0, 1], lat, p, dtr, DIV, -lim, lim, 'drift [years per year]', fig, ylabel=False)
    ax[0, 1].set_title(f'Drift: year {nyr} minus year {nyr-1} annual mean [yr/yr]')
fig.suptitle(f'Clock-tracer age of air on the cycled 1997 emulator winds ({nyr} years; source 143 hPa, 10S-10N)')
fig.tight_layout(); fig.savefig(f'{OUT}/aoa_final_year_annual_mean.png', dpi=150); plt.close(fig)

# time series at reference points
pts = [(10, 0), (10, 60), (10, -60), (50, 0), (50, 60), (50, -60), (2, 0), (2, 60)]
k = lambda P: int(np.argmin(np.abs(p - P))); j = lambda L: int(np.argmin(np.abs(lat - L)))
tt = np.arange(1, aoa.shape[0] + 1) / 365.0
fig, ax = plt.subplots(figsize=(8, 4.5))
cols = ['#1f4e79', '#2e86c1', '#85c1e9', '#7d3c98', '#c39bd3', '#e8daef', '#b9770e', '#f5cba7']
for (P, L), c in zip(pts, cols):
    ax.plot(tt, aoa[:, k(P), j(L)], color=c, lw=1.5, label=f'{p[k(P)]:.1f} hPa, {lat[j(L)]:+.0f}')
ax.plot(tt, tt, 'k:', lw=1, label='upper bound (= elapsed time)')
ax.set_xlabel('model years (1997 met cycled)'); ax.set_ylabel('zonal-mean age of air [years]')
ax.set_title('Age-of-air spin-up at reference points'); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f'{OUT}/aoa_spinup_timeseries.png', dpi=150); plt.close(fig)

# equilibrium numbers
wl = np.cos(np.deg2rad(lat))
def mass_mean(f):
    w = (wl[None, :] * np.gradient(p)[:, None]); return float((f * w).sum() / w.sum())
notes.append(f'AOA run: {nyr} model years ({aoa.shape[0]} days), mean step {d["step_time"][100:].mean():.3f} s '
             f'(1460 steps/yr -> {d["step_time"][100:].mean()*1460/60:.1f} min/yr)')
for (P, L) in pts:
    s = f'  AOA {p[k(P)]:.1f} hPa {lat[j(L)]:+.0f}: final-yr mean {fin[k(P), j(L)]:.2f} yr'
    if prev is not None:
        s += f', drift {fin[k(P), j(L)] - prev[k(P), j(L)]:+.3f} yr/yr ({100*(fin[k(P), j(L)]-prev[k(P), j(L)])/max(fin[k(P), j(L)],1e-6):+.1f} %/yr)'
    notes.append(s)
notes.append(f'  band mass-weighted mean AOA final yr {mass_mean(fin):.2f} yr' +
             (f', drift {mass_mean(fin)-mass_mean(prev):+.3f} yr/yr' if prev is not None else ''))
if prev is not None:
    rel = np.abs(fin - prev) / np.maximum(fin, 0.1)
    notes.append(f'  fraction of band (area*dp weighted) with |drift| < 1 %/yr: '
                 f'{float(((rel < 0.01) * wl[None, :] * np.gradient(p)[:, None]).sum() / (wl[None, :] * np.gradient(p)[:, None]).sum() / 1):.2f}; '
                 f'< 5 %/yr: {float(((rel < 0.05) * wl[None, :] * np.gradient(p)[:, None]).sum() / (wl[None, :] * np.gradient(p)[:, None]).sum()):.2f}')

# AOA GIF: every 30 days over the whole run (monthly-ish snapshots of the daily zonal mean)
step = max(30, int(np.ceil(aoa.shape[0] / 150)))
frames = []
for i in list(range(step - 1, aoa.shape[0], step)):
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    latp(ax, lat, p, aoa[i], 'Blues', 0, amax, 'age of air [years]', fig, contours=np.arange(0.5, 20, 0.5))
    ax.set_title(f'Age of air (zonal mean) [years], model year {i//365 + 1}, day {i%365 + 1} '
                 f'({date_of_day(i % 365 + 1)[5:]})', fontsize=9)
    fig.tight_layout(); frames.append(fig_to_img(fig, 80)); plt.close(fig)
save_gif(frames, f'{OUT}/aoa_evolution.gif', ms=120)

# =================== INJECTION TRACER ===================
nd = inj_col.shape[0]
col_hi = float(10 ** np.ceil(np.log10(np.percentile(inj_col[-1], 99.99))))
col_lo = col_hi * 1e-4
zm = inj_zm * 1e9   # ppbm (1e-9 kg/kg)
zm_hi = float(10 ** np.ceil(np.log10(np.percentile(zm[-1], 99.99))))
zm_lo = zm_hi * 1e-4
coast = coastlines()
LON2 = np.concatenate([lon, [360.0]])

def mapax(ax, f, fig, title):
    fc = np.clip(f, col_lo, None)
    m = ax.pcolormesh(lon, lat, fc, cmap='Blues', norm=LogNorm(col_lo, col_hi), shading='nearest', rasterized=True)
    for c in coast:
        ax.plot(c[:, 0], c[:, 1], color='0.35', lw=0.5)
    ax.plot(cells[:, 2], cells[:, 1], 'x', color='#cb181d', ms=7, mew=1.6)
    ax.set_xlim(0, 360); ax.set_ylim(-90, 90); ax.set_aspect('equal')
    ax.set_xticks(range(0, 361, 60)); ax.set_xticklabels(['0', '60E', '120E', '180', '120W', '60W', '0'])
    ax.set_yticks(range(-90, 91, 30)); ax.set_yticklabels(['90S', '60S', '30S', 'EQ', '30N', '60N', '90N'])
    cb = fig.colorbar(m, ax=ax, pad=0.02, shrink=0.8); cb.set_label('column tracer mass [kg/m$^2$]')
    ax.set_title(title, fontsize=9)

def zmax(ax, f, fig, title, ylabel=True):
    latp(ax, lat, p, np.clip(f, zm_lo, None), 'Blues', None, None, 'mass mixing ratio [ppbm = 1e-9 kg/kg]', fig,
         norm=LogNorm(zm_lo, zm_hi), ylabel=ylabel)
    ax.plot(cells[:, 1], cells[:, 0], 'x', color='#cb181d', ms=7, mew=1.6)
    ax.set_title(title, fontsize=9)

frames_m, frames_z = [], []
for i in range(0, nd, 2):
    day = i + 1
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    mapax(ax, inj_col[i], fig, f'Passive injection tracer: column mass (1.2-143 hPa) [kg/m$^2$], {date_of_day(day)} (day {day})')
    fig.tight_layout(); frames_m.append(fig_to_img(fig, 75)); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    zmax(ax, zm[i], fig, f'Injection tracer, zonal-mean mixing ratio [ppbm], {date_of_day(day)} (day {day})')
    fig.tight_layout(); frames_z.append(fig_to_img(fig, 75)); plt.close(fig)
save_gif(frames_m, f'{OUT}/inj_column_map.gif', ms=130)
save_gif(frames_z, f'{OUT}/inj_zonal_mean.gif', ms=80)

days = [30, 90, 180, 365]
fig, ax = plt.subplots(4, 2, figsize=(14, 15), gridspec_kw=dict(width_ratios=[1.6, 1]))
for r, dd in enumerate(days):
    mapax(ax[r, 0], inj_col[dd - 1], fig, f'column mass [kg/m$^2$], day {dd} ({date_of_day(dd)})')
    zmax(ax[r, 1], zm[dd - 1], fig, f'zonal-mean mixing ratio [ppbm], day {dd} ({date_of_day(dd)})')
fig.suptitle('Passive injection tracer, 10 Tg/yr (5+5 at 51.7 hPa, 29.7N/S, 180E), sl K=12 transport, no removal but band-edge export', fontsize=11)
fig.tight_layout(); fig.savefig(f'{OUT}/inj_panels_d30_90_180_365.png', dpi=150); plt.close(fig)

# budget
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.plot(bud[:, 0], bud[:, 1] / 1e9, color='0.4', lw=1.5, ls='--', label='cumulative injected')
ax.plot(bud[:, 0], bud[:, 2] / 1e9, color='#2166ac', lw=2, label='in the 1.2-143 hPa band')
ax.set_xlabel('day of 1997'); ax.set_ylabel('tracer mass [Tg]'); ax.grid(alpha=0.3); ax.legend()
ax.set_title('Injection-tracer budget: export through the band edges')
fig.tight_layout(); fig.savefig(f'{OUT}/inj_budget.png', dpi=150); plt.close(fig)
for dd in days:
    b = bud[dd - 1]
    # meridional spread: fraction of column mass poleward of 45 deg
    c = inj_col[dd - 1]; A = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(c)
    fpol = float((c * A)[np.abs(lat) > 45].sum() / (c * A).sum())
    ftr = float((c * A)[np.abs(lat) < 20].sum() / (c * A).sum())
    notes.append(f'  INJ day {dd}: injected {b[1]/1e9:.3f} Tg, in band {b[2]/1e9:.3f} Tg ({100*b[2]/b[1]:.1f} %); '
                 f'column mass fraction |lat|>45: {fpol:.2f}, |lat|<20: {ftr:.2f}')
notes.append(f'  INJ colour limits: column {col_lo:.1e}-{col_hi:.1e} kg/m2, zonal mean {zm_lo:.1e}-{zm_hi:.1e} ppbm')
open(f'{OUT}/data/transport_stats.txt', 'w').write('\n'.join(notes) + '\n')
print('\n'.join(notes))
for fn in sorted(os.listdir(OUT)):
    if fn.endswith(('.gif', '.png')): print(fn, os.path.getsize(f'{OUT}/{fn}') // 1024, 'KB')
