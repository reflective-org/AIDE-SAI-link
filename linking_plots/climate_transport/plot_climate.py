#!/usr/bin/env python3
import os, numpy as np, matplotlib.pyplot as plt
from plotlib import DIV, latp, fig_to_img, save_gif, date_of_day
OUT = '/data/AIDE-SAI-link/linking_plots/climate_transport'
d = np.load(f'{OUT}/data/climate_zm.npz')
lat = d['lat']; p = d['plev'] / 100
eT, eu, ev, ew = d['emu_T'], d['emu_u'], d['emu_v'], d['emu_omega']
cT, cu = d['cesm_T'], d['cesm_u']
cv = d['cesm_v'] if 'cesm_v' in d else None; cw = d['cesm_omega'] if 'cesm_omega' in d else None
nd = eT.shape[0]

# ---- GIF: T and u, every 2 days, fixed limits ----
Tlo, Thi = np.floor(np.percentile(eT, 0.5)), np.ceil(np.percentile(eT, 99.5))
ulim = np.ceil(np.percentile(np.abs(eu), 99.5) / 10) * 10
frames = []
for i in (range(0, nd, 2) if not os.path.exists(f'{OUT}/climate_T_u_zonalmean_1997.gif') else []):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    latp(ax[0], lat, p, eT[i], 'viridis', Tlo, Thi, 'T [K]', fig, contours=np.arange(180, 300, 10))
    latp(ax[1], lat, p, eu[i], DIV, -ulim, ulim, 'u [m/s]', fig, contours=np.arange(-100, 101, 20), ylabel=False)
    ax[0].set_title(f'Zonal-mean temperature T [K], {date_of_day(i)}')
    ax[1].set_title(f'Zonal-mean zonal wind u [m/s], {date_of_day(i)}')
    fig.suptitle('Emulated meteorology (P2@200 JCM emulator), 1997, stratospheric band 1.2-143 hPa', fontsize=10)
    fig.tight_layout(); frames.append(fig_to_img(fig, 80)); plt.close(fig)
if not os.path.exists(f'{OUT}/climate_T_u_zonalmean_1997.gif'):
    save_gif(frames, f'{OUT}/climate_T_u_zonalmean_1997.gif', ms=100)

# ---- annual means ----
def am(x): return x[:-1].mean(0)   # 365 daily frames (drop the 1998-01-01 duplicate)
ua = max(np.abs(am(eu)).max(), np.abs(am(cu)).max()); ulim_am = float(np.ceil(ua / 10) * 10)
vl = max(np.percentile(np.abs(am(ev)), 99), np.percentile(np.abs(am(cv)), 99))
wl = max(np.percentile(np.abs(am(ew)), 99), np.percentile(np.abs(am(cw)), 99)) * 1e3
SHARED = {'u [m/s]': ulim_am, 'v [m/s]': vl, 'omega [mPa/s]': wl}
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
latp(ax[0, 0], lat, p, am(eT), 'viridis', Tlo, Thi, 'T [K]', fig, contours=np.arange(180, 300, 10))
ax[0, 0].set_title('T [K], annual mean 1997')
latp(ax[0, 1], lat, p, am(eu), DIV, -ulim_am, ulim_am, 'u [m/s]', fig, contours=np.arange(-100, 101, 10), ylabel=False)
ax[0, 1].set_title('u [m/s], annual mean 1997')
latp(ax[1, 0], lat, p, am(ev), DIV, -vl, vl, 'v [m/s]', fig)
ax[1, 0].set_title('v [m/s], annual mean 1997')
latp(ax[1, 1], lat, p, am(ew) * 1e3, DIV, -wl, wl, 'omega [mPa/s]  (>0 = descent)', fig, ylabel=False)
ax[1, 1].set_title('omega [mPa/s], annual mean 1997')
fig.suptitle('Emulated meteorology (P2@200 JCM emulator): annual-mean zonal means, 1997')
fig.tight_layout(); fig.savefig(f'{OUT}/climate_annual_mean_emulator.png', dpi=150); plt.close(fig)

# ---- emulator minus CESM ----
pairs = [('T [K]', eT, cT, 1), ('u [m/s]', eu, cu, 1)]
if cv is not None: pairs.append(('v [m/s]', ev, cv, 1))
if cw is not None: pairs.append(('omega [mPa/s]', ew, cw, 1e3))
nc = len(pairs)
fig, ax = plt.subplots(2, nc, figsize=(5 * nc, 8), squeeze=False)
stats = []
for j, (lab, e, c, sc) in enumerate(pairs):
    da = (am(e) - am(c)) * sc
    lim = float(np.percentile(np.abs(da), 99))
    lim = float(np.ceil(lim)) if lim > 2 else lim
    if lab.startswith('T'):
        latp(ax[0, j], lat, p, am(c), 'viridis', Tlo, Thi, lab, fig, contours=np.arange(180, 300, 10), ylabel=(j == 0))
    else:
        cl = SHARED[lab]
        latp(ax[0, j], lat, p, am(c) * sc, DIV, -cl, cl, lab, fig, ylabel=(j == 0),
             contours=np.arange(-100, 101, 10) if lab.startswith('u') else None)
    ax[0, j].set_title(f'CESM {lab.split()[0]}, annual mean 1997')
    latp(ax[1, j], lat, p, da, DIV, -lim, lim, 'emulator - CESM ' + lab, fig, ylabel=(j == 0))
    ax[1, j].set_title(f'emulator - CESM: {lab}')
    wl_ = np.cos(np.deg2rad(lat)); wl_ /= wl_.sum()
    rms = np.sqrt(((da ** 2) * wl_).sum(-1).mean())
    stats.append((lab, float(rms), float(np.abs(da).max())))
fig.suptitle('Annual-mean zonal means 1997 (CESM colour limits = emulator panels): CESM archive (top) and emulator minus CESM (bottom), same hours, daily samples')
fig.tight_layout(); fig.savefig(f'{OUT}/climate_annual_mean_emu_minus_cesm.png', dpi=150); plt.close(fig)
with open(f'{OUT}/data/climate_stats.txt', 'w') as f:
    for lab, r, m in stats:
        f.write(f'{lab}: annual-mean zonal-mean diff area-weighted RMS {r:.3g}, max|diff| {m:.3g}\n')
    f.write(f'T limits {Tlo}-{Thi} K, u limits +-{ulim}\n')
    # a few headline numbers
    j = lambda L: int(np.argmin(np.abs(lat - L))); k = lambda P: int(np.argmin(np.abs(p - P)))
    f.write(f'annual-mean T at 100 hPa EQ: emu {am(eT)[k(100), j(0)]:.1f} cesm {am(cT)[k(100), j(0)]:.1f}\n')
    f.write(f'max annual-mean u (emu/cesm): {am(eu).max():.1f}/{am(cu).max():.1f}; min {am(eu).min():.1f}/{am(cu).min():.1f}\n')
print(open(f'{OUT}/data/climate_stats.txt').read())
for fn in os.listdir(OUT):
    if fn.endswith(('.gif', '.png')): print(fn, os.path.getsize(f'{OUT}/{fn}') // 1024, 'KB')
