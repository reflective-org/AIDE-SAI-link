#!/usr/bin/env python3
"""ERA5 vs CESM vs JCM emulator zonal means, 1997: annual-mean 4-row PNG, T/u GIF, stats."""
import os, numpy as np, matplotlib.pyplot as plt
from plotlib import DIV, latp, fig_to_img, save_gif, date_of_day
OUT = '/data/AIDE-SAI-link/linking_plots/climate_transport'
d = np.load(f'{OUT}/data/climate_zm.npz'); r = np.load(f'{OUT}/data/era5_zm.npz')
lat = d['lat']; p = d['plev'] / 100
assert np.array_equal(r['hours'], d['hours'])
F = {k: (r[f'era5_{k}'], d[f'cesm_{k}'], d[f'emu_{k}']) for k in ('T', 'u', 'v')}
def am(x): return x[:-1].mean(0)   # 365 daily frames (drop the 1998-01-01 duplicate)
wl_ = np.cos(np.deg2rad(lat)); wl_ /= wl_.sum()
def rms(x): return float(np.sqrt(((x ** 2) * wl_).sum(-1).mean()))
CT = np.arange(180, 300, 10)
LAB = {'T': 'T [K]', 'u': 'u [m/s]', 'v': 'v [m/s]'}
ROWS = ['ERA5', 'CESM', 'emulator']

# ---- 1. annual mean, 4 rows ----
A = {k: [am(x) for x in F[k]] for k in F}
Tlo = float(np.floor(min(a.min() for a in A['T']))); Thi = float(np.ceil(max(a.max() for a in A['T'])))
ulim = float(np.ceil(max(np.abs(a).max() for a in A['u']) / 10) * 10)
vlim = float(max(np.percentile(np.abs(a), 99) for a in A['v']))
fig, ax = plt.subplots(4, 3, figsize=(15, 14))
for j, k in enumerate(('T', 'u', 'v')):
    for i, name in enumerate(ROWS):
        if k == 'T':
            latp(ax[i, j], lat, p, A[k][i], 'viridis', Tlo, Thi, LAB[k], fig, contours=CT, ylabel=(j == 0))
        else:
            lim = ulim if k == 'u' else vlim
            latp(ax[i, j], lat, p, A[k][i], DIV, -lim, lim, LAB[k], fig, ylabel=(j == 0),
                 contours=np.arange(-100, 101, 10) if k == 'u' else None)
        ax[i, j].set_title(f'{name}: {LAB[k]}, annual mean 1997')
    da = A[k][2] - A[k][0]
    lim = float(np.percentile(np.abs(da), 99)); lim = float(np.ceil(lim)) if lim > 2 else lim
    latp(ax[3, j], lat, p, da, DIV, -lim, lim, 'emulator - ERA5 ' + LAB[k], fig, ylabel=(j == 0))
    ax[3, j].set_title(f'emulator - ERA5: {LAB[k]}')
fig.suptitle('Annual-mean zonal means 1997, 1.2-143 hPa: ERA5 (WB2 37-level, 06 UTC daily), CESM archive, '
             'JCM emulator P2@200 (shared colour limits), and emulator - ERA5', fontsize=11)
fig.tight_layout(); fig.savefig(f'{OUT}/climate_annual_mean_era5_cesm_emu_4row.png', dpi=150); plt.close(fig)

# ---- 3. stats ----
j = lambda L: int(np.argmin(np.abs(lat - L))); kk = lambda P: int(np.argmin(np.abs(p - P)))
eq = np.abs(lat) <= 10
with open(f'{OUT}/data/era5_stats.txt', 'w') as f:
    f.write('Annual-mean zonal means 1997, 1.2-143 hPa band, 365 daily 06 UTC samples; RMS area-weighted (cos lat), equal weight per level\n')
    for k in ('T', 'u', 'v'):
        e, c, m = A[k]
        f.write(f'{LAB[k]}: RMS(emu-ERA5) {rms(m - e):.3g}  RMS(CESM-ERA5) {rms(c - e):.3g}  RMS(emu-CESM) {rms(m - c):.3g}; '
                f'max|emu-ERA5| {np.abs(m - e).max():.3g}  max|CESM-ERA5| {np.abs(c - e).max():.3g}\n')
    # band-restricted RMS
    for lo, hi in ((1, 10), (10, 50), (50, 150)):
        s = (p >= lo) & (p < hi)
        f.write(f'  {lo}-{hi} hPa: T RMS emu-ERA5 {rms(A["T"][2][s]-A["T"][0][s]):.2f} CESM-ERA5 {rms(A["T"][1][s]-A["T"][0][s]):.2f}; '
                f'u RMS emu-ERA5 {rms(A["u"][2][s]-A["u"][0][s]):.2f} CESM-ERA5 {rms(A["u"][1][s]-A["u"][0][s]):.2f}\n')
    for P in (100, 50, 10, 1.6):
        f.write(f'annual-mean T at {p[kk(P)]:.1f} hPa, 10S-10N mean: ' + '  '.join(
            f'{n} {(A["T"][i][kk(P), eq] * wl_[eq]).sum() / wl_[eq].sum():.1f}' for i, n in enumerate(ROWS)) + '\n')
    f.write('annual-mean T min (cold point) [K @ hPa, lat]: ' + '  '.join(
        f'{n} {A["T"][i].min():.1f} @ {p[np.unravel_index(A["T"][i].argmin(), A["T"][i].shape)[0]]:.0f}hPa,{lat[np.unravel_index(A["T"][i].argmin(), A["T"][i].shape)[1]]:.0f}' for i, n in enumerate(ROWS)) + '\n')
    f.write('annual-mean T max [K @ hPa]: ' + '  '.join(
        f'{n} {A["T"][i].max():.1f} @ {p[np.unravel_index(A["T"][i].argmax(), A["T"][i].shape)[0]]:.1f}' for i, n in enumerate(ROWS)) + '\n')
    for hemi, sl in (('SH', lat < 0), ('NH', lat > 0)):
        f.write(f'annual-mean peak westerly u {hemi} [m/s @ hPa, lat]: ' + '  '.join(
            f'{n} {A["u"][i][:, sl].max():.1f} @ {p[np.unravel_index(A["u"][i][:, sl].argmax(), A["u"][i][:, sl].shape)[0]]:.1f},{lat[sl][np.unravel_index(A["u"][i][:, sl].argmax(), A["u"][i][:, sl].shape)[1]]:.0f}' for i, n in enumerate(ROWS)) + '\n')
    f.write('annual-mean min u (easterly) [m/s]: ' + '  '.join(f'{n} {A["u"][i].min():.1f}' for i, n in enumerate(ROWS)) + '\n')
    # daily extremes: polar-night jet peaks
    for hemi, sl in (('SH', lat < 0), ('NH', lat > 0)):
        f.write(f'max daily zonal-mean u over 1997, {hemi}: ' + '  '.join(
            f'{n} {F["u"][i][:-1][:, :, sl].max():.1f}' for i, n in enumerate(ROWS)) + '\n')
    f.write(f'colour limits: T {Tlo}-{Thi} K, u +-{ulim}, v +-{vlim:.3g}\n')
print(open(f'{OUT}/data/era5_stats.txt').read())

# ---- 2. GIF: T and u, every 3rd day ----
eT, cT, mT = F['T']; eu, cu, mu = F['u']
gTlo = np.floor(min(np.percentile(x, .5) for x in F['T'])); gThi = np.ceil(max(np.percentile(x, 99.5) for x in F['T']))
gul = float(np.ceil(max(np.percentile(np.abs(x), 99.5) for x in F['u']) / 10) * 10)
dTl = float(np.ceil(np.percentile(np.abs(mT - eT), 99.5))); dul = float(np.ceil(np.percentile(np.abs(mu - eu), 99.5) / 5) * 5)
GIF = f'{OUT}/climate_T_u_zonalmean_era5_cesm_emu_1997.gif'
for step, dpi in ((3, 55), (3, 50), (4, 50), (6, 50)):
    frames = []
    for i in range(0, eT.shape[0] - 1, step):
        fig, ax = plt.subplots(4, 2, figsize=(11, 13))
        for rr, name in enumerate(ROWS):
            latp(ax[rr, 0], lat, p, F['T'][rr][i], 'viridis', gTlo, gThi, 'T [K]', fig, contours=CT)
            latp(ax[rr, 1], lat, p, F['u'][rr][i], DIV, -gul, gul, 'u [m/s]', fig, contours=np.arange(-100, 101, 20), ylabel=False)
            ax[rr, 0].set_title(f'{name}: zonal-mean T [K]'); ax[rr, 1].set_title(f'{name}: zonal-mean u [m/s]')
        latp(ax[3, 0], lat, p, mT[i] - eT[i], DIV, -dTl, dTl, 'emulator - ERA5 T [K]', fig)
        latp(ax[3, 1], lat, p, mu[i] - eu[i], DIV, -dul, dul, 'emulator - ERA5 u [m/s]', fig, ylabel=False)
        ax[3, 0].set_title('emulator - ERA5: T [K]'); ax[3, 1].set_title('emulator - ERA5: u [m/s]')
        fig.suptitle(f'Zonal means, {date_of_day(i)}: ERA5, CESM, JCM emulator P2@200, emulator - ERA5', fontsize=11)
        fig.tight_layout(); frames.append(fig_to_img(fig, dpi)); plt.close(fig)
    save_gif(frames, GIF, ms=130)
    sz = os.path.getsize(GIF) / 2**20
    print(f'gif every {step} days dpi {dpi}: {len(frames)} frames {sz:.1f} MB; limits T {gTlo}-{gThi} u+-{gul} dT+-{dTl} du+-{dul}')
    if sz <= 12: break
