#!/usr/bin/env python3
"""GIF: zonal-mean T and u, rows CESM | emulator | emulator - CESM, every 2 days of 1997."""
import numpy as np, matplotlib.pyplot as plt
from plotlib import DIV, latp, fig_to_img, save_gif, date_of_day
OUT = '/data/AIDE-SAI-link/linking_plots/climate_transport'
d = np.load(f'{OUT}/data/climate_zm.npz'); lat = d['lat']; p = d['plev'] / 100
cT, eT, cu, eu = d['cesm_T'], d['emu_T'], d['cesm_u'], d['emu_u']
Tlo = np.floor(min(np.percentile(cT, .5), np.percentile(eT, .5))); Thi = np.ceil(max(np.percentile(cT, 99.5), np.percentile(eT, 99.5)))
ul = float(np.ceil(max(np.percentile(np.abs(cu), 99.5), np.percentile(np.abs(eu), 99.5)) / 10) * 10)
dTl = float(np.ceil(np.percentile(np.abs(eT - cT), 99.5))); dul = float(np.ceil(np.percentile(np.abs(eu - cu), 99.5) / 5) * 5)
frames = []
for i in range(0, eT.shape[0] - 1, 3):
    fig, ax = plt.subplots(3, 2, figsize=(11, 10))
    for r, (name, T, u) in enumerate([('CESM', cT[i], cu[i]), ('emulator', eT[i], eu[i])]):
        latp(ax[r, 0], lat, p, T, 'viridis', Tlo, Thi, 'T [K]', fig, contours=np.arange(180, 300, 10))
        latp(ax[r, 1], lat, p, u, DIV, -ul, ul, 'u [m/s]', fig, contours=np.arange(-100, 101, 20), ylabel=False)
        ax[r, 0].set_title(f'{name}: zonal-mean T [K]'); ax[r, 1].set_title(f'{name}: zonal-mean u [m/s]')
    latp(ax[2, 0], lat, p, eT[i] - cT[i], DIV, -dTl, dTl, 'emulator - CESM T [K]', fig)
    latp(ax[2, 1], lat, p, eu[i] - cu[i], DIV, -dul, dul, 'emulator - CESM u [m/s]', fig, ylabel=False)
    ax[2, 0].set_title('emulator - CESM: T [K]'); ax[2, 1].set_title('emulator - CESM: u [m/s]')
    fig.suptitle(f'Zonal means, {date_of_day(i)}: CESM (top), JCM emulator P2@200 (middle), difference (bottom)', fontsize=11)
    fig.tight_layout(); frames.append(fig_to_img(fig, 56)); plt.close(fig)
save_gif(frames, f'{OUT}/climate_T_u_zonalmean_cesm_emu_diff_1997.gif', ms=130)
print(len(frames), Tlo, Thi, ul, dTl, dul)
