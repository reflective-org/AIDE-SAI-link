"""Global-mean stratospheric aerosol heating (injection - control), daily, emulator vs numerical: SW, LW, net, % diff.
Mass-weighted (cos lat * dp) mean over stratospheric band cells, from cache/day_DDD.npz zonal sums."""
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
C = '/data/AIDE-SAI-link/linking_plots/coupled/cache'
lat = np.load(f'{C}/timeseries.npz')['lat']
plev = np.load('/data/AIDE-SAI-link/runs/emu_campaign_1997/r2_tomas_fast/frames3d/step_0400.npz')['plev_pa']
w = np.abs(np.gradient(plev))[:, None] * np.cos(np.deg2rad(lat))[None, :]
out = {k: np.zeros(365) for k in ('emu_net', 'emu_sw', 'num_net', 'num_sw', 'emu_pk', 'num_pk')}
for d in range(365):
    z = np.load(f'{C}/day_{d:03d}.npz'); cnt = z['strat_zcount']
    den = (cnt * w).sum()
    for m in ('emu', 'num'):
        out[f'{m}_net'][d] = (z[f'{m}_dhr_kday_zsum'] * w).sum() / den
        out[f'{m}_sw'][d] = (z[f'{m}_dhr_sw_kday_zsum'] * w).sum() / den
        zm = np.where(cnt > 0, z[f'{m}_dhr_kday_zsum'] / np.maximum(cnt, 1), np.nan)
        out[f'{m}_pk'][d] = np.nanmax(zm)
for m in ('emu', 'num'): out[f'{m}_lw'] = out[f'{m}_net'] - out[f'{m}_sw']
np.savez(f'{C}/../extra/heating_timeseries.npz', **out)
day = np.arange(1, 366); run = lambda x: np.convolve(x, np.ones(30) / 30, 'valid')
CE, CN = '#eb6834', '#2a78d6'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
for a, (k, t) in zip(ax[:3], (('sw', 'shortwave (solar near-IR absorption)'), ('lw', 'longwave'), ('net', 'net (SW + LW)'))):
    for m, c, lab in (('emu', CE, 'TOMAS emulator'), ('num', CN, 'TOMAS numerical')):
        a.plot(day, out[f'{m}_{k}'] * 1e3, color=c, lw=0.6, alpha=0.5)
        a.plot(day[29:], run(out[f'{m}_{k}']) * 1e3, color=c, lw=2, label=lab)
    a.axhline(0, color='0.6', lw=0.6); a.set_ylabel('mK / day'); a.set_title(f'Aerosol heating rate, {t}', loc='left')
ax[0].legend(frameon=False)
for k, ls in (('sw', ':'), ('lw', '--'), ('net', '-')):
    e, n = run(out[f'emu_{k}']), run(out[f'num_{k}'])
    ax[3].plot(day[29:], 100 * (e - n) / n, color='k', ls=ls, lw=1.5, label=k.upper() if k != 'net' else 'net')
ax[3].axhline(0, color='0.6', lw=0.6); ax[3].set_ylabel('%'); ax[3].legend(frameon=False, ncol=3)
ax[3].set_title('Emulator minus numerical, % of numerical (30-day running means)', loc='left'); ax[3].set_xlabel('day of 1997')
fig.suptitle('ΔT as heating: global mass-weighted mean over stratospheric cells, injection minus control\n'
             'thin = daily, thick = 30-day running mean', fontsize=11)
fig.tight_layout(); fig.savefig('/data/AIDE-SAI-link/linking_plots/coupled/extra/fig3_d_heating_timeseries.png', dpi=150)
for m in ('emu', 'num'):
    print(m, 'last30 net %.4f sw %.4f lw %.4f K/day; annual net %.4f; peak zm last30 %.3f' % tuple(
        [out[f'{m}_{k}'][-30:].mean() for k in ('net', 'sw', 'lw')] + [out[f'{m}_net'].mean(), out[f'{m}_pk'][-30:].mean()]))
