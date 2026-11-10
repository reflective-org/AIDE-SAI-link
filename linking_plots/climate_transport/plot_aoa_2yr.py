#!/usr/bin/env python3
"""Age-of-air figures for the 10-year runs on the cycled 1997-1998 emulator winds (2026-11-09).
Inputs: data/age2yr_fullcol_p150_sfc.npz (resets p150 and sfc), data/age2yr_band_trop143.npz (published setup on the
2-year winds), data/aoa_refs.npz (aoa_refs.py: CLaMS-ERA5 2000-2014 = the ERA5 age of air; WACCM6 REF-D1.04 1997-1998 =
the CESM age of air; CLaMS off150).
Model climatology = mean over the LAST full 1997-1998 cycle (years 9-10 when 10 years are done); drift = that minus the
previous cycle (years 7-8).
Figures (figs/):
  aoa2yr_surface_vs_150.png   surface-reset vs 150-hPa-reset age and their difference, 1-1000 hPa
  aoa2yr_sfc_vs_clams.png     surface-reset age vs CLaMS-ERA5 (both surface clocks), 1-300 hPa
  aoa2yr_entry_vs_refs.png    150-hPa-reset and band (143 hPa tropical) ages vs WACCM6 entry age (based at 103 hPa) and
                              vs CLaMS - off150 (the jcm-strat sweep_score convention)
  aoa2yr_profiles.png         tropical 10S-10N profile, latitude profiles near 55 and 12 hPa, spin-up of tropical 20 hPa age
Numbers to data/aoa2yr_stats.txt."""
import numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.colors as mcolors
D = '/data/AIDE-SAI-link/linking_plots/climate_transport/data'; FIGD = '/data/AIDE-SAI-link/linking_plots/climate_transport/figs'
os.makedirs(FIGD, exist_ok=True)
DIV = mcolors.LinearSegmentedColormap.from_list('bgr', ['#0b3c78', '#2f6fb3', '#8fb6dd', '#e3e3e1', '#eba08a', '#c2452f', '#6d1209'])
SEQ = 'magma_r'
plt.rcParams.update({'font.size': 8.5, 'axes.titlesize': 8.5})
F = np.load(f'{D}/age2yr_fullcol_p150_sfc.npz'); B = np.load(f'{D}/age2yr_band_trop143.npz'); R = np.load(f'{D}/aoa_refs.npz')
lat = F['lat']; pF = F['plev'] / 100; pB = B['plev'] / 100
CYC = 730


def clim(z):
    a = z['aoa_zm']; n = a.shape[0] // CYC
    assert n >= 1, a.shape
    last = a[(n - 1) * CYC:n * CYC].mean(0)
    prev = a[(n - 2) * CYC:(n - 1) * CYC].mean(0) if n >= 2 else np.full_like(last, np.nan)
    return last, prev, n


cF, pFv, nF = clim(F); cB, pBv, nB = clim(B)
RES = {str(r): i for i, r in enumerate(F['resets'])}
sfc, p150 = cF[RES['sfc']], cF[RES['p150']]; band = cB[0]
d_sfc = (sfc - pFv[RES['sfc']]) / 2; d_150 = (p150 - pFv[RES['p150']]) / 2; d_band = (band - pBv[0]) / 2      # yr per yr
CL = R['clams_ann']; CLp, CLl = R['clams_p'], R['clams_lat']; OFF = float(R['clams_off150'])
WA = R['waccm_ann']; WAp, WAl = R['waccm_p'], R['waccm_lat']
o = np.argsort(CLp); CLp_s, CL_s = CLp[o], CL[o]                    # ascending p for interpolation
o2 = np.argsort(WAp); WAp_s, WA_s = WAp[o2], WA[o2]


def to_model(x, px, lx, pt):
    """reference (np, nlat) on ascending px -> model (len(pt), len(lat)); lat then log p; NaN outside."""
    xl = np.stack([np.interp(lat, lx, x[i], left=np.nan, right=np.nan) for i in range(x.shape[0])])
    out = np.stack([np.interp(np.log(pt), np.log(px), xl[:, j], left=np.nan, right=np.nan) for j in range(lat.size)], 1)
    return out


def axp(ax, pmax=300):
    ax.set_yscale('log'); ax.set_ylim(pmax, 1); t = [1, 2, 5, 10, 20, 50, 100, 200, 300, 500, 1000]
    t = [x for x in t if x <= pmax]; ax.set_yticks(t); ax.set_yticklabels([str(x) for x in t]); ax.set_xlim(-90, 90)
    ax.set_xticks([-60, -30, 0, 30, 60]); ax.set_ylabel('hPa')


def panel(ax, x, la, p, norm, cm, title, pmax=300, contours=np.arange(0.5, 8, 0.5)):
    im = ax.pcolormesh(la, p, x, cmap=cm, norm=norm, shading='nearest', rasterized=True)
    if contours is not None:
        cs = ax.contour(la, p, x, levels=contours, colors='k', linewidths=0.45); ax.clabel(cs, fmt='%.1f', fontsize=6)
    axp(ax, pmax); ax.set_title(title); return im


def wst(d, p, pmin=5, pmax=100, latmax=80):
    """cos-lat weighted RMS / mean over pmin..pmax hPa, |lat| <= latmax (the sweep_score window: 100-5 hPa, |lat|<=80)."""
    m = ((p >= pmin) & (p <= pmax))[:, None] & (np.abs(lat) <= latmax)[None] & np.isfinite(d)
    w = (np.cos(np.deg2rad(lat))[None] * np.ones_like(d))[m]
    return float(np.sqrt((w * d[m] ** 2).sum() / w.sum())), float((w * d[m]).sum() / w.sum())


S = [f'model cycles complete: fullcol {nF} (years {2 * nF - 1}-{2 * nF} used), band {nB}',
     f'CLaMS-ERA5 2000-2014 off150 (0-10 deg, {float(R["clams_p150"]):.1f} hPa) = {OFF:.3f} yr']
VMAX = 6.0
# ---- A: surface vs 150 hPa reset ----
fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), layout='constrained')
nrm = mcolors.Normalize(0, VMAX)
panel(axs[0], sfc, lat, pF, nrm, SEQ, 'age, reset at the SURFACE (p ≥ ps)', 1000)
im = panel(axs[1], p150, lat, pF, nrm, SEQ, 'age, reset at p ≥ 150 hPa (entry age)', 1000)
fig.colorbar(im, ax=axs[:2], shrink=0.85, extend='max').set_label('age of air, yr')
dd = sfc - p150
im = panel(axs[2], dd, lat, pF, mcolors.Normalize(0, np.nanpercentile(dd[pF <= 150], 99.5)), 'viridis',
           'surface-reset minus 150-hPa-reset\n(transit time from the surface into the 150 hPa entry region)', 1000, contours=np.arange(0, 1.5, 0.1))
fig.colorbar(im, ax=axs[2], shrink=0.85).set_label('yr')
fig.suptitle(f'Emulator winds (1997-1998 cycled, 10 yr): age of air for two reset conditions, mean of the last 1997-1998 cycle', fontsize=10)
fig.savefig(f'{FIGD}/aoa2yr_surface_vs_150.png', dpi=140); plt.close(fig)
k = lambda p, pp: int(np.argmin(abs(p - pp)))
eq = np.abs(lat) <= 10; we = np.cos(np.deg2rad(lat[eq]))
trm = lambda x, kk: float((x[kk, eq] * we).sum() / we.sum())
S.append(f'sfc - p150 at 50 hPa, 10S-10N: {trm(dd, k(pF, 50)):.3f} yr; at 10 hPa {trm(dd, k(pF, 10)):.3f} yr')
# ---- B: surface reset vs CLaMS ----
CLm = to_model(CL_s, CLp_s, CLl, pF)
fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), layout='constrained')
panel(axs[0], sfc, lat, pF, nrm, SEQ, 'emulator winds, surface-reset age\n(last 1997-1998 cycle of 10 yr)')
im = panel(axs[1], CL, CLl, CLp, nrm, SEQ, 'CLaMS driven by ERA5 (surface clock)\n2000-2014 climatology (closest available to 1997-98)')
fig.colorbar(im, ax=axs[:2], shrink=0.85, extend='max').set_label('age of air, yr')
d = sfc - CLm; L = np.nanpercentile(np.abs(d[pF <= 300]), 99)
r, m = wst(d, pF)
im = panel(axs[2], d, lat, pF, mcolors.Normalize(-L, L), DIV, f'emulator minus CLaMS-ERA5\nRMS {r:.2f} yr, mean {m:+.2f} yr (100-5 hPa, |lat|≤80)', contours=None)
fig.colorbar(im, ax=axs[2], shrink=0.85, extend='both').set_label('yr')
fig.suptitle('Surface-reset age of air: emulator winds vs CLaMS-ERA5 (both surface clocks, no offset)', fontsize=10)
fig.savefig(f'{FIGD}/aoa2yr_sfc_vs_clams.png', dpi=140); plt.close(fig)
S.append(f'sfc vs CLaMS-ERA5: RMS {r:.3f} mean {m:+.3f} yr')
# ---- C: entry ages vs WACCM and CLaMS - off150 ----
WAm = to_model(WA_s, WAp_s, WAl, pF); WAmB = to_model(WA_s, WAp_s, WAl, pB)
CLo = CLm - OFF; CLoB = to_model(CL_s, CLp_s, CLl, pB) - OFF
fig, axs = plt.subplots(2, 4, figsize=(20, 8.6), layout='constrained')
panel(axs[0, 0], p150, lat, pF, nrm, SEQ, 'emulator winds, reset at p ≥ 150 hPa')
panel(axs[0, 1], band, lat, pB, nrm, SEQ, 'emulator winds, band only (published setup):\nreset at 143 hPa, 10S-10N')
panel(axs[0, 2], WA, WAl, WAp, nrm, SEQ, 'WACCM6 REF-D1.04 (CESM) entry age, 1997-1998\n(based at 103 hPa, 0.5N; < 0 masked)')
im = panel(axs[0, 3], np.where(CL - OFF > 0, CL - OFF, np.nan), CLl, CLp, nrm, SEQ, f'CLaMS-ERA5 2000-2014 minus off150 ({OFF:.3f} yr)')
fig.colorbar(im, ax=axs[0, :], shrink=0.85, extend='max').set_label('age of air, yr')
diffs = (('p150 minus WACCM6', p150 - WAm, pF), ('p150 minus (CLaMS - off150)', p150 - CLo, pF),
         ('band minus WACCM6', band - WAmB, pB), ('band minus (CLaMS - off150)', band - CLoB, pB))
Ld = np.nanpercentile(np.abs(np.concatenate([x[((pp >= 5) & (pp <= 100))].ravel() for _, x, pp in diffs])), 99)
for j, (nm, x, pp) in enumerate(diffs):
    latm = (np.abs(lat) <= 80)
    m_ = ((pp >= 5) & (pp <= 100))[:, None] & latm[None] & np.isfinite(x)
    w = (np.cos(np.deg2rad(lat))[None] * np.ones_like(x))[m_]
    r_, mm = float(np.sqrt((w * x[m_] ** 2).sum() / w.sum())), float((w * x[m_]).sum() / w.sum())
    im2 = panel(axs[1, j], x, lat, pp, mcolors.Normalize(-Ld, Ld), DIV, f'{nm}\nRMS {r_:.2f}, mean {mm:+.2f} yr (100-5 hPa, |lat|≤80)', contours=None)
    S.append(f'{nm}: RMS {r_:.3f} mean {mm:+.3f} yr')
fig.colorbar(im2, ax=axs[1, :], shrink=0.85, extend='both').set_label('yr')
fig.suptitle('Entry-type ages: emulator winds vs the CESM (WACCM6) age of air and vs CLaMS-ERA5 shifted by its tropical 150 hPa age', fontsize=10)
fig.savefig(f'{FIGD}/aoa2yr_entry_vs_refs.png', dpi=140); plt.close(fig)
# ---- D: profiles + spin-up ----
fig, axs = plt.subplots(1, 4, figsize=(20, 4.8), layout='constrained')
C = {'sfc': '#c2452f', 'p150': '#2f6fb3', 'band': '#7b6bb0', 'clams': 'k', 'clamso': '0.45', 'waccm': '#1baf7a'}
def tprof(x, la, p):
    m = np.abs(la) <= 10; w = np.cos(np.deg2rad(la[m])); return (x[:, m] * w).sum(1) / w.sum()
ax = axs[0]
for x, la, p, c, nm, ls in ((sfc, lat, pF, C['sfc'], 'emulator, surface reset', '-'), (p150, lat, pF, C['p150'], 'emulator, 150 hPa reset', '-'),
                            (band, lat, pB, C['band'], 'emulator, band (143 hPa tropics)', '--'), (CL, CLl, CLp, C['clams'], 'CLaMS-ERA5 2000-14', '-'),
                            (CL - OFF, CLl, CLp, C['clamso'], 'CLaMS-ERA5 - off150', ':'), (WA, WAl, WAp, C['waccm'], 'WACCM6 1997-98 (entry)', '-')):
    ax.plot(tprof(x, la, p), p, color=c, ls=ls, lw=1.6, label=nm)
ax.set_yscale('log'); ax.set_ylim(300, 1); ax.set_xlim(0, None); ax.set_xlabel('age, yr'); ax.set_ylabel('hPa'); ax.grid(alpha=0.3)
ax.set_title('10S-10N mean profile'); ax.legend(fontsize=7, frameon=False)
for ax, pp in ((axs[1], 55), (axs[2], 12)):
    for x, la, p, c, nm, ls in ((sfc, lat, pF, C['sfc'], 'surface reset', '-'), (p150, lat, pF, C['p150'], '150 hPa reset', '-'),
                                (band, lat, pB, C['band'], 'band', '--'), (CL, CLl, CLp, C['clams'], 'CLaMS-ERA5', '-'),
                                (CL - OFF, CLl, CLp, C['clamso'], 'CLaMS - off150', ':'), (WA, WAl, WAp, C['waccm'], 'WACCM6', '-')):
        kk = int(np.argmin(abs(p - pp))); ax.plot(la, x[kk], color=c, ls=ls, lw=1.6, label=f'{nm} ({p[kk]:.1f} hPa)')
    ax.set_xlim(-90, 90); ax.set_xlabel('latitude'); ax.set_ylabel('age, yr'); ax.grid(alpha=0.3); ax.set_title(f'near {pp} hPa'); ax.legend(fontsize=6.5, frameon=False)
ax = axs[3]
for z, key, c, nm in ((F, 'sfc', C['sfc'], 'surface reset'), (F, 'p150', C['p150'], '150 hPa reset'), (B, 'trop143', C['band'], 'band')):
    a = z['aoa_zm'][:, {str(r): i for i, r in enumerate(z['resets'])}[key]]; pz = z['plev'] / 100; kk = int(np.argmin(abs(pz - 20)))
    t = np.arange(a.shape[0]) / 365; ax.plot(t, (a[:, kk][:, eq] * we).sum(1) / we.sum(), color=c, lw=1.2, label=nm)
ax.axhline(float(tprof(CL, CLl, CLp)[int(np.argmin(abs(CLp - 20)))]), color='k', lw=0.8, ls='--', label='CLaMS-ERA5 20 hPa')
ax.set_xlabel('model year'); ax.set_ylabel('10S-10N age at 20 hPa, yr'); ax.grid(alpha=0.3); ax.legend(fontsize=7, frameon=False)
ax.set_title('spin-up (daily; the seasonal cycle is real)')
fig.suptitle('Age of air profiles: emulator winds (last 1997-1998 cycle) vs CLaMS-ERA5 and WACCM6', fontsize=10)
fig.savefig(f'{FIGD}/aoa2yr_profiles.png', dpi=140); plt.close(fig)
for nm, x, p in (('sfc', d_sfc, pF), ('p150', d_150, pF), ('band', d_band, pB)):
    if np.isfinite(x).any():
        w = np.cos(np.deg2rad(lat))[None] * np.ones_like(x); mm = (p >= 5)[:, None] & (p <= 100)[:, None] & np.isfinite(x)
        S.append(f'drift {nm} (last cycle minus previous, per yr), 100-5 hPa mean {float((w[mm] * x[mm]).sum() / w[mm].sum()):+.4f} yr/yr, max |.| {float(np.nanmax(np.abs(x[(p >= 5) & (p <= 100)]))):.4f}')
for nm, x, p in (('sfc', sfc, pF), ('p150', p150, pF), ('band', band, pB), ('CLaMS', CL, CLp), ('WACCM', WA, WAp)):
    la = lat if nm in ('sfc', 'p150', 'band') else (CLl if nm == 'CLaMS' else WAl)
    tp = tprof(x, la, p); S.append(f'{nm:6s} 10S-10N age at ~50 hPa {tp[int(np.argmin(abs(p - 50)))]:.2f}, ~20 hPa {tp[int(np.argmin(abs(p - 20)))]:.2f}, ~10 hPa {tp[int(np.argmin(abs(p - 10)))]:.2f} yr')
open(f'{D}/aoa2yr_stats.txt', 'w').write('\n'.join(S) + '\n'); print('\n'.join(S))
