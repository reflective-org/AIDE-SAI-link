"""v2 figures: emulator | numerical | MAM4 (shared scale), then emulator-numerical | MAM4-numerical."""
import os, io, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from PIL import Image
from common import *

A = dict(np.load(f'{CACHE}/agg_v2.npz')); TS = dict(np.load(f'{CACHE}/timeseries_v2.npz'))
lat = A['lat']; lon = np.arange(288) * 1.25; ddays = A['ddays']
plev = np.load(f'{RUNS}/r2_tomas_fast/frames3d/step_0000.npz')['plev_pa'] / 100
dpb = np.load(f'{RUNS}/r2_tomas_fast/frames3d/step_0000.npz')['dp_bin_nm']
w = np.cos(np.deg2rad(lat)); W2 = w[:, None] / (w.sum() * 288)
gm = lambda x: float(np.nansum(x * W2) / np.nansum(W2 * np.isfinite(x)))
DIV = mcolors.LinearSegmentedColormap.from_list('bgr', ['#0b3c78', '#2f6fb3', '#8fb6dd', '#e3e3e1', '#eba08a', '#c2452f', '#6d1209'])
SEQ = mcolors.LinearSegmentedColormap.from_list('seq', ['#f3f0f7', '#bcb3d8', '#7b6bb0', '#4a2f86', '#25104f'])
COL = {'emu': '#eb6834', 'num': '#2a78d6', 'mam': '#1baf7a'}
LAB = {'emu': 'TOMAS emulator', 'num': 'TOMAS numerical', 'mam': 'MAM4 (modal, binned)'}
FIG = f'{OUT}/figs_v2'; os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9})
numbers = {}
PER = 'annual mean' if NDAYS == 365 else f'{NDAYS // 365}-year mean (1997-{1996 + NDAYS // 365})'
import datetime as _dt
_dstr = lambda d: (_dt.date(1997, 1, 1) + _dt.timedelta(days=int(d))).strftime('%d %b %Y')

def sym(*xs, q=99.0):
    v = np.concatenate([np.abs(x[np.isfinite(x)]).ravel() for x in xs]); l = np.percentile(v, q)
    return float(l) if l > 0 else 1.0

def mapax(fig, spec):
    ax = fig.add_subplot(spec, projection=ccrs.PlateCarree(central_longitude=180))
    ax.coastlines(linewidth=0.4, color='0.3'); ax.set_global()
    ax.plot([180, 180], [29.7, -29.7], 'k+', ms=7, mew=1.2, transform=ccrs.PlateCarree(), ls='none')
    return ax

def pmesh(ax, x, cmap, norm, title):
    im = ax.pcolormesh(lon, lat, x, cmap=cmap, norm=norm, shading='auto', transform=ccrs.PlateCarree(), rasterized=True)
    ax.set_title(title, fontsize=8.5); return im

def block(fig, gs, r0, F, unit, name, diverging=True, lim=None, log=False, rel=False, fmt='{:+.3g}', q=99.0, dq=99.0):
    """F = dict(emu=, num=, mam=). Rows r0 (three schemes, shared scale) and r0+1 (two differences vs numerical)."""
    if diverging:
        L = lim or sym(*F.values(), q=q); norm = mcolors.Normalize(-L, L); cm = DIV
    else:
        allv = np.concatenate([x.ravel() for x in F.values()]); allv = allv[np.isfinite(allv)]
        lo, hi = np.percentile(allv, 1), np.percentile(allv, 99)
        norm = mcolors.LogNorm(max(lo, 1e-3), hi) if log else mcolors.Normalize(lo, hi); cm = SEQ
    for j, k in enumerate(('emu', 'num', 'mam')):
        ax = mapax(fig, gs[r0, j]); im = pmesh(ax, F[k], cm, norm, f'{LAB[k]}: {name}\nglobal mean ' + fmt.format(gm(F[k])) + f' {unit}')
    fig.colorbar(im, ax=ax, shrink=0.85, extend='both', pad=0.02).set_label(unit)
    for j, k in enumerate(('emu', 'mam')):
        d = F[k] - F['num']
        if rel:
            d = 100 * d / F['num']; t = f'global mean of per-cell % {gm(d):+.2g} %'; du = '% of numerical'
        else:
            t = 'global mean ' + fmt.format(gm(d)) + f' {unit} ({100 * gm(d) / gm(F["num"]):+.1f} %)'; du = unit
        Ld = sym(d, q=dq)
        ax = mapax(fig, gs[r0 + 1, j]); im = pmesh(ax, d, DIV, mcolors.Normalize(-Ld, Ld), f'{"emulator" if k == "emu" else "MAM4"} minus numerical: {name}\n{t}')
        fig.colorbar(im, ax=ax, shrink=0.85, extend='both', pad=0.02).set_label(du + ' (own scale)')

def save(fig, name):
    fig.savefig(f'{FIG}/{name}', dpi=150, bbox_inches='tight'); plt.close(fig); print('wrote', name)

def three(pref, v):
    return {k: A[f'{pref}_{k}_{v}'] for k in ('emu', 'num', 'mam')}

ndd = len(ddays); nl30 = int((ddays >= NDAYS - 30).sum())
# ---------- 1. forcing maps (diurnal SW) ----------
for per, plab in [x for x in (('ann', f'{PER} ({ndd} days, every 5th day)'), ('l30', f'last 30 days ({nl30} sampled days in days {NDAYS - 29}-{NDAYS})')) if nl30 > 0 or x[0] == 'ann']:
    fig = plt.figure(figsize=(16, 21), layout='constrained'); gs = fig.add_gridspec(6, 3)
    block(fig, gs, 0, three(f'd{per}', 'Fnet16'), 'W m$^{-2}$', 'net forcing')
    block(fig, gs, 2, three(f'd{per}', 'Fsw16'), 'W m$^{-2}$', 'SW forcing (diurnal mean)')
    block(fig, gs, 4, three(f'd{per}', 'Flw_d'), 'W m$^{-2}$', 'LW forcing')
    fig.suptitle(f'TOA SAI forcing F = (SW$_{{up}}$+OLR)$_{{control}}$ - (SW$_{{up}}$+OLR)$_{{injection}}$, {plab}; negative = cooling.\n'
                 'SW: true diurnal mean, the model\'s own RRTMGP flux-mode radiation re-solved offline on the saved states at 16 solar times per day '
                 '(every 1.5 h). LW from the saved frames. Net = SW + LW.', fontsize=10)
    save(fig, f'fig1_{"a" if per == "ann" else "b"}_forcing_maps_{per}.png')
    for k in ('emu', 'num', 'mam'):
        for v in ('Fnet16', 'Fsw16', 'Fsw8', 'Flw_d', 'Fsw_raw_d', 'Fsw_est_d', 'Fnet_raw_d'):
            numbers[f'{per}_{k}_{v}_gm'] = gm(A[f'd{per}_{k}_{v}'])
fig = plt.figure(figsize=(16, 7), layout='constrained'); gs = fig.add_gridspec(2, 3)
block(fig, gs, 0, three('dann', 'Fsw_raw_d'), 'W m$^{-2}$', 'SW forcing, RAW frame mean')
fig.suptitle(f'For transparency: raw SW forcing (plain mean of the 4 six-hourly frames at fixed UTC times, same {ndd} days). '
             'Each longitude is always seen at the same 4 local times -> daylight lobes.', fontsize=10)
save(fig, 'fig1_c_forcing_sw_raw_annual.png')
# method comparison (numerical pair)
fig = plt.figure(figsize=(16, 7), layout='constrained'); gs = fig.add_gridspec(2, 3)
raw, est, d16, d8 = (A[f'dann_num_{v}'] for v in ('Fsw_raw_d', 'Fsw_est_d', 'Fsw16', 'Fsw8'))
L = sym(raw, est, d16); nrm = mcolors.Normalize(-L, L)
for j, (x, t) in enumerate(((raw, 'raw frame mean'), (est, 'v1 estimate: insolation-weighted albedo change x daily-mean insolation'), (d16, 'diurnal re-solve, 16 solar times/day'))):
    ax = mapax(fig, gs[0, j]); im = pmesh(ax, x, DIV, nrm, f'TOMAS numerical SW forcing, {t}\nglobal mean {gm(x):+.3f} W m$^{{-2}}$')
fig.colorbar(im, ax=ax, shrink=0.85, extend='both').set_label('W m$^{-2}$')
for j, (x, t) in enumerate(((est - d16, 'v1 estimate minus diurnal(16)'), (d8 - d16, 'diurnal(8, every 3 h) minus diurnal(16)'), (raw - d16, 'raw minus diurnal(16)'))):
    Ld = sym(x, q=99.5); ax = mapax(fig, gs[1, j]); im = pmesh(ax, x, DIV, mcolors.Normalize(-Ld, Ld), f'{t}\nglobal mean {gm(x):+.4f} W m$^{{-2}}$, p99 |diff| {np.percentile(np.abs(x), 99):.3f}')
    fig.colorbar(im, ax=ax, shrink=0.85, extend='both').set_label('W m$^{-2}$ (own scale)')
    numbers[f'method_{t}_gm'] = gm(x); numbers[f'method_{t}_p99abs'] = float(np.percentile(np.abs(x), 99))
fig.suptitle(f'SW forcing methods, TOMAS numerical pair, mean over the {ndd} re-solved days', fontsize=10)
save(fig, 'fig1_d_sw_method_comparison.png')

# ---------- 2. time series ----------
day = np.arange(1, NDAYS + 1)
def run30(x):
    k = np.ones(30) / 30; return np.convolve(x, k, 'same') / np.convolve(np.ones_like(x), k, 'same')
fig, axs = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
for ax, (v, nm) in zip(axs[:3], (('Fsw_raw', 'SW'), ('Flw', 'LW'), ('Fnet_raw', 'net'))):
    for k in ('num', 'emu', 'mam'):
        x = TS[f'{k}_{v}']; ax.plot(day, x, color=COL[k], lw=0.6, alpha=0.45)
        ax.plot(day, run30(x), color=COL[k], lw=2, label=LAB[k])
    ax.axhline(0, color='0.5', lw=0.6); ax.set_ylabel(f'{nm} forcing, W m$^{{-2}}$'); ax.grid(alpha=0.25)
    ax.set_title(f'Global-mean {nm} TOA SAI forcing (thin: daily, thick: 30-day running mean)', loc='left')
axs[0].legend(loc='lower left', frameon=False)
for k in ('emu', 'mam'):
    for v, nm, ls in (('Fnet_raw', 'net', '-'), ('Fsw_raw', 'SW', '--'), ('Flw', 'LW', ':')):
        e, n = run30(TS[f'{k}_{v}']), run30(TS[f'num_{v}'])
        pct = 100 * (e - n) / n; pct[:30] = np.nan
        axs[3].plot(day, pct, color=COL[k], lw=1.6, ls=ls, label=f'{"emulator" if k == "emu" else "MAM4"} {nm}')
axs[3].axhline(0, color='0.5', lw=0.6); axs[3].set_ylabel('minus numerical, % of numerical'); axs[3].grid(alpha=0.25)
axs[3].legend(frameon=False, ncol=2, fontsize=8); axs[3].set_xlabel('day of 1997' if NDAYS == 365 else 'day since 1996-12-31 (day 366 = 1 Jan 1998)')
axs[3].set_title('Difference from TOMAS numerical, % (30-day running means; first 30 days omitted)', loc='left')
fig.tight_layout(); save(fig, 'fig2_forcing_timeseries.png')
for k in ('emu', 'num', 'mam'):
    for v in ('Fsw_raw', 'Flw', 'Fnet_raw', 'dAOD'):
        numbers[f'ts_{k}_{v}_ann'] = float(TS[f'{k}_{v}'].mean()); numbers[f'ts_{k}_{v}_l30'] = float(TS[f'{k}_{v}'][-30:].mean())
    for v in ('Fsw16', 'Fnet16', 'Fsw_raw_d', 'Fnet_raw_d'):
        numbers[f'tsd_{k}_{v}_ann'] = float(TS[f'd_{k}_{v}'].mean())

# ---------- 3. heating ----------
def latp(ax, x, norm, title):
    im = ax.pcolormesh(lat, plev, x, cmap=DIV, norm=norm, shading='nearest', rasterized=True)
    ax.set_yscale('log'); ax.set_ylim(plev.max() * 1.05, plev.min()); ax.set_yticks([1, 2, 5, 10, 20, 50, 100, 140])
    ax.set_yticklabels(['1', '2', '5', '10', '20', '50', '100', '140']); ax.set_ylabel('hPa'); ax.set_xlabel('latitude')
    ax.set_title(title, fontsize=8.5); return im
def shared_palette(frames, n=200, k=8):
    idx = np.linspace(0, len(frames) - 1, min(k, len(frames))).astype(int)
    W = max(frames[i].size[0] for i in idx); H = sum(frames[i].size[1] for i in idx)
    canvas = Image.new('RGB', (W, H), 'white'); y = 0
    for i in idx:
        canvas.paste(frames[i], (0, y)); y += frames[i].size[1]
    return canvas.quantize(colors=n, method=Image.Quantize.MAXCOVERAGE)

def latp_fig(Z, title, dpi=None, L=None, Ld=None):
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.4), layout='constrained')
    L = L or sym(*Z.values(), q=99.5)
    for j, k in enumerate(('emu', 'num', 'mam')):
        im = latp(axs[0, j], Z[k], mcolors.Normalize(-L, L), f'{LAB[k]}  (max {np.nanmax(Z[k]):.3f} K/day)')
    fig.colorbar(im, ax=axs[0, 2], extend='both').set_label('K day$^{-1}$')
    for j, k in enumerate(('emu', 'mam')):
        d = Z[k] - Z['num']; Ldk = (Ld or {}).get(k) or sym(d, q=99.5)
        im = latp(axs[1, j], d, mcolors.Normalize(-Ldk, Ldk), f'{"emulator" if k == "emu" else "MAM4"} minus numerical')
        fig.colorbar(im, ax=axs[1, j], extend='both').set_label('K day$^{-1}$ (own scale)')
    axs[1, 2].axis('off'); fig.suptitle(title, fontsize=10); return fig
Z = {k: A[f'ann_{k}_dhr_kday_zm'] for k in ('emu', 'num', 'mam')}
save(latp_fig(Z, f'Stratospheric aerosol heating rate hr$_{{inj}}$ - hr$_{{ctl}}$, zonal mean over stratospheric cells, {PER}'), 'fig3_a_heating_latp_annual.png')
for k in Z:
    numbers[f'dhr_zm_max_{k}'] = float(np.nanmax(Z[k]))
mon = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
frames = []
MZ = {k: A[f'mon_{k}_dhr_kday_zm'] for k in ('emu', 'num', 'mam')}
LM = sym(*MZ.values(), q=99.5); LDM = {k: sym(MZ[k] - MZ['num'], q=99.5) for k in ('emu', 'mam')}
for m in range(12):
    fig = latp_fig({k: MZ[k][m] for k in MZ},
                   f'Aerosol heating rate (injection minus control), zonal mean, strat cells, {mon[m]} {"1997" if NDAYS == 365 else "(1997 and 1998 pooled)"} (colour limits fixed over all months)', L=LM, Ld=LDM)
    b = io.BytesIO(); fig.savefig(b, dpi=80, bbox_inches='tight'); plt.close(fig); frames.append(Image.open(b).convert('RGB'))
pal = shared_palette(frames)
frames = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
frames[0].save(f'{FIG}/fig3_b_heating_latp_monthly.gif', save_all=True, append_images=frames[1:], duration=1000, loop=0, optimize=False)
fig = plt.figure(figsize=(16, 7), layout='constrained'); gs = fig.add_gridspec(2, 3)
block(fig, gs, 0, {k: A[f'ann_{k}_dhr_kday_50'] for k in ('emu', 'num', 'mam')}, 'K day$^{-1}$', f'heating at {plev[17]:.1f} hPa', q=99.5, dq=99.5)
fig.suptitle(f'Aerosol heating rate hr$_{{inj}}$ - hr$_{{ctl}}$ at {plev[17]:.1f} hPa (nearest 50 hPa, the injection level), {PER}', fontsize=10)
save(fig, 'fig3_c_heating_map_50hPa_annual.png')

# ---------- 4. size statistics ----------
SZ = (('dg', 'number geom.-mean diam. D$_g$', 'nm', True), ('sig', 'geometric std. dev. $\\sigma_g$', '-', False),
      ('dmass', 'mass-weighted geom.-mean diam.', 'nm', False), ('deff_dry', 'effective diameter, dry', 'nm', False),
      ('deff_wet', 'effective diam., wet (Tabazadeh)', 'nm', False))
CAV = ('MAM4 is modal: its 4 lognormal modes (fixed sigma_g 1.8/1.6/1.8) are cut onto the 40 TOMAS bins (mam4_adapter.modes_to_bins, number and mass conserved), '
       'so its binned D$_g$/$\\sigma_g$ describe a sum of fixed-width lognormals, not a free sectional shape - not like-for-like with TOMAS.\n'
       'All columns are bimodal (nucleation + aged mode): the single-lognormal D$_g$/$\\sigma_g$ is a compromise; read with mass-weighted and effective diameters. '
       'Difference panels: per-cell % of numerical.')
for per, plab in (('ann', PER), ('l30', 'last 30 days')):
    for part, rows in (('1', SZ[:2]), ('2', SZ[2:])):
        fig = plt.figure(figsize=(16, 7 * len(rows)), layout='constrained'); gs = fig.add_gridspec(2 * len(rows), 3)
        for i, (k, nm, unit, lg) in enumerate(rows):
            block(fig, gs, 2 * i, three(per, k), unit, nm, diverging=False, log=lg, rel=True, fmt='{:.3g}')
        fig.suptitle(f'Stratospheric column size statistics, injection runs, {plab} of per-frame column values.\n' + CAV, fontsize=9)
        save(fig, f'fig4_{"a" if per == "ann" else "b"}{part}_size_maps_{per}.png')
    for k in ('emu', 'num', 'mam'):
        for v, _, _, _ in SZ:
            numbers[f'{per}_{k}_{v}_gm'] = gm(A[f'{per}_{k}_{v}'])
        numbers[f'{per}_{k}_dM_gm_mg'] = gm(A[f'{per}_{k}_dM']); numbers[f'{per}_{k}_dN_gm_1e10'] = gm(A[f'{per}_{k}_dN']) / 1e10
        numbers[f'{per}_{k}_dAOD_gm'] = gm(A[f'{per}_{k}_dAOD'])
    fig = plt.figure(figsize=(16, 14), layout='constrained'); gs = fig.add_gridspec(4, 3)
    block(fig, gs, 0, three(per, 'dM'), 'mg m$^{-2}$', 'column SO$_4$, SAI - ctl', q=99.5, dq=99.5)
    block(fig, gs, 2, {k: v / 1e10 for k, v in three(per, 'dN').items()}, '10$^{10}$ m$^{-2}$', 'column number, SAI - ctl', q=99.5, dq=99.5)
    fig.suptitle(f'Stratospheric column sulfate mass and particle number, injection minus control, {plab}', fontsize=10)
    save(fig, f'fig4_{"c" if per == "ann" else "d"}_column_NM_sai_minus_ctl_{per}.png')
dlnD = np.log(dpb[1] / dpb[0])
fig, axs = plt.subplots(1, 2, figsize=(11, 3.9))
for k in ('num', 'emu', 'mam'):
    for ax, v, nm, u in ((axs[0], 'Nk_gm', 'number', 'm$^{-2}$'), (axs[1], 'Mk_gm', 'mass', 'kg m$^{-2}$')):
        for per, ls in (('ann', '-'), ('l30', '--')):
            ax.loglog(dpb, A[f'{per}_{k}_{v}'] / (w.sum() * 288) / dlnD, color=COL[k], ls=ls, lw=1.8, label=f'{LAB[k]}, {("annual" if NDAYS == 365 else "2-yr mean") if per == "ann" else "last 30 d"}')
        ax.set_xlabel('dry diameter, nm'); ax.set_ylabel(f'd{nm}/dlnD, {u}'); ax.grid(alpha=0.25, which='both')
        ax.set_title(f'global-mean strat. column {nm} distribution, injection run', fontsize=9)
axs[0].set_ylim(bottom=1e7); axs[1].set_ylim(bottom=1e-9); axs[0].legend(frameon=False, fontsize=7)
fig.tight_layout(); save(fig, 'fig4_e_column_size_distribution.png')

# ---------- 5. tracer gif ----------
G5 = np.load(f'{CACHE}/daily_gif_v2.npz'); E = {k: G5[f'{k}_dM'] for k in ('emu', 'num', 'mam')}
L = sym(*(v[-60:] for v in E.values()), q=99.5); nrm = mcolors.Normalize(-L, L); frames = []
for d in range(0, NDAYS, 2 if NDAYS <= 365 else 4):
    fig = plt.figure(figsize=(16, 3.3)); gs = fig.add_gridspec(1, 3, wspace=0.05)
    for j, k in enumerate(('emu', 'num', 'mam')):
        ax = mapax(fig, gs[j]); im = pmesh(ax, E[k][d], DIV, nrm, f'{LAB[k]}   global mean {gm(E[k][d]):.3g} mg m$^{{-2}}$')
    fig.colorbar(im, ax=fig.axes, shrink=0.9, extend='both', pad=0.01).set_label('mg SO$_4$ m$^{-2}$')
    fig.suptitle(f'Stratospheric column sulfate, injection minus control, {_dstr(d)} (day {d + 1}, daily mean)', fontsize=10)
    b = io.BytesIO(); fig.savefig(b, dpi=int(os.environ.get('GIF5_DPI', '90')), bbox_inches='tight'); plt.close(fig)
    frames.append(Image.open(b).convert('RGB'))
pal = shared_palette(frames)
frames = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
frames[0].save(f'{FIG}/fig5_column_sulfate_transport.gif', save_all=True, append_images=frames[1:], duration=120, loop=0, optimize=False)
print('gif5 MB', os.path.getsize(f'{FIG}/fig5_column_sulfate_transport.gif') / 1e6)
numbers['ddays_n'] = ndd; numbers['ddays_l30_n'] = nl30
json.dump(numbers, open(f'{CACHE}/numbers_v2.json', 'w'), indent=1, default=float)
