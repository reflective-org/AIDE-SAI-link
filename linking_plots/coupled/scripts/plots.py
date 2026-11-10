import os, io, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from PIL import Image
from common import *

A = dict(np.load(f'{CACHE}/agg.npz')); TS = dict(np.load(f'{CACHE}/timeseries.npz'))
lat = A['lat']; lon = np.arange(288) * 1.25
plev = np.load(f'{RUNS}/r2_tomas_fast/frames3d/step_0000.npz')['plev_pa'] / 100
w = np.cos(np.deg2rad(lat)); W2 = w[:, None] / (w.sum() * 288)
gm = lambda x: float(np.nansum(x * W2) / np.nansum(W2 * np.isfinite(x)))
DIV = mcolors.LinearSegmentedColormap.from_list('bgr', ['#0b3c78', '#2f6fb3', '#8fb6dd', '#e3e3e1', '#eba08a', '#c2452f', '#6d1209'])
SEQ = mcolors.LinearSegmentedColormap.from_list('seq', ['#f3f0f7', '#bcb3d8', '#7b6bb0', '#4a2f86', '#25104f'])
C_EMU, C_NUM = '#eb6834', '#2a78d6'
LAB = {'emu': 'TOMAS emulator', 'num': 'TOMAS numerical'}
INJ = dict(lon=[180, 180], lat=[29.7, -29.7])
os.makedirs(f'{OUT}/figs', exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9.5})

def sym(*xs, q=99.0):
    v = np.concatenate([np.abs(x[np.isfinite(x)]).ravel() for x in xs]); l = np.percentile(v, q)
    return float(l) if l > 0 else 1.0

def mapax(fig, spec):
    ax = fig.add_subplot(spec, projection=ccrs.PlateCarree(central_longitude=180))
    ax.coastlines(linewidth=0.4, color='0.3'); ax.set_global()
    ax.plot(INJ['lon'], INJ['lat'], 'k+', ms=7, mew=1.2, transform=ccrs.PlateCarree(), ls='none')
    return ax

def pmesh(ax, x, cmap, norm, title, fontsize=None):
    im = ax.pcolormesh(lon, lat, x, cmap=cmap, norm=norm, shading='auto', transform=ccrs.PlateCarree(), rasterized=True)
    ax.set_title(title, fontsize=fontsize); return im

def triple_row(fig, gs, row, e, n, unit, name, diverging=True, lim=None, seqlim=None, dlim=None, gmfmt='{:+.3g}', log=False, rel=False):
    """emulator | numerical | emulator-minus-numerical."""
    axs = [mapax(fig, gs[row, j]) for j in range(3)]
    if diverging:
        L = lim or sym(e, n); norm = mcolors.Normalize(-L, L); cm = DIV; ext = 'both'
    else:
        lo, hi = seqlim or (np.nanpercentile(np.concatenate([e.ravel(), n.ravel()]), 1), np.nanpercentile(np.concatenate([e.ravel(), n.ravel()]), 99))
        norm = mcolors.LogNorm(lo, hi) if log else mcolors.Normalize(lo, hi); cm = SEQ; ext = 'both'
    for ax, x, k in ((axs[0], e, 'emu'), (axs[1], n, 'num')):
        im = pmesh(ax, x, cm, norm, f'{LAB[k]}: {name}\nglobal mean ' + gmfmt.format(gm(x)) + f' {unit}', fontsize=8.5)
    cb = fig.colorbar(im, ax=axs[1], shrink=0.85, extend=ext, pad=0.02); cb.set_label(unit)
    d = e - n
    if rel:
        d = 100 * d / n; du = '%'
    else:
        du = unit
    Ld = dlim or sym(d)
    im2 = pmesh(axs[2], d, DIV, mcolors.Normalize(-Ld, Ld),
                f'emulator minus numerical\nglobal mean ' + (f'{gm(d):+.2g} %' if rel else gmfmt.format(gm(d)) + f' {unit}  ({100*gm(d)/gm(n):+.1f} %)'))
    cb2 = fig.colorbar(im2, ax=axs[2], shrink=0.85, extend='both', pad=0.02); cb2.set_label(du)
    return axs

def save(fig, name):
    fig.savefig(f'{OUT}/figs/{name}', dpi=150, bbox_inches='tight'); plt.close(fig); print('wrote', name)

numbers = {}
# ---------------- 0. tropopause ----------------
pt = A['ann_ptrop'] / 100; sf = A['ann_strat_frac_cells']
fig = plt.figure(figsize=(13, 3.8), layout='constrained'); gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1])
ax = mapax(fig, gs[0]); im = pmesh(ax, pt, SEQ.reversed(), mcolors.Normalize(80, 350), 'Annual-mean WMO lapse-rate tropopause pressure (emulator T)')
fig.colorbar(im, ax=ax, shrink=0.85, extend='both').set_label('hPa')
ax2 = fig.add_subplot(gs[1]); excl = 1 - sf.mean(1)
ax2.plot(100 * excl, lat, color='0.2', lw=1.5); ax2.set_xlabel('band cells excluded as tropospheric, % (annual mean, zonal)')
ax2.set_ylabel('latitude'); ax2.set_ylim(-90, 90); ax2.grid(alpha=0.3)
ax2.set_title(f'excluded: {100*(1-sf.mean()):.2f} % of band cells, {100*(1-gm(sf)):.2f} % area-weighted')
save(fig, 'fig0_tropopause.png')
numbers['excluded_cells_pct'] = 100 * (1 - sf.mean()); numbers['excluded_area_pct'] = 100 * (1 - gm(sf))
numbers['excluded_daily_range_pct'] = (100 * (1 - TS['strat_frac'].max()), 100 * (1 - TS['strat_frac'].min()))
numbers['trop_nofound_pct'] = 100 * TS['nofound'].mean()
for a, b in ((-10, 10), (30, 40), (60, 90), (-90, -60)):
    s = (lat >= a) & (lat < b); numbers[f'ptrop_{a}_{b}_hPa'] = float(np.median(pt[s]))

# ---------------- 1. forcing maps ----------------
for per, plab in (('ann', 'annual mean'), ('l30', 'last 30 days (days 336-365)')):
    fig = plt.figure(figsize=(16, 9.6), layout='constrained'); gs = fig.add_gridspec(3, 3)
    e, n = A[f'{per}_emu_Fnet_fix'], A[f'{per}_num_Fnet_fix']
    triple_row(fig, gs, 0, e, n, 'W m$^{-2}$', 'net TOA SAI forcing')
    triple_row(fig, gs, 1, A[f'{per}_emu_Fsw_fix'], A[f'{per}_num_Fsw_fix'], 'W m$^{-2}$', 'SW forcing (daily-mean insol.)')
    triple_row(fig, gs, 2, A[f'{per}_emu_Flw'], A[f'{per}_num_Flw'], 'W m$^{-2}$', 'LW forcing')
    fig.suptitle(f'TOA SAI forcing F = (SW$_{{up}}$+OLR)$_{{control}}$ - (SW$_{{up}}$+OLR)$_{{injection}}$, {plab}; negative = cooling. '
                 'SW converted to a daily-mean estimate (insolation-weighted albedo change x daily-mean insolation); net = that SW + LW.', fontsize=10)
    save(fig, f'fig1_{"a" if per=="ann" else "b"}_forcing_maps_{per}.png')
    for k in ('emu', 'num'):
        for v in ('Fnet_fix', 'Fnet_raw', 'Fsw_fix', 'Fsw_raw', 'Flw', 'dAOD'):
            numbers[f'{per}_{k}_{v}_gm'] = gm(A[f'{per}_{k}_{v}'])
fig = plt.figure(figsize=(16, 3.6), layout='constrained'); gs = fig.add_gridspec(1, 3)
triple_row(fig, gs, 0, A['ann_emu_Fsw_raw'], A['ann_num_Fsw_raw'], 'W m$^{-2}$', 'SW forcing, RAW frame mean')
fig.suptitle('Raw annual-mean SW forcing (mean of the 4 six-hourly frames, fixed UTC times): each longitude is always sampled at the same 4 local times, '
             'giving the daylight lobes. For transparency only; use fig1a/b.', fontsize=9.5)
save(fig, 'fig1_c_forcing_sw_raw_annual.png')

# ---------------- 2. time series ----------------
day = np.arange(1, 366)
def run30(x):
    k = np.ones(30) / 30; y = np.convolve(x, k, 'same'); c = np.convolve(np.ones_like(x), k, 'same'); return y / c
fig, axs = plt.subplots(4, 1, figsize=(10, 10.5), sharex=True)
series = (('Fsw_raw', 'SW'), ('Flw', 'LW'), ('Fnet_raw', 'net'))
for ax, (v, nm) in zip(axs[:3], series):
    for k, c in (('num', C_NUM), ('emu', C_EMU)):
        x = TS[f'{k}_{v}']; ax.plot(day, x, color=c, lw=0.6, alpha=0.45)
        ax.plot(day, run30(x), color=c, lw=2, label=f'{LAB[k]} (30-day running mean)')
    ax.axhline(0, color='0.5', lw=0.6); ax.set_ylabel(f'{nm} forcing, W m$^{{-2}}$'); ax.grid(alpha=0.25)
    ax.set_title(f'Global-mean {nm} TOA SAI forcing (thin: daily, thick: 30-day running mean)', loc='left')
axs[0].legend(loc='lower left', frameon=False)
for (v, nm), ls in zip(series, ('-', '--', '-')):
    e, n = run30(TS[f'emu_{v}']), run30(TS[f'num_{v}'])
    pct = 100 * (e - n) / n; pct[:30] = np.nan
    axs[3].plot(day, pct, color={'SW': '#2a78d6', 'LW': '#c2452f', 'net': '0.15'}[nm], lw=1.6, ls=ls, label=nm)
axs[3].axhline(0, color='0.5', lw=0.6); axs[3].set_ylabel('emulator minus numerical, %'); axs[3].grid(alpha=0.25)
axs[3].legend(frameon=False, ncol=3); axs[3].set_xlabel('day of 1997')
axs[3].set_title('Emulator minus numerical, % of numerical (30-day running means; first 30 days omitted, forcing ~0)', loc='left')
fig.tight_layout(); save(fig, 'fig2_forcing_timeseries.png')
for k in ('emu', 'num'):
    for v in ('Fsw_raw', 'Fsw_fix', 'Flw', 'Fnet_raw', 'Fnet_fix'):
        numbers[f'ts_{k}_{v}_ann'] = float(TS[f'{k}_{v}'].mean()); numbers[f'ts_{k}_{v}_l30'] = float(TS[f'{k}_{v}'][-30:].mean())
# cross-check against the dashboard csv
try:
    import csv
    rows = list(csv.reader(open(os.environ['FORCING_CSV'])))
    hdr = rows[0]; R = {}
    for r in rows[1:]:
        pair = ','.join(r[:len(r) - len(hdr) + 1]); R.setdefault(pair, []).append([float(x) for x in r[len(r) - len(hdr) + 1:]])
    for p, v in R.items():
        v = np.array(v); numbers[f'csv[{p}] n={len(v)} net mean'] = float(v[:, 1].mean())
except Exception as ex:
    numbers['csv_error'] = str(ex)

# ---------------- 3. heating ----------------
def latp(ax, x, norm, title):
    im = ax.pcolormesh(lat, plev, x, cmap=DIV, norm=norm, shading='nearest', rasterized=True)
    ax.set_yscale('log'); ax.set_ylim(plev.max() * 1.05, plev.min()); ax.set_yticks([1, 2, 5, 10, 20, 50, 100, 140])
    ax.set_yticklabels(['1', '2', '5', '10', '20', '50', '100', '140']); ax.set_ylabel('pressure, hPa'); ax.set_xlabel('latitude')
    ax.set_title(title); return im
e, n = A['ann_emu_dhr_kday_zm'], A['ann_num_dhr_kday_zm']
fig, axs = plt.subplots(1, 3, figsize=(16, 4.3))
L = sym(e, n, q=99.5); nrm = mcolors.Normalize(-L, L)
latp(axs[0], e, nrm, f'{LAB["emu"]}: aerosol heating rate'); im = latp(axs[1], n, nrm, f'{LAB["num"]}: aerosol heating rate')
fig.colorbar(im, ax=axs[:2], extend='both', pad=0.02).set_label('K day$^{-1}$')
Ld = sym(e - n, q=99.5); im = latp(axs[2], e - n, mcolors.Normalize(-Ld, Ld), 'emulator minus numerical')
fig.colorbar(im, ax=axs[2], extend='both', pad=0.02).set_label('K day$^{-1}$')
fig.suptitle('Stratospheric aerosol heating rate hr$_{inj}$ - hr$_{ctl}$, zonal mean over stratospheric cells only, annual mean', fontsize=10)
save(fig, 'fig3_a_heating_latp_annual.png')
numbers['dhr_zm_max_emu'] = float(np.nanmax(e)); numbers['dhr_zm_max_num'] = float(np.nanmax(n))
ij = np.unravel_index(np.nanargmax(n), n.shape); numbers['dhr_zm_max_num_at'] = (float(plev[ij[0]]), float(lat[ij[1]]))
# monthly gif
mon = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
me, mn = A['mon_emu_dhr_kday_zm'], A['mon_num_dhr_kday_zm']
L = sym(me, mn, q=99.5); Ld = sym(me - mn, q=99.5); frames = []
for m in range(12):
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.9))
    latp(axs[0], me[m], mcolors.Normalize(-L, L), f'{LAB["emu"]}'); im = latp(axs[1], mn[m], mcolors.Normalize(-L, L), f'{LAB["num"]}')
    fig.colorbar(im, ax=axs[:2], extend='both', pad=0.02).set_label('K day$^{-1}$')
    im = latp(axs[2], me[m] - mn[m], mcolors.Normalize(-Ld, Ld), 'emulator minus numerical')
    fig.colorbar(im, ax=axs[2], extend='both', pad=0.02).set_label('K day$^{-1}$')
    fig.suptitle(f'Stratospheric aerosol heating rate (injection minus control), zonal mean, {mon[m]} 1997 monthly mean')
    b = io.BytesIO(); fig.savefig(b, dpi=90, bbox_inches='tight'); plt.close(fig); frames.append(Image.open(b).convert('RGB'))
frames = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames]
frames[0].save(f'{OUT}/figs/fig3_b_heating_latp_monthly.gif', save_all=True, append_images=frames[1:], duration=900, loop=0, optimize=True)
print('wrote gif 3b')
e, n = A['ann_emu_dhr_kday_50'], A['ann_num_dhr_kday_50']
fig = plt.figure(figsize=(16, 3.6), layout='constrained'); gs = fig.add_gridspec(1, 3)
triple_row(fig, gs, 0, e, n, 'K day$^{-1}$', f'heating rate at {plev[17]:.1f} hPa', lim=sym(e, n, q=99.5))
fig.suptitle(f'Aerosol heating rate hr$_{{inj}}$ - hr$_{{ctl}}$ at the band level nearest 50 hPa ({plev[17]:.1f} hPa, the injection level), annual mean', fontsize=10)
save(fig, 'fig3_c_heating_map_50hPa_annual.png')

# ---------------- 4. size statistics ----------------
SZ = (('dg', 'number geom.-mean diameter D$_g$', 'nm', True, (5, 200)),
      ('sig', 'geometric standard deviation $\\sigma_g$', '-', False, None),
      ('dmass', 'mass-weighted geom.-mean diam.', 'nm', False, None),
      ('deff_dry', 'effective diameter, dry', 'nm', False, None),
      ('deff_wet', 'effective diam., wet (Tabazadeh)', 'nm', False, None))
for per, plab in (('ann', 'annual mean'), ('l30', 'last 30 days')):
    fig = plt.figure(figsize=(16, 15.5), layout='constrained'); gs = fig.add_gridspec(5, 3)
    for r, (k, nm, unit, lg, sl) in enumerate(SZ):
        e, n = A[f'{per}_emu_{k}'], A[f'{per}_num_{k}']
        triple_row(fig, gs, r, e, n, unit, nm, diverging=False, log=lg, seqlim=None, gmfmt='{:.3g}', rel=True)
        for kk in ('emu', 'num'):
            numbers[f'{per}_{kk}_{k}_gm'] = gm(A[f'{per}_{kk}_{k}'])
    fig.suptitle(f'Stratospheric column size statistics of the INJECTION runs, {plab} of per-frame column values '
                 '(column number/mass per bin over stratospheric levels; lognormal by moments). Difference = (emulator - numerical)/numerical, %.\n'
                 'Caveat: columns are bimodal (nucleation + aged mode); the single-lognormal D$_g$/$\\sigma_g$ is a compromise - read with the mass-weighted and effective diameters.', fontsize=10)
    save(fig, f'fig4_{"a" if per=="ann" else "b"}_size_maps_{per}.png')
    fig = plt.figure(figsize=(16, 6.6), layout='constrained'); gs = fig.add_gridspec(2, 3)
    triple_row(fig, gs, 0, A[f'{per}_emu_dM'], A[f'{per}_num_dM'], 'mg m$^{-2}$', 'column SO$_4$ mass, SAI - control', lim=sym(A[f'{per}_emu_dM'], A[f'{per}_num_dM'], q=99.5))
    triple_row(fig, gs, 1, A[f'{per}_emu_dN'] / 1e10, A[f'{per}_num_dN'] / 1e10, '10$^{10}$ m$^{-2}$', 'column number, SAI - control', lim=sym(A[f'{per}_emu_dN'], A[f'{per}_num_dN'], q=99.5) / 1e10)
    fig.suptitle(f'Stratospheric column sulfate mass and particle number, injection minus control, {plab}', fontsize=10)
    save(fig, f'fig4_{"c" if per=="ann" else "d"}_column_NM_sai_minus_ctl_{per}.png')
    for kk in ('emu', 'num'):
        numbers[f'{per}_{kk}_dM_gm_mg'] = gm(A[f'{per}_{kk}_dM']); numbers[f'{per}_{kk}_dN_gm_1e10'] = gm(A[f'{per}_{kk}_dN']) / 1e10
# global column size distribution (bimodality)
dpb = np.load(f'{RUNS}/r2_tomas_fast/frames3d/step_0000.npz')['dp_bin_nm']; dlnD = np.log(dpb[1] / dpb[0])
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
for k, c in (('num', C_NUM), ('emu', C_EMU)):
    for ax, v, nm, u in ((axs[0], 'Nk_gm', 'number', 'm$^{-2}$'), (axs[1], 'Mk_gm', 'mass', 'kg m$^{-2}$')):
        for per, ls in (('ann', '-'), ('l30', '--')):
            x = A[f'{per}_{k}_{v}'] / (w.sum() * 288) / dlnD
            ax.loglog(dpb, x, color=c, ls=ls, lw=1.8, label=f'{LAB[k]}, {"annual" if per=="ann" else "last 30 d"}')
        ax.set_xlabel('dry diameter, nm'); ax.set_ylabel(f'd{nm}/dlnD, {u}'); ax.grid(alpha=0.25, which='both')
        ax.set_title(f'global-mean stratospheric column {nm} distribution, injection run')
axs[0].set_ylim(bottom=1e7); axs[1].set_ylim(bottom=1e-9); axs[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); save(fig, 'fig4_e_column_size_distribution.png')

# ---------------- 5. tracer gif ----------------
G5 = np.load(f'{CACHE}/daily_gif.npz'); e5, n5 = G5['emu_dM'], G5['num_dM']
L = sym(e5[-60:], n5[-60:], q=99.5); nrm = mcolors.Normalize(-L, L); frames = []
STEP = int(os.environ.get('GIF5_STEP', '2'))
for d in range(0, 365, STEP):
    fig = plt.figure(figsize=(12, 3.4)); gs = fig.add_gridspec(1, 2)
    for j, (x, k) in enumerate(((e5[d], 'emu'), (n5[d], 'num'))):
        ax = mapax(fig, gs[j]); im = pmesh(ax, x, DIV, nrm, f'{LAB[k]}   global mean {gm(x):.3g} mg m$^{{-2}}$')
    fig.colorbar(im, ax=fig.axes, shrink=0.9, extend='both', pad=0.015).set_label('mg SO$_4$ m$^{-2}$')
    fig.suptitle(f'Stratospheric column sulfate, injection minus control, day {d+1} of 1997 (daily mean)', fontsize=10)
    b = io.BytesIO(); fig.savefig(b, dpi=int(os.environ.get('GIF5_DPI', '72')), bbox_inches='tight'); plt.close(fig)
    frames.append(Image.open(b).convert('RGB'))
pal = frames[-1].quantize(colors=200, method=Image.Quantize.MAXCOVERAGE)
frames = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
frames[0].save(f'{OUT}/figs/fig5_column_sulfate_transport.gif', save_all=True, append_images=frames[1:], duration=120, loop=0, optimize=True)
print('wrote gif 5', os.path.getsize(f'{OUT}/figs/fig5_column_sulfate_transport.gif') / 1e6, 'MB')
import json
json.dump(numbers, open(f'{OUT}/cache/numbers.json', 'w'), indent=1, default=float)
for k, v in numbers.items(): print(k, v)
