"""Extra coupled figures (2026-11-09), PYC (cartopy). Follows LP_NDAYS (365 = year 1, 730 = 1997-1998).
  fig0_tropopause.png                     tropopause map + zonal profile with a DOUBLE x axis
                                          (bottom: zonal-mean tropopause pressure, top: % of band cells excluded)
  fig5_b_column_sulfate_num_diffs.gif     TOMAS numerical column sulfate (inj - ctl) | emulator - numerical | MAM4 - numerical
  fig6_aod550_map_timeseries.gif          added AOD550 maps (numerical, emu - num, MAM4 - num) side by side with an animated
                                          line plot of the global-mean and latitude-band added AOD (seasonal cycle)
Inputs: cache/agg_v2.npz, cache/daily_gif_v2.npz, cache/daily_aod_v2.npz (agg2.py).
usage: plots_extra.py [fig0] [fig5b] [fig6]   (default: all)"""
import os, io, sys, datetime
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.ticker
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from PIL import Image
from common import *

WHICH = set(sys.argv[1:]) or {'fig0', 'fig5b', 'fig6'}
A = dict(np.load(f'{CACHE}/agg_v2.npz'))
lat = A['lat']; lon = np.arange(288) * 1.25
w = np.cos(np.deg2rad(lat)); W2 = w[:, None] / (w.sum() * 288)
gm = lambda x: float(np.nansum(x * W2) / np.nansum(W2 * np.isfinite(x)))
DIV = mcolors.LinearSegmentedColormap.from_list('bgr', ['#0b3c78', '#2f6fb3', '#8fb6dd', '#e3e3e1', '#eba08a', '#c2452f', '#6d1209'])
SEQ = mcolors.LinearSegmentedColormap.from_list('seq', ['#f3f0f7', '#bcb3d8', '#7b6bb0', '#4a2f86', '#25104f'])
COL = {'emu': '#eb6834', 'num': '#2a78d6', 'mam': '#1baf7a'}
LAB = {'emu': 'TOMAS emulator', 'num': 'TOMAS numerical', 'mam': 'MAM4 (modal, binned)'}
FIG = f'{OUT}/figs_v2'; os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9})
PER = 'annual mean' if NDAYS == 365 else f'{NDAYS // 365}-year mean (1997-{1996 + NDAYS // 365})'
DSTEP = int(os.environ.get('GIF_DSTEP', '2' if NDAYS <= 365 else '4'))
D0 = datetime.date(1997, 1, 1)                     # 1997 and 1998 are not leap years: day index d -> D0 + d

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

def shared_palette(frames, n=200, k=8):
    idx = np.linspace(0, len(frames) - 1, min(k, len(frames))).astype(int)
    W = max(frames[i].size[0] for i in idx); H = sum(frames[i].size[1] for i in idx)
    canvas = Image.new('RGB', (W, H), 'white'); y = 0
    for i in idx:
        canvas.paste(frames[i], (0, y)); y += frames[i].size[1]
    return canvas.quantize(colors=n, method=Image.Quantize.MAXCOVERAGE)

def grab(fig, dpi):
    b = io.BytesIO(); fig.savefig(b, dpi=dpi); return Image.open(b).convert('RGB')

def write_gif(frames, path, ms):
    pal = shared_palette(frames)
    frames = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0, optimize=False)
    print('wrote', os.path.basename(path), f'{len(frames)} frames, {os.path.getsize(path) / 1e6:.1f} MB')

datestr = lambda d: (D0 + datetime.timedelta(days=int(d))).strftime('%d %b %Y')

# ---------------- fig0: tropopause, double x axis ----------------
if 'fig0' in WHICH:
    Z = np.load(f'{CACHE}/daily_aod_v2.npz')
    pt = A['ann_ptrop'] / 100; sf = A['ann_strat_frac_cells']
    ptz = np.nanmean(Z['ptrop_zm'], 0) / 100                      # zonal-mean tropopause pressure, hPa (time mean of daily zonal means)
    fig = plt.figure(figsize=(13, 3.8), layout='constrained'); gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1])
    ax = mapax(fig, gs[0]); im = pmesh(ax, pt, SEQ.reversed(), mcolors.Normalize(80, 350), f'{PER[0].upper() + PER[1:]} WMO lapse-rate tropopause pressure (emulator T)')
    fig.colorbar(im, ax=ax, shrink=0.85, extend='both').set_label('hPa')
    ax2 = fig.add_subplot(gs[1]); excl = 100 * (1 - sf.mean(1))
    c1, c2 = '0.15', '#7b6bb0'
    ax2.plot(ptz, lat, color=c2, lw=1.8, label='tropopause pressure (bottom axis)')
    ax2.set_xscale('log'); ax2.set_xlim(500, 70); ax2.set_xticks([500, 300, 200, 150, 100, 70]); ax2.set_xticklabels(['500', '300', '200', '150', '100', '70']); ax2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.set_xlabel('zonal-mean tropopause pressure, hPa (log; higher tropopause to the right)', color=c2); ax2.tick_params(axis='x', colors=c2)
    ax2.set_ylabel('latitude'); ax2.set_ylim(-90, 90); ax2.grid(alpha=0.3)
    ax3 = ax2.twiny()
    ax3.plot(excl, lat, color=c1, lw=1.5, label='% excluded (top axis)')
    h2, l2 = ax2.get_legend_handles_labels(); h3, l3 = ax3.get_legend_handles_labels()
    ax3.legend(h2 + h3, l2 + l3, loc='lower right', frameon=False, fontsize=8)
    ax3.set_xlabel(f'band cells excluded as tropospheric, % ({PER}, zonal)', color=c1)
    ax3.axvline(0, color='0.7', lw=0.5)
    ax2.set_title(f'excluded: {100*(1-sf.mean()):.2f} % of band cells, {100*(1-gm(sf)):.2f} % area-weighted   '
                  f'(band bottom {143:.0f} hPa)', fontsize=8.5)
    fig.savefig(f'{FIG}/fig0_tropopause.png', dpi=150, bbox_inches='tight'); plt.close(fig); print('wrote fig0_tropopause.png')

# ---------------- fig5b: numerical + differences ----------------
if 'fig5b' in WHICH:
    G = np.load(f'{CACHE}/daily_gif_v2.npz'); E = {k: G[f'{k}_dM'] for k in ('emu', 'num', 'mam')}
    nd = E['num'].shape[0]; days = list(range(0, nd, DSTEP))
    L = sym(E['num'][-60:], q=99.5)                              # same convention as fig5: 99.5th pct of the last 60 days
    Ld = {k: sym(E[k][-60:] - E['num'][-60:], q=99.5) for k in ('emu', 'mam')}
    fig = plt.figure(figsize=(16, 3.6)); gs = fig.add_gridspec(1, 3, wspace=0.12)
    axs = [mapax(fig, gs[j]) for j in range(3)]
    ims = [pmesh(axs[0], E['num'][0], SEQ, mcolors.Normalize(0, L), '')]    # sequential: the reference field is (almost) all positive
    fig.colorbar(ims[0], ax=axs[0], orientation='horizontal', shrink=0.8, pad=0.04, extend='both').set_label('mg SO$_4$ m$^{-2}$')
    for j, k in ((1, 'emu'), (2, 'mam')):
        ims.append(pmesh(axs[j], E[k][0] - E['num'][0], DIV, mcolors.Normalize(-Ld[k], Ld[k]), ''))
        fig.colorbar(ims[j], ax=axs[j], orientation='horizontal', shrink=0.8, pad=0.04, extend='both').set_label('mg SO$_4$ m$^{-2}$ (own scale, fixed over time)')
    sup = fig.suptitle('', fontsize=10); frames = []
    for d in days:
        n = E['num'][d]; ims[0].set_array(n.ravel())
        axs[0].set_title(f'{LAB["num"]}: injection minus control\nglobal mean {gm(n):.3g} mg m$^{{-2}}$', fontsize=8.5)
        for j, k in ((1, 'emu'), (2, 'mam')):
            x = E[k][d] - n; ims[j].set_array(x.ravel())
            axs[j].set_title(f'{"TOMAS emulator" if k == "emu" else "MAM4"} minus numerical\nglobal mean {gm(x):+.3g} mg m$^{{-2}}$ ({100 * gm(x) / gm(n):+.1f} %)', fontsize=8.5)
        sup.set_text(f'Stratospheric column sulfate, daily mean, {datestr(d)} (day {d + 1})')
        frames.append(grab(fig, int(os.environ.get('GIF5_DPI', '80'))))
    plt.close(fig)
    write_gif(frames, f'{FIG}/fig5_b_column_sulfate_num_diffs.gif', 120)

# ---------------- fig6: AOD maps + animated time series ----------------
if 'fig6' in WHICH:
    Z = np.load(f'{CACHE}/daily_aod_v2.npz'); E = {k: Z[f'{k}_dAOD'] for k in ('emu', 'num', 'mam')}
    nd = E['num'].shape[0]; days = list(range(0, nd, DSTEP)); t = np.arange(nd)
    gmt = {k: (E[k] * W2[None]).sum((1, 2)) for k in E}
    bands = (('20N-90N', lat >= 20), ('20S-20N', np.abs(lat) < 20), ('90S-20S', lat <= -20))
    def band_mean(x, m):
        ww = np.where(m, w, 0.0); return (x.mean(2) * ww).sum(1) / ww.sum()
    bm = {nm: {k: band_mean(E[k], m) for k in E} for nm, m in bands}
    L = float(np.percentile(E['num'][-60:], 99.5)); Ld = {k: sym(E[k][-60:] - E['num'][-60:], q=99.5) for k in ('emu', 'mam')}
    fig = plt.figure(figsize=(15, 8.6)); gs = fig.add_gridspec(3, 2, width_ratios=[1.3, 1], hspace=0.32, wspace=0.16,
                                                           left=0.02, right=0.97, top=0.92, bottom=0.06)
    mx = [mapax(fig, gs[i, 0]) for i in range(3)]
    ims = [pmesh(mx[0], E['num'][0], SEQ, mcolors.Normalize(0, L), '')]
    fig.colorbar(ims[0], ax=mx[0], shrink=0.85, pad=0.02, extend='max').set_label('added AOD 550 nm')
    for i, k in ((1, 'emu'), (2, 'mam')):
        ims.append(pmesh(mx[i], E[k][0] - E['num'][0], DIV, mcolors.Normalize(-Ld[k], Ld[k]), ''))
        fig.colorbar(ims[i], ax=mx[i], shrink=0.85, pad=0.02, extend='both').set_label('AOD difference (own scale)')
    a1 = fig.add_subplot(gs[0:2, 1]); a2 = fig.add_subplot(gs[2, 1], sharex=a1)
    xd = [D0 + datetime.timedelta(days=int(i)) for i in t]
    ln1, dot1 = {}, {}
    for k in ('num', 'emu', 'mam'):
        a1.plot(xd, gmt[k], color=COL[k], lw=0.8, alpha=0.18)                  # full series, faint (shows where it is going)
        ln1[k], = a1.plot([], [], color=COL[k], lw=2, label=LAB[k]); dot1[k], = a1.plot([], [], 'o', color=COL[k], ms=5)
    a1.set_ylim(0, 1.08 * max(g.max() for g in gmt.values())); a1.set_ylabel('global-mean added AOD 550 nm'); a1.grid(alpha=0.25)
    a1.legend(loc='upper left', frameon=False); a1.set_title('Global mean (injection minus control), daily', loc='left')
    BC = {'20N-90N': '#c2452f', '20S-20N': '0.25', '90S-20S': '#2f6fb3'}
    ln2 = {}
    for nm, _ in bands:
        for k, ls in (('num', '-'), ('emu', '--'), ('mam', ':')):
            a2.plot(xd, bm[nm][k], color=BC[nm], lw=0.6, alpha=0.15, ls=ls)
            ln2[(nm, k)], = a2.plot([], [], color=BC[nm], lw=1.6 if k == 'num' else 1.1, ls=ls,
                                    label=f'{nm}' if k == 'num' else None)
    a2.plot([], [], color='0.4', ls='-', label='numerical'); a2.plot([], [], color='0.4', ls='--', label='emulator'); a2.plot([], [], color='0.4', ls=':', label='MAM4')
    a2.set_ylim(0, 1.08 * max(b.max() for bb in bm.values() for b in bb.values())); a2.grid(alpha=0.25)
    a2.set_ylabel('band-mean added AOD'); a2.legend(loc='upper left', frameon=False, ncol=2, fontsize=7.5)
    a2.set_title('Latitude bands (area-weighted): seasonal cycle of the hemispheric spread', loc='left')
    import matplotlib.dates as mdates
    a2.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y' if nd > 365 else '%b')); a1.tick_params(labelbottom=False)
    cur = [a.axvline(xd[0], color='0.5', lw=0.8) for a in (a1, a2)]
    sup = fig.suptitle('', fontsize=10.5); frames = []
    for d in days:
        n = E['num'][d]; ims[0].set_array(n.ravel())
        mx[0].set_title(f'{LAB["num"]}: added AOD 550 nm, global mean {gmt["num"][d]:.4f}', fontsize=8.5)
        for i, k in ((1, 'emu'), (2, 'mam')):
            x = E[k][d] - n; ims[i].set_array(x.ravel())
            mx[i].set_title(f'{"TOMAS emulator" if k == "emu" else "MAM4"} minus numerical, global mean {gm(x):+.4f} ({100 * gm(x) / max(gm(n), 1e-12):+.1f} %)', fontsize=8.5)
        for k in ln1:
            ln1[k].set_data(xd[:d + 1], gmt[k][:d + 1]); dot1[k].set_data([xd[d]], [gmt[k][d]])
        for (nm, k), l in ln2.items():
            l.set_data(xd[:d + 1], bm[nm][k][:d + 1])
        for c in cur:
            c.set_xdata([xd[d], xd[d]])
        sup.set_text(f'Added stratospheric AOD at 550 nm (injection minus control, daily mean), {datestr(d)}   (day {d + 1} of {nd})')
        frames.append(grab(fig, int(os.environ.get('GIF6_DPI', '72'))))
    plt.close(fig)
    write_gif(frames, f'{FIG}/fig6_aod550_map_timeseries.gif', 120)
    # last frame also as PNG (full resolution) for slides
    print('AOD numbers: ' + ', '.join(f'{k} mean {gmt[k].mean():.4f} last30 {gmt[k][-30:].mean():.4f}' for k in gmt))

# ---------------- fig6b: WEEKLY means. AOD maps numerical | MAM4 | MAM4 - num + animated global-mean net TOA SAI forcing ----------------
# (added 2026-11-09 on request; run as `plots_extra.py fig6b`, not part of the default set)
if 'fig6b' in WHICH:
    Z = np.load(f'{CACHE}/daily_aod_v2.npz'); TS = np.load(f'{CACHE}/timeseries_v2.npz')
    nd = Z['num_dAOD'].shape[0]; nw = nd // 7                        # complete 7-day weeks (730 d -> 104 weeks; last 2 days dropped)
    wk = lambda x: x[:nw * 7].reshape((nw, 7) + x.shape[1:]).mean(1)
    E = {k: wk(Z[f'{k}_dAOD']) for k in ('num', 'mam')}
    F = {k: wk(TS[f'{k}_Fnet_raw']) for k in ('num', 'mam')}         # global-mean net TOA SAI forcing (all-frame daily mean)
    xd = [D0 + datetime.timedelta(days=7 * i + 3) for i in range(nw)] # week centre
    L = float(np.percentile(np.concatenate([E['num'][-9:], E['mam'][-9:]]), 99.5))   # shared scale: last ~60 days, both schemes
    AMIN = 0.005                                                      # relative difference only where numerical added AOD >= this
    rel = lambda m, n: np.where(n >= AMIN, 100 * (m - n) / np.where(n >= AMIN, n, 1), np.nan)
    Ld = sym(rel(E['mam'], E['num']), q=99.0)                        # % scale, fixed over all weeks
    VIR = plt.get_cmap('YlGnBu'); RB = plt.get_cmap('RdBu_r').copy(); RB.set_bad('0.85')
    fig = plt.figure(figsize=(15, 8.6)); gs = fig.add_gridspec(3, 2, width_ratios=[1.3, 1], hspace=0.32, wspace=0.16,
                                                           left=0.02, right=0.97, top=0.92, bottom=0.07)
    mx = [mapax(fig, gs[i, 0]) for i in range(3)]
    ims = [pmesh(mx[0], E['num'][0], VIR, mcolors.Normalize(0, L), ''), pmesh(mx[1], E['mam'][0], VIR, mcolors.Normalize(0, L), ''),
           pmesh(mx[2], rel(E['mam'][0], E['num'][0]), RB, mcolors.Normalize(-Ld, Ld), '')]
    for i in range(3):
        fig.colorbar(ims[i], ax=mx[i], shrink=0.85, pad=0.02, extend='both' if i == 2 else 'max').set_label(
            'relative difference, %' if i == 2 else 'added AOD 550 nm')
    a1 = fig.add_subplot(gs[:, 1])
    for k in ('num', 'mam'):
        a1.plot(xd, F[k], color=COL[k], lw=0.8, alpha=0.18)
    ln = {k: a1.plot([], [], color=COL[k], lw=2, label=LAB[k])[0] for k in ('num', 'mam')}
    dot = {k: a1.plot([], [], 'o', color=COL[k], ms=5)[0] for k in ('num', 'mam')}
    a1.set_ylim(1.08 * min(f.min() for f in F.values()), 0.15); a1.axhline(0, color='0.6', lw=0.6)
    a1.set_ylabel('global-mean net TOA SAI forcing, W m$^{-2}$'); a1.grid(alpha=0.25); a1.legend(loc='lower left', frameon=False)
    a1.set_title('Global-mean net TOA forcing (injection minus control), weekly mean', loc='left')
    import matplotlib.dates as mdates
    a1.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    cur = a1.axvline(xd[0], color='0.5', lw=0.8)
    sup = fig.suptitle('', fontsize=10.5); frames = []
    for i in range(nw):
        n, m = E['num'][i], E['mam'][i]
        ims[0].set_array(n.ravel()); ims[1].set_array(m.ravel()); ims[2].set_array(rel(m, n).ravel())
        mx[0].set_title(f'TOMAS numerical: added AOD 550 nm, global mean {gm(n):.4f}', fontsize=8.5)
        mx[1].set_title(f'MAM4: added AOD 550 nm, global mean {gm(m):.4f}', fontsize=8.5)
        mx[2].set_title(f'(MAM4 - TOMAS numerical) / TOMAS numerical; global means differ by {100 * gm(m - n) / max(gm(n), 1e-12):+.1f} %\n(grey: numerical added AOD < {AMIN:g})', fontsize=8.5)
        for k in ln:
            ln[k].set_data(xd[:i + 1], F[k][:i + 1]); dot[k].set_data([xd[i]], [F[k][i]])
        cur.set_xdata([xd[i], xd[i]])
        d0 = D0 + datetime.timedelta(days=7 * i); d1 = d0 + datetime.timedelta(days=6)
        sup.set_text(f'Added stratospheric AOD at 550 nm (injection minus control), weekly mean {d0:%d %b} - {d1:%d %b %Y}   (week {i + 1} of {nw})')
        frames.append(grab(fig, int(os.environ.get('GIF6_DPI', '72'))))
    plt.close(fig)
    write_gif(frames, f'{FIG}/fig6_b_aod550_num_mam_forcing_weekly.gif', 250)
    print(f'fig6b: weeks {nw}; last week AOD num {gm(E["num"][-1]):.4f} mam {gm(E["mam"][-1]):.4f}; '
          f'net forcing last week num {F["num"][-1]:.3f} mam {F["mam"][-1]:.3f} W/m2; min num {F["num"].min():.3f} mam {F["mam"].min():.3f}')

# ---------------- fig6c: WEEKLY means. TOMAS emulator added AOD, single map, no line plot ----------------
# (added 2026-11-10 on request; run as `plots_extra.py fig6c`)
if 'fig6c' in WHICH:
    Z = np.load(f'{CACHE}/daily_aod_v2.npz')
    nd = Z['emu_dAOD'].shape[0]; nw = nd // 7
    e = Z['emu_dAOD'][:nw * 7].reshape((nw, 7) + Z['emu_dAOD'].shape[1:]).mean(1)
    L = float(np.percentile(e[-9:], 99.5))                           # fixed scale: last ~60 days
    fig = plt.figure(figsize=(10, 5.6)); gs = fig.add_gridspec(1, 1, left=0.03, right=0.97, top=0.88, bottom=0.04)
    ax = mapax(fig, gs[0, 0])
    im = pmesh(ax, e[0], plt.get_cmap('YlGnBu'), mcolors.Normalize(0, L), '')
    fig.colorbar(im, ax=ax, orientation='horizontal', shrink=0.6, pad=0.04, extend='max').set_label('added AOD 550 nm (injection minus control)')
    sup = fig.suptitle('', fontsize=11); frames = []
    for i in range(nw):
        im.set_array(e[i].ravel())
        d0 = D0 + datetime.timedelta(days=7 * i); d1 = d0 + datetime.timedelta(days=6)
        sup.set_text(f'TOMAS emulator: added stratospheric AOD at 550 nm, weekly mean {d0:%d %b} - {d1:%d %b %Y}\n'
                     f'week {i + 1} of {nw}, global mean {gm(e[i]):.4f}')
        frames.append(grab(fig, int(os.environ.get('GIF6_DPI', '90'))))
    plt.close(fig)
    write_gif(frames, f'{FIG}/fig6_c_aod550_emu_weekly.gif', 250)
    frames[-1].save(f'{FIG}/fig6_c_aod550_emu_weekly_lastframe.png')
    print(f'fig6c: weeks {nw}; scale 0-{L:.4f}; last week global mean {gm(e[-1]):.4f}; size {os.path.getsize(f"{FIG}/fig6_c_aod550_emu_weekly.gif") / 1e6:.1f} MB')
