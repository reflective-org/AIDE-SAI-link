#!/usr/bin/env python3
"""Figures for the radiation emulator A/B (rad_bench_<TAG>.npz written by
RAD_REF=rrtmgp) and the offline validation stored in the checkpoints.

    python3 scripts/radiation/plot_rad_bench.py TAG [--ckpt DIR] [--out NAME]

Writes outputs/rad_emulator/rad_bench_<NAME>.png: (a) per-step rms error of
the emulated heating anomaly vs the anomaly's own rms, (b) the global TOA
forcing of both engines, (c) wall time of both, (d) the per-level error of the
anomaly over the run, (e) the zonal-mean |error| map at the last compared
step, (f) the offline per-level error/signal of the two networks.
"""
import os, sys, argparse, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(); ap.add_argument('tag'); ap.add_argument('--ckpt', default=None); ap.add_argument('--out', default=None)
args = ap.parse_args()
z = np.load(f'{ROOT}/runs/{args.tag}/rad_bench_{args.tag}.npz', allow_pickle=True)
step = z['step']; day = (step + 1) * 6 / 24.0
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
a = ax[0, 0]
a.semilogy(day, z['anom_sig_rms'], 'k-', label='|anomaly| rms (physics)')
a.semilogy(day, z['anom_rms'], 'r-', label='emulator error rms')
a.semilogy(day, z['anom_max'], 'r:', label='emulator error max')
a.semilogy(day, z['hr1_rms'], 'b--', label='hr1 error rms')
a.set_xlabel('day'); a.set_ylabel('K/day'); a.set_title('heating anomaly: signal vs emulator error'); a.legend(fontsize=8); a.grid(alpha=.3)
a = ax[0, 1]
a.plot(day, z['arf_phys'], 'k-', label='physics'); a.plot(day, z['arf_emu'], 'r-', label='emulator')
a.set_xlabel('day'); a.set_ylabel('W m$^{-2}$'); a.set_title('instantaneous global TOA aerosol forcing'); a.legend(); a.grid(alpha=.3)
a = ax[0, 2]
a.plot(day, z['t_phys'], 'k-', label='physics (2 solves, cached ref)'); a.plot(day, z['t_emu'], 'r-', label='emulator')
a.set_xlabel('day'); a.set_ylabel('s per radiation step'); a.set_title(f'wall time: median physics {np.median(z["t_phys"]):.2f}s, emulator {np.median(z["t_emu"]):.2f}s'); a.legend(); a.grid(alpha=.3)
plev = z['plev_pa'] / 100
a = ax[1, 0]
le = np.sqrt((z['anom_lvl_rms']**2).mean(0)); ls = np.sqrt((z['anom_lvl_sig']**2).mean(0))
a.semilogx(ls, plev, 'k-', label='anomaly rms'); a.semilogx(le, plev, 'r-', label='error rms')
a.set_yscale('log'); a.invert_yaxis(); a.set_xlabel('K/day'); a.set_ylabel('hPa'); a.set_title('per level, run mean'); a.legend(); a.grid(alpha=.3)
a = ax[1, 1]
m = z['anom_zm_abs_err'][-1]; s = np.abs(z['anom_zm_sig'][-1])
im = a.pcolormesh(z['lat'], plev, m, cmap='magma_r', shading='auto'); a.set_yscale('log'); a.invert_yaxis()
a.set_title(f'zonal-mean |error| K/day, day {day[-1]:.0f} (max {m.max():.1e}; |signal| max {s.max():.1e})'); a.set_xlabel('lat'); a.set_ylabel('hPa'); plt.colorbar(im, ax=a)
a = ax[1, 2]
ck = args.ckpt or os.environ.get('RAD_EMU_CKPT', f'{ROOT}/inputs/rad_emulator/current')
for n, c in (('aer', 'r'), ('planck', 'b')):
    p = os.path.join(ck, f'{n}.npz')
    if os.path.exists(p):
        w = np.load(p); nzb = int(w['nzb']); e = w['level_err']; sg = w['level_sig']
        a.semilogx(sg[:nzb], plev, c + '--', label=f'{n} SW signal'); a.semilogx(e[:nzb], plev, c + ':', label=f'{n} SW err')
        a.semilogx(sg[nzb:2*nzb], plev, c + '-', label=f'{n} LW signal'); a.semilogx(e[nzb:2*nzb], plev, c + '-.', label=f'{n} LW err')
a.set_yscale('log'); a.invert_yaxis(); a.set_xlabel('K/day'); a.set_title('offline validation (held-out columns)'); a.legend(fontsize=7); a.grid(alpha=.3)
fig.suptitle(f'radiation emulator A/B: {args.tag} (anomaly reference = {z["anom_mode"]})')
fig.tight_layout()
out = f'{ROOT}/outputs/rad_emulator/rad_bench_{args.out or args.tag}.png'
os.makedirs(os.path.dirname(out), exist_ok=True); fig.savefig(out, dpi=110); print('wrote', out)
print(f'steps compared {len(step)}: anomaly err/signal rms (run mean) {np.sqrt((z["anom_rms"]**2).mean())/np.sqrt((z["anom_sig_rms"]**2).mean()):.1%}; '
      f'ARF emu-phys mean {np.mean(z["arf_emu"]-z["arf_phys"]):+.4f} W/m2 (phys mean {np.mean(z["arf_phys"]):+.4f}); '
      f'time emu {np.median(z["t_emu"]):.2f}s phys {np.median(z["t_phys"]):.2f}s')
