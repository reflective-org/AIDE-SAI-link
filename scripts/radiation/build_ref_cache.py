#!/usr/bin/env python3
"""Build the per-archive-hour radiation cache (src/radiation/rad_cache.py) for
a range of hours: the reference-aerosol solve (what RAD_MODE=anomaly needs,
bit-identical to the in-loop call), the clear-sky fields and their Planck
sensitivity (what RAD_ENGINE=emulator needs), and optionally the emulator's
own output on the reference aerosol (RAD_EMU_ANOM=net).

The driver and the reference aerosol are built EXACTLY as coupling.main()
builds them (same env knobs: AER_SRC, INIT_BIN, P_LO/P_HI, WET_OPTICS, CO2,
...), so the cache key matches the run's. Run it under the same environment
as the run, e.g. through run_prod.py's presets:

  CUDA_VISIBLE_DEVICES=4 LD_LIBRARY_PATH=... python3 scripts/radiation/build_ref_cache.py \
      --hours 0:721:6            # absolute h1 indices start:stop:step
      [--only ref|clear|all] [--emu-ckpt PATH]

~8.5 s per hour for all three solves + ~3 s of MAM4 binning, i.e. a 30-day
run's 121 hours in ~25 min, a year in ~5 h on one GPU. Entries already
complete are skipped, so it is safe to re-run or to run beside a model run
that is filling the same cache.
"""
import os, sys, time, argparse
import numpy as np
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, 'src'))
import _paths                      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--hours', required=True, help='start:stop:step of absolute h1 hour indices, or a comma list')
ap.add_argument('--only', default='all', choices=['ref', 'clear', 'all'])
ap.add_argument('--emu-ckpt', default=None, help='also store this emulator\'s output on the reference aerosol')
ap.add_argument('--cache', default=None, help='cache root (default: RAD_REF_CACHE / inputs/cache/rad_ref)')
args = ap.parse_args()

import coupling as C
import radiation as R
import rad_cache, rad_cache_build
import jax, jax.numpy as jnp

if ':' in args.hours:
    a, b, s = (int(x) for x in args.hours.split(':'))
    hours = list(range(a, b, s))
else:
    hours = [int(x) for x in args.hours.split(',')]

t0 = time.time()
dU = C.open_var('U'); lat = dU['lat'].values; lon = dU['lon'].values
plev = dU['hyam'].values * dU['P0'].values + dU['hybm'].values * C.PS_REF
band = np.where((plev >= C.P_LO_HPA * 100) & (plev <= C.P_HI_HPA * 100))[0]
full = list(range(band.min(), band.max() + 1))
if C.N_LEV and C.N_LEV < len(full):
    sel = np.unique(np.round(np.linspace(0, len(full) - 1, C.N_LEV)).astype(int))
    klevs = [full[i] for i in sel]
else:
    klevs = full
drv = R.RadiationDriver(C.open_var, klevs, lat, lon, C.DP_BIN * 1e-9, mmid_kg=C.MMID, rho_dry=C.RHO_AER)
root = args.cache or C.RAD_REF_CACHE
drv.ref_cache = rad_cache.RefCache(root, drv.cache_key(**C.rad_ref_src()))
print(f'cache {drv.ref_cache.dir}; {len(hours)} hours {hours[0]}..{hours[-1]}; driver ready {time.time()-t0:.0f}s', flush=True)

if C.AER_SRC == 'mam4':
    ds_mam = {f'{p}_a{m}': C.open_var(f'{p}_a{m}') for p in ('num', 'so4') for m in (1, 2, 3)}
    if C._INIT_BIN == 'dgnum':
        ds_mam.update({f'dgnumwet{m}': C.open_var(f'dgnumwet{m}') for m in (1, 2, 3)})
    def num_ref_fn(t):            # bin_mam4 takes the hour RELATIVE to H0
        return C.bin_mam4(ds_mam, t - C.H0, klevs)[0]
else:
    raise SystemExit('build_ref_cache: only AER_SRC=mam4 is supported (CARMA is a static reservoir)')

emu = None
if args.emu_ckpt:
    import rad_emulator
    emu = rad_emulator.RadEmulator(drv, ckpt=args.emu_ckpt)

n_done = 0
for i, t in enumerate(hours):
    ts = time.time(); did = []
    if args.only in ('ref', 'all') and not drv.ref_ready(t):
        rad_cache_build.ensure_ref(drv, t, num_ref_fn); did.append('ref')
    if args.only in ('clear', 'all') and not drv.ref_cache.has(t, rad_cache.CLEAR_KEYS):
        rad_cache_build.ensure_clear(drv, t); did.append('clear')
    if emu is not None and not drv.ref_cache.has(t, (emu.net_key,)):
        emu.ensure_net_ref(t, num_ref_fn); did.append('net')
    n_done += bool(did)
    if did or i % 20 == 0:
        print(f'  [{i+1}/{len(hours)}] hour {t}: {"+".join(did) or "complete"} {time.time()-ts:.1f}s', flush=True)
print(f'done: {n_done} hours built/completed in {time.time()-t0:.0f}s; {drv.ref_cache.stats()}', flush=True)
print('BUILD_REF_CACHE_DONE', flush=True)
