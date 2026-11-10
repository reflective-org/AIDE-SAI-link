#!/usr/bin/env python3
"""Write a MET_SRC=file directory from the CESM h1 archive itself.

The reader's identity test: a run with MET_SRC=file on a directory built by
this script from the same archive hours must reproduce MET_SRC=cesm exactly
(same float32 fields, same hours), which pins the hour mapping, the level
order and the array layout of scripts/met/emu_to_met.py's format before any
emulator field goes through it. It is also the "CESM winds at 6-h cadence"
control for an emulator-driven run.

  python3 scripts/met/cesm_to_met.py --hour0 8766 --nframe 9 --out runs/<tag>/met_cesm

Same env knobs as coupling.py for the archive (CESM_DIR/CESM_PREFIX/CESM_SUF)
and the band (P_LO_HPA/P_HI_HPA).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from emu_to_met import CESM_DIR, CESM_PREFIX, CESM_SUF, cesm_grid  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--hour0', type=int, required=True, help='archive hour index of frame 0')
    ap.add_argument('--nframe', type=int, required=True)
    ap.add_argument('--step-hours', type=int, default=6)
    ap.add_argument('--out', required=True)
    ap.add_argument('--p-lo', type=float, default=float(os.environ.get('P_LO_HPA', '1.0')))
    ap.add_argument('--p-hi', type=float, default=float(os.environ.get('P_HI_HPA', '150.0')))
    a = ap.parse_args()
    lat, lon, klevs, plev = cesm_grid(a.p_lo, a.p_hi)
    hours = a.hour0 + a.step_hours * np.arange(a.nframe)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    for var, name in (('U', 'u'), ('V', 'v'), ('OMEGA', 'omega'), ('T', 'T')):
        ds = xr.open_dataset(f'{CESM_DIR}/hour_1/{CESM_PREFIX}.h1.{var}{CESM_SUF}')
        f = ds[var].isel(time=hours, lev=klevs).values.astype(np.float32)
        np.save(os.path.join(a.out, f'{name}.npy'), f)
        ds.close()
        print(f'  {var} {f.shape} {time.time()-t0:.0f}s', flush=True)
    ds = xr.open_dataset(f'{CESM_DIR}/hour_1/{CESM_PREFIX}.h1.T{CESM_SUF}', decode_times=False)
    t_days = ds['time'].values[hours]
    ds.close()
    meta = dict(source='cesm-h1', hour0=int(a.hour0), archive_t0='1996-01-01T00:00:00',
                calendar='noleap hour index on the CESM h1 axis', step_hours=a.step_hours,
                nframe=int(a.nframe), klevs=[int(k) for k in klevs],
                plev_pa=[float(p) for p in plev], p_lo_hpa=a.p_lo, p_hi_hpa=a.p_hi,
                lat=[float(x) for x in lat], lon=[float(x) for x in lon],
                t0=f'{t_days[0]:.4f} days since 1996-01-01 (noleap)',
                fields=dict(u='m/s', v='m/s', omega='Pa/s', T='K'),
                level_order='CESM top->bottom (klevs ascending)')
    with open(os.path.join(a.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print(f'wrote {a.out}: hours {hours[0]}..{hours[-1]}', flush=True)


if __name__ == '__main__':
    main()
