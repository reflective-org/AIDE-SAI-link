"""Offline re-run of the model's own flux-mode radiation (src/radiation/radiation.py RadiationDriver.heating, exactly as
coupling.py calls it under RAD_MODE=flux: rad.heating(H0+it1, aer_to_bins(num,mas)[0], T_band_override=temp(it1), split=True))
on the SAVED 3-D frames, with the sun optionally shifted by a number of hours (state, gases, surface fixed).
No src/ edits: the solar-time shift monkeypatches radiation.solar_zenith from outside."""
import os, sys, json, time, datetime
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
for k, v in dict(MET_SRC='file', MET_DIR=os.environ.get('LP_MET', '/data/AIDE-SAI-link/runs/emu_winds_P2s200_1997/met'), MET_RH='cesm', BC_SRC='clim',
                 CLIM_FILE='/data/AIDE-SAI-link/runs/bc_clim/bc_clim.npz', RAD_MODE='flux', H0='8766').items():
    os.environ.setdefault(k, v)
sys.path.insert(0, '/data/AIDE-SAI-link/src')
import _paths  # noqa
import numpy as np
import coupling as C   # constants only (MMID, DP_BIN, RHO_AER); main() is not run
import bc_clim
import radiation

RUNS = os.environ.get('LP_RUNS', '/data/AIDE-SAI-link/runs/emu_campaign_1997')
RUNS_Y2 = os.environ.get('LP_RUNS_Y2', '/data/AIDE-SAI-link/runs/emu_campaign_1997_y2')
def frame_file(run, s):
    return f'{RUNS if s < 1460 else RUNS_Y2}/{run}/frames3d/step_{s:04d}.npz'
MET = os.environ['MET_DIR']; H0 = 8766
SHIFT_H = [0.0]
_orig_sz = radiation.solar_zenith
def _shifted(time_val, lat, lon):
    if SHIFT_H[0] == 0.0:
        return _orig_sz(time_val, lat, lon)
    return _orig_sz(time_val + datetime.timedelta(hours=SHIFT_H[0]), lat, lon)
radiation.solar_zenith = _shifted

def build():
    meta = json.load(open(f'{MET}/meta.json'))
    clim = bc_clim.ClimArchive(os.environ['CLIM_FILE'], MET)
    dU = clim.open_var('U')
    lat = dU['lat'].values; lon = dU['lon'].values
    rad = radiation.RadiationDriver(clim.open_var, [int(k) for k in meta['klevs']], lat, lon, C.DP_BIN * 1e-9,
                                    mmid_kg=C.MMID, rho_dry=C.RHO_AER)
    Tb = np.load(f'{MET}/T.npy', mmap_mode='r')
    return rad, Tb

def bins(z):
    num = z['num']
    if num.shape[0] != C.NBINS:
        sys.path.insert(0, '/data/AIDE-SAI-link/linking_plots/coupled/scripts')
        from common import mam_modes_to_bins
        num, _ = mam_modes_to_bins(num, z['mas'], z['dp_bin_nm'])
    return np.asarray(num, np.float64)

def call(rad, Tb, s, num, shift_h, split=True):
    SHIFT_H[0] = float(shift_h)
    hr, d = rad.heating(H0 + 6 * (s + 1), num, T_band_override=np.array(Tb[s + 1]), split=split)
    return {k: np.asarray(d[k]) for k in ('sw_dn_toa', 'sw_up_toa', 'olr')}

if __name__ == '__main__':
    run = sys.argv[1]; s = int(sys.argv[2])
    rad, Tb = build()
    z = np.load(frame_file(run, s))
    num = bins(z)
    for split in (True, False):
        t = time.time(); d = call(rad, Tb, s, num, 0.0, split=split); dt = time.time() - t
        for k, fk in (('sw_up_toa', 'toa_sw_up'), ('olr', 'toa_olr'), ('sw_dn_toa', 'toa_sw_dn')):
            a = d[k].astype(np.float32); b = z[fk]
            print(f'split={split} {k}: max|diff| {np.abs(a - b).max():.3e} W/m2, bitwise-equal frac {(a == b).mean():.4f}, '
                  f'max rel {np.abs(a - b).max() / np.abs(b).max():.2e}   ({dt:.1f}s)')
    t = time.time(); d = call(rad, Tb, s, num, 3.0); print('shift 3h', time.time() - t, d['sw_dn_toa'].mean(), z['toa_sw_dn'].mean())
