"""Phase 1: sample 5000 background (r2c control) + 5000 SAI-plume (r2 injection)
cells from the 1997 campaign frames3d, with the frame-time T, P, RH, and the
production-routing flags each cell would get on ITS run's full grid.
CPU only. Output: cells.npz."""
import os, sys, json
import numpy as np
from concurrent.futures import ThreadPoolExecutor

ROOT = '/data/AIDE-SAI-link'
sys.path.insert(0, f'{ROOT}/src/met')
import bc_clim

CAMP = f'{ROOT}/runs/emu_campaign_1997'
MET_DIR = f'{ROOT}/runs/emu_winds_P2s200_1997/met'
CLIM = f'{ROOT}/runs/bc_clim/bc_clim.npz'
H0 = 8766; RD = 287.05
OUT = sys.argv[1]
NSTEP = 40; PER = 125; RATIO = 3.0
STEPS = np.round(np.linspace(20, 1439, NSTEP)).astype(int)
rng = np.random.default_rng(1997)

meta = json.load(open(f'{MET_DIR}/meta.json'))
assert meta['hour0'] == H0 and meta['step_hours'] == 6
klevs = [int(k) for k in meta['klevs']]
Tmet = np.load(f'{MET_DIR}/T.npy', mmap_mode='r')
arch = bc_clim.ClimArchive(CLIM, MET_DIR)
dRH = arch.open_var('RELHUM')
K_MW = 96.0 / 64.0; K_MWH = 96.0 / 98.0
NC = 24 * 192 * 288


def route_flags(num, mas, so2, h2, rho):
    """production selection (tomas_emulator.run_microphysics_full_emulator):
    TOP 1 % by sulfur, NUM 4 % of the rest by N, COARSE 5 % of the rest by
    mass in bins >= 21; on this frame's full grid. Returns (NC,) int8 code:
    0 network, 1 top, 2 num, 3 coarse."""
    r = rho.reshape(-1)
    M = mas.reshape(40, -1)
    N = num.reshape(40, -1)
    sulf = (M.sum(0) + so2.reshape(-1) * K_MW + h2.reshape(-1) * K_MWH) * r
    code = np.zeros(NC, np.int8)
    k = int(round(0.01 * NC)); i1 = np.argpartition(-sulf, k)[:k]; code[i1] = 1
    nt = (N.sum(0) * r).astype(np.float64); nt[code > 0] = -1
    k = int(round(0.04 * NC)); i2 = np.argpartition(-nt, k)[:k]; code[i2] = 2
    cm = (M[21:].sum(0) * r).astype(np.float64); cm[code > 0] = -1
    k = int(round(0.05 * NC)); i3 = np.argpartition(-cm, k)[:k]; code[i3] = 3
    return code


def do_step(s):
    a = np.load(f'{CAMP}/r2_tomas_fast/frames3d/step_{s:04d}.npz')
    c = np.load(f'{CAMP}/r2c_tomas_fast_ctl/frames3d/step_{s:04d}.npz')
    assert int(a['step']) == s and int(c['step']) == s
    it0 = int(a['hour'])                   # = 6 (s+1): the next micro call's it0
    assert it0 == 6 * (s + 1) and int(a['hour_abs']) == H0 + it0
    imet = it0 // 6                        # met frame index ((H0+it0)-hour0)/6
    T = np.asarray(Tmet[imet], np.float64) + np.asarray(c['dT_rad'], np.float64)
    Ts = np.asarray(Tmet[imet], np.float64) + np.asarray(a['dT_rad'], np.float64)
    P = np.broadcast_to(np.asarray(a['plev_pa'])[:, None, None], T.shape)
    RH = np.clip(dRH['RELHUM'].isel(time=H0 + it0, lev=klevs).values / 100.0, 0.0, 1.0)
    out = {}
    lrng = np.random.default_rng(1997 + s)
    # pick cells
    Ms, Mc, Ns, Nc = a['mtot'], c['mtot'], a['ntot'], c['ntot']
    plume = ((Ms > RATIO * Mc) | (Ns > RATIO * Nc)) & (Ms > 0) & (Ns > 0)
    pidx = np.flatnonzero(plume.reshape(-1))
    sai = lrng.choice(pidx, PER, replace=False)
    bidx = np.flatnonzero((Nc > 0).reshape(-1))
    bg = lrng.choice(bidx, PER, replace=False)
    for pop, fr, Tf, idx in (('bg', c, T, bg), ('sai', a, Ts, sai)):
        num = np.asarray(fr['num'], np.float64); mas = np.asarray(fr['mas'], np.float64)
        so2 = np.asarray(fr['so2'], np.float64); h2 = np.asarray(fr['h2so4'], np.float64)
        rho = P / (RD * Tf)
        code = route_flags(num, mas, so2, h2, rho)
        out[pop] = dict(num=num.reshape(40, -1)[:, idx].T, mas=mas.reshape(40, -1)[:, idx].T,
                        so2=so2.reshape(-1)[idx], h2so4=h2.reshape(-1)[idx],
                        T=Tf.reshape(-1)[idx], P=P.reshape(-1)[idx], RH=RH.reshape(-1)[idx],
                        flat=idx, step=np.full(PER, s), route=code[idx],
                        plume_frac=np.full(PER, plume.mean()))
        del num, mas
    print(f'step {s}: plume frac {plume.mean():.3f}', flush=True)
    return out


with ThreadPoolExecutor(4) as ex:
    res = list(ex.map(do_step, STEPS))
lat = np.load(f'{CAMP}/r2_tomas_fast/frames3d/step_0020.npz')['lat']
lon = np.load(f'{CAMP}/r2_tomas_fast/frames3d/step_0020.npz')['lon']
d0 = np.load(f'{CAMP}/r2_tomas_fast/frames3d/step_0020.npz')
save = {}
for pop in ('bg', 'sai'):
    for k in res[0][pop]:
        save[f'{pop}_{k}'] = np.concatenate([r[pop][k] for r in res])
save['dp_bin_nm'] = d0['dp_bin_nm']; save['lat'] = d0['lat']; save['lon'] = d0['lon']; save['plev_pa'] = d0['plev_pa']
save['steps'] = STEPS
np.savez(OUT, **save)
print('saved', OUT, {k: v.shape for k, v in save.items()})
