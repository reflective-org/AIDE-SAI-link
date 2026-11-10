"""Phase 2: 24 h box runs (4 x 6 h calls) of the sampled cells under
(a) TOMAS numerical = tomas_fast.run_microphysics_full_fast (the campaign's r2 engine)
(b) the TOMAS emulator = tomas_emulator.run_microphysics_full_emulator, chained on its own output,
with the knobs in the environment (set by the launcher). T, P, RH held at the frame-time
values; OH = the campaign's SZA-parabola diurnal OH (coupling.oh_sza logic, 60 samples per
6 h) advancing with the clock from the frame time over the 24 h.
usage: run_box.py cells.npz out.npz [engines=fast,emulator]"""
import os, sys, time
import numpy as np

ROOT = '/data/AIDE-SAI-link'
for k, v in dict(P_LO_HPA='1.0', P_HI_HPA='150.0', STEP_HOURS='6', MICRO='full', OH_SUBSTEPS='60',
                 FAST_CELL_CAP='50000', FAST_SORT='1', MET_SRC='file',
                 MET_DIR=f'{ROOT}/runs/emu_winds_P2s200_1997/met', MET_RH='cesm', BC_SRC='clim',
                 CLIM_FILE=f'{ROOT}/runs/bc_clim/bc_clim.npz', H0='8766',
                 XLA_PYTHON_CLIENT_PREALLOCATE='false').items():
    os.environ.setdefault(k, v)
sys.path.insert(0, f'{ROOT}/src')
import _paths  # noqa
import coupling as C
import jax.numpy as jnp

cells_f, out_f = sys.argv[1], sys.argv[2]
engines = (sys.argv[3] if len(sys.argv) > 3 else 'fast,emulator').split(',')
H0 = int(os.environ['H0'])
d = np.load(cells_f)
pops = ('bg', 'sai')
cat = lambda k: np.concatenate([d[f'{p}_{k}'] for p in pops])
num0 = cat('num'); mas0 = cat('mas'); so20 = cat('so2'); h20 = cat('h2so4')
T = cat('T'); P = cat('P'); RH = cat('RH'); step = cat('step'); flat = cat('flat')
N = num0.shape[0]
nlat, nlon = d['lat'].size, d['lon'].size
lat = d['lat'][(flat % (nlat * nlon)) // nlon]; lon = d['lon'][flat % nlon]
it0 = 6 * (step + 1)
print(f'{N} cells; C.OH_SUBSTEPS={C.OH_SUBSTEPS} DT_MICRO={C.DT_MICRO} BOXVOL={C.BOXVOL}', flush=True)


def oh_cells(k):
    """(60, 1, 1, N) OH for call k (hours it0+6k .. +6) per cell: coupling.oh_sza at substep centres."""
    nsub = C.OH_SUBSTEPS; dt_h = (C.DT_MICRO / nsub) / 3600.0
    oh = np.empty((nsub, N))
    la = jnp.asarray(lat); lo = jnp.asarray(lon)
    for m in range(nsub):
        hr = H0 + it0 + 6 * k + (m + 0.5) * dt_h
        doy = ((hr // 24) % 365 + 1).astype(float)
        cs = C.calc_solar_zenith_angle(la, jnp.asarray(doy), jnp.asarray(hr % 24.0), lo)
        oh[m] = np.asarray(C.calc_oh_parabola(cs, jnp.asarray(C.OH_PEAK)))
    return oh.reshape(nsub, 1, 1, N)


OH = [oh_cells(k) for k in range(4)]
# sanity: reproduce coupling.oh_sza's grid form for one cell (scalar doy/hour path)
sh = lambda a: a.reshape(1, 1, N)
T3, P3, RH3 = sh(T), sh(P), sh(RH)
W3 = np.ones((1, 1, N))
res = dict(oh_mean=np.stack([o.mean(0).reshape(N) for o in OH]))

for eng in engines:
    if eng == 'fast':
        import tomas_fast as TFm
        fn = TFm.run_microphysics_full_fast
    else:
        import tomas_emulator as TE
        fn = TE.run_microphysics_full_emulator
        print('emulator:', TE.describe(), flush=True)
    num = num0.T.reshape(40, 1, 1, N).copy(); mas = mas0.T.reshape(40, 1, 1, N).copy()
    so2 = sh(so20).copy(); h2 = sh(h20).copy()
    trajN = [num0]; trajM = [mas0]; trajS = [so20]; trajH = [h20]; tim = []; clips = []
    for k in range(4):
        num, mas, so2, h2 = (np.maximum(np.asarray(a, np.float64), 0.0) for a in (num, mas, so2, h2))
        t0 = time.time()
        r = fn(num, mas, so2, h2, T3, P3, RH3, OH[k], W3)
        num, mas, so2, h2 = (np.asarray(a, np.float64) for a in r[:4])
        tim.append(time.time() - t0); clips.append(r[4:6])
        trajN.append(num.reshape(40, N).T.copy()); trajM.append(mas.reshape(40, N).T.copy())
        trajS.append(so2.reshape(N).copy()); trajH.append(h2.reshape(N).copy())
        print(f'{eng} call {k}: {tim[-1]:.2f}s finite N {np.isfinite(num).all()} M {np.isfinite(mas).all()} '
              f'minN {num.min():.3g} minM {mas.min():.3g}', flush=True)
    res[f'{eng}_N'] = np.stack(trajN).astype(np.float64); res[f'{eng}_M'] = np.stack(trajM)
    res[f'{eng}_S'] = np.stack(trajS); res[f'{eng}_H'] = np.stack(trajH)
    res[f'{eng}_time'] = np.array(tim); res[f'{eng}_clip'] = np.array(clips)
    if eng == 'emulator':
        # production triage on these cells (cell-local test, first call only)
        import torch
        rho = P / (C.RD * T)
        tt = lambda a: torch.as_tensor(np.ascontiguousarray(a, np.float64), device='cuda')
        Nk = num0 * rho[:, None]; Mk = mas0 * rho[:, None]
        Nk = np.where(Nk > C.NEPS_N, Nk, 0.0)
        Mk = np.clip(Mk, Nk * C.XK_NP[None, :-1], Nk * C.XK_NP[None, 1:])
        qm = TE.quiet_mask(tt(Nk), tt(Mk), tt(h20 * rho), tt(so20 * rho), tt(T), tt(P), tt(RH),
                           tt(OH[0].reshape(60, N).mean(0)))[0]
        res['triage_quiet'] = qm.cpu().numpy()
np.savez(out_f, **res)
print('saved', out_f, flush=True)
