"""Deployed (routed) emulator on the FULL frame grid, 4 chained 6 h calls per frame, no transport,
then extraction of the sampled cells. Forcing identical to run_box.py (T/P/RH frozen at frame time,
SZA OH advancing). Records each sampled cell's route per call by intercepting the wrapper's own
selection (TE._oh_substeps_for gets the physics index sets; TE.quiet_mask the triage mask).
usage: run_routed.py cells.npz out.npz i_proc n_proc"""
import os, sys, time, json
import numpy as np
ROOT = '/data/AIDE-SAI-link'
for k, v in dict(P_LO_HPA='1.0', P_HI_HPA='150.0', STEP_HOURS='6', MICRO='full', OH_SUBSTEPS='60',
                 FAST_CELL_CAP='50000', FAST_SORT='1', MET_SRC='file',
                 MET_DIR=f'{ROOT}/runs/emu_winds_P2s200_1997/met', MET_RH='cesm', BC_SRC='clim',
                 CLIM_FILE=f'{ROOT}/runs/bc_clim/bc_clim.npz', H0='8766',
                 XLA_PYTHON_CLIENT_PREALLOCATE='false').items():
    os.environ.setdefault(k, v)
sys.path.insert(0, f'{ROOT}/src'); sys.path.insert(0, f'{ROOT}/src/met')
import _paths  # noqa
import coupling as C
import jax.numpy as jnp
import tomas_emulator as TE
import bc_clim
print('emulator:', TE.describe(), flush=True)

cells_f, out_f, ip, npr = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
CAMP = f'{ROOT}/runs/emu_campaign_1997'
MET_DIR = os.environ['MET_DIR']; H0 = int(os.environ['H0'])
meta = json.load(open(f'{MET_DIR}/meta.json')); klevs = [int(k) for k in meta['klevs']]
Tmet = np.load(f'{MET_DIR}/T.npy', mmap_mode='r')
dRH = bc_clim.ClimArchive(os.environ['CLIM_FILE'], MET_DIR).open_var('RELHUM')
d = np.load(cells_f)
steps = d['steps']; PER = 125; NB = d['bg_num'].shape[0]; N = 2 * NB
lat = jnp.asarray(d['lat'])[:, None]; lon = jnp.asarray(d['lon'])[None, :]

rec = {}
_orig_oh = TE._oh_substeps_for
def _oh_wrap(oh3d, idx, *a, **k):
    rec['phys'].append(np.asarray(idx).copy()); return _orig_oh(oh3d, idx, *a, **k)
TE._oh_substeps_for = _oh_wrap
_orig_qm = TE.quiet_mask
def _qm_wrap(*a, **k):
    r = _orig_qm(*a, **k); rec['quiet'] = r[0].clone(); return r
TE.quiet_mask = _qm_wrap


def oh_grid(hour0, nlev):
    nsub = C.OH_SUBSTEPS; dt_h = (C.DT_MICRO / nsub) / 3600.0
    oh = np.empty((nsub, lat.shape[0], lon.shape[1]))
    for m in range(nsub):
        hr = hour0 + (m + 0.5) * dt_h
        cs = C.calc_solar_zenith_angle(lat, jnp.asarray(float(int(hr // 24) % 365 + 1)), jnp.asarray(float(hr % 24.0)), lon)
        oh[m] = np.asarray(C.calc_oh_parabola(cs, jnp.asarray(C.OH_PEAK)))
    return np.broadcast_to(oh[:, None], (nsub, nlev) + oh.shape[1:])


outN = np.full((5, N, 40), np.nan); outM = np.full((5, N, 40), np.nan)
outS = np.full((5, N), np.nan); outH = np.full((5, N), np.nan)
route = np.full((4, N), -1, np.int8); glob = []
mine = [j for j in range(len(steps)) if j % npr == ip]
for j in mine:
    s = int(steps[j])
    for pop, run, off in (('bg', 'r2c_tomas_fast_ctl', 0), ('sai', 'r2_tomas_fast', NB)):
        t0 = time.time()
        rows = np.arange(off + j * PER, off + (j + 1) * PER)
        assert (d[f'{pop}_step'][j * PER:(j + 1) * PER] == s).all()
        flat = d[f'{pop}_flat'][j * PER:(j + 1) * PER]
        fr = np.load(f'{CAMP}/{run}/frames3d/step_{s:04d}.npz')
        it0 = int(fr['hour']); assert it0 == 6 * (s + 1)
        num = np.asarray(fr['num'], np.float64); mas = np.asarray(fr['mas'], np.float64)
        so2 = np.asarray(fr['so2'], np.float64); h2 = np.asarray(fr['h2so4'], np.float64)
        nbin, nlev, nlat, nlon = num.shape
        T = np.asarray(Tmet[it0 // 6], np.float64) + np.asarray(fr['dT_rad'], np.float64)
        P = np.broadcast_to(np.asarray(fr['plev_pa'])[:, None, None], T.shape).copy()
        RH = np.clip(dRH['RELHUM'].isel(time=H0 + it0, lev=klevs).values / 100.0, 0.0, 1.0)
        W = np.ones(T.shape)
        x = lambda a: a.reshape(nbin, -1)[:, flat].T
        assert np.allclose(x(num), d[f'{pop}_num'][j * PER:(j + 1) * PER]) and np.allclose(T.reshape(-1)[flat], d[f'{pop}_T'][j * PER:(j + 1) * PER])
        outN[0, rows] = x(num); outM[0, rows] = x(mas); outS[0, rows] = so2.reshape(-1)[flat]; outH[0, rows] = h2.reshape(-1)[flat]
        TE._drift = TE._DriftState()                       # fresh online-correction EMA + call counter per frame
        for k in range(4):
            num, mas, so2, h2 = (np.maximum(np.asarray(a, np.float64), 0.0) for a in (num, mas, so2, h2))
            rec.clear(); rec['phys'] = []
            tc = time.time()
            r = TE.run_microphysics_full_emulator(num, mas, so2, h2, T, P, RH, oh_grid(H0 + it0 + 6 * k, nlev), W)
            num, mas, so2, h2 = (np.asarray(a, np.float64) for a in r[:4])
            ncell = num[0].size
            code = np.zeros(ncell, np.int8)                        # 0 network
            code[rec['quiet'].cpu().numpy()] = 3                   # 3 triage-quiet (analytic)
            if len(rec['phys']) >= 2:
                code[rec['phys'][1]] = 2                           # 2 burst re-run by physics
            ns = TE._drift.last.get('n_samp', 0)
            p0 = rec['phys'][0]
            code[p0] = 1                                           # 1 static physics (TOP/NUM/COARSE)
            if ns:
                code[p0[-ns:]] = 4                                 # 4 online-correction sample (physics answer kept)
            route[k, rows] = code[flat]
            glob.append(dict(step=s, pop=pop, call=k, sec=time.time() - tc,
                             **{f'frac_{n}': float((code == c).mean()) for c, n in enumerate(('net', 'phys', 'burst', 'quiet', 'corr'))}))
            outN[k + 1, rows] = x(num); outM[k + 1, rows] = x(mas)
            outS[k + 1, rows] = so2.reshape(-1)[flat]; outH[k + 1, rows] = h2.reshape(-1)[flat]
            print(f'step {s} {pop} call {k}: {time.time() - tc:.1f}s finite {np.isfinite(num).all() and np.isfinite(mas).all()} '
                  f'sample routes {np.bincount(code[flat], minlength=5)} grid {[round(glob[-1][q], 4) for q in glob[-1] if q.startswith("frac")]}', flush=True)
        print(f'== step {s} {pop} done in {time.time() - t0:.0f}s', flush=True)
        del num, mas, so2, h2, fr
    np.savez(out_f, emulator_N=outN, emulator_M=outM, emulator_S=outS, emulator_H=outH, route=route,
             glob=json.dumps(glob), done_steps=np.array([steps[i] for i in mine if i <= j]))
print('saved', out_f, flush=True)
