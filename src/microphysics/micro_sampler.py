"""Training-data sampler at the microphysics seam.

`make_sampled(engine, every, n, tag)` wraps a drop-in for
`coupling.run_microphysics_full` so that every `every` steps it also writes a
stratified sample of the cells the engine is about to see -- the exact
post-advection, post-clip per-box state and forcing -- to
`micro_samples_<TAG>_s<step>.npz`. The run itself is unchanged.

This is how the emulator gets training data from the states THIS model
produces (docs/EMULATOR.md: every failure of the first integration was a
state the generator never made -- zero night-time OH, SO2 advected to zero,
post-advection and post-settling shapes, fresh injection into very clean
air). `scripts/emulator/label_chains.py` turns the samples into physics-
labelled 30-minute chains in the emulator's dataset format, and
`scripts/emulator/finetune_bintf.py` fine-tunes on them.

Sampling. Four strata, so the rare cells that carry the error are
over-represented without losing the background:
  plume   the top 1 % of cells by gas + aerosol sulfur                -> all of them
  active  cells the triage test would send to the network            -> n * 0.6
  quiet   cells the triage test would hand to the analytic step      -> n * 0.1
  empty   cells with N < 1 cm-3                                       -> n * 0.05
(the remaining 0.25 of n is spent on the plume when it is larger than that).
Stratum labels are stored, so the labeller can re-weight.

What is stored per cell (per box, box = 1 m3, float32 unless noted):
  Nk (n,40) Mk (n,40) H S T P RH (n,) -- the engine's own input after the clip
  OH13 (n,13) -- the OH endpoints this step saw (for the first 6 h window)
  lev, lat_i, lon_i (int32), lat, lon (deg), hour_abs (float64, hours since
  1996-01-01 00Z of the CESM h1 series) -- what label_chains.py needs to
  extend the diurnal OH beyond the step with coupling's own SZA parabola.
"""
import os
import time

import numpy as np

RD = 287.05
BOX_CM3 = 1.0e6


def make_sampled(engine, every=4, n=5000, tag='run', seed=0, out_dir='.'):
    import coupling as C                     # already imported by the driver
    import tomas_emulator as TE              # the triage test, in torch
    import torch
    import xarray as xr
    rng = np.random.default_rng(seed)
    step = [0]
    # lat/lon are locals of coupling.main(); read them the way it does
    with xr.open_dataset(f"{C.H1}.U{C.SUF_H}") as _d:
        LAT = _d['lat'].values.astype(np.float64); LON = _d['lon'].values.astype(np.float64)

    def run(num, mas, so2, h2so4, temp3d, pres3d, rh3d, oh3d, wgt3d):
        s = step[0]; step[0] += 1
        if s % every == 0:
            t0 = time.time()
            nbin, nlev, nlat, nlon = num.shape
            ncell = nlev * nlat * nlon
            T_f = TE._to_torch(temp3d).reshape(ncell); P_f = TE._to_torch(pres3d).reshape(ncell)
            RH_f = TE._to_torch(rh3d).reshape(ncell)
            rho_f = P_f / (RD * T_f)
            OH13, OHbar = TE._oh_endpoints_and_mean(oh3d, nlev, nlat, nlon)
            Nk = TE._to_torch(num).reshape(nbin, ncell).T * rho_f[:, None]
            Mk = TE._to_torch(mas).reshape(nbin, ncell).T * rho_f[:, None]
            S = TE._to_torch(so2).reshape(ncell) * rho_f
            H = TE._to_torch(h2so4).reshape(ncell) * rho_f
            Nk = torch.where(Nk > C.NEPS_N, Nk, 0.0)
            Mk = torch.minimum(torch.maximum(Mk, Nk * TE._XK[:-1]), Nk * TE._XK[1:])
            qm, _, _ = TE.quiet_mask(Nk, Mk, H, S, T_f, P_f, RH_f, OHbar)
            Ntot = Nk.sum(1)
            gas = (S * TE._S_SO2 + H * TE._S_H2SO4 + Mk.sum(1) * TE._S_SO4).cpu().numpy()
            quiet = qm.cpu().numpy(); empty = (Ntot / BOX_CM3 < 1.0).cpu().numpy()
            plume = gas >= np.percentile(gas, 99.0)
            active = ~quiet & ~empty & ~plume
            quiet_ = quiet & ~plume; empty_ = empty & ~plume & ~quiet
            idx = np.nonzero(plume)[0]
            if len(idx) > n // 2:                       # the plume is 1 % of the grid = 13k cells; cap it
                idx = rng.choice(idx, n // 2, replace=False)
            picks = [(idx, 0)]
            budget = max(n - min(len(idx), n // 4), 0)
            for name, mask, frac, lab in (('active', active, 0.6 / 0.75, 1), ('quiet', quiet_, 0.1 / 0.75, 2),
                                          ('empty', empty_, 0.05 / 0.75, 3)):
                cand = np.nonzero(mask)[0]
                k = min(len(cand), int(round(budget * frac)))
                if k:
                    picks.append((rng.choice(cand, k, replace=False), lab))
            sel = np.concatenate([p[0] for p in picks]); lab = np.concatenate([np.full(len(p[0]), p[1], np.int8) for p in picks])
            sel_t = torch.from_numpy(sel).to(Nk.device)
            kk, jj, ii = np.unravel_index(sel, (nlev, nlat, nlon))
            hour_abs = float(C.H0 + s * C.STEP_HOURS)
            f32 = lambda t: t[sel_t].cpu().numpy().astype(np.float32)
            fn = os.path.join(out_dir, f'micro_samples_{tag}_s{s:04d}.npz')
            np.savez(fn, Nk=f32(Nk), Mk=f32(Mk), H=f32(H), S=f32(S), T=f32(T_f), P=f32(P_f), RH=f32(RH_f),
                     OH13=OH13.T[sel_t].cpu().numpy().astype(np.float32),
                     lev=kk.astype(np.int32), lat_i=jj.astype(np.int32), lon_i=ii.astype(np.int32),
                     lat=LAT[jj].astype(np.float32), lon=LON[ii].astype(np.float32),
                     plev_pa=np.asarray(pres3d)[:, 0, 0][kk].astype(np.float32), hour_abs=np.full(len(sel), hour_abs),
                     stratum=lab, step=s, n_plume=int(plume.sum()), n_active=int(active.sum()),
                     n_quiet=int(quiet_.sum()), n_empty=int(empty_.sum()))
            print(f"  [sample] s={s}: {len(sel)} cells -> {fn} (plume {int((lab == 0).sum())}, active "
                  f"{int((lab == 1).sum())}, quiet {int((lab == 2).sum())}, empty {int((lab == 3).sum())}; "
                  f"grid: {plume.sum()}/{active.sum()}/{quiet_.sum()}/{empty_.sum()}) in {time.time() - t0:.1f}s", flush=True)
        return engine(num, mas, so2, h2so4, temp3d, pres3d, rh3d, oh3d, wgt3d)

    return run
