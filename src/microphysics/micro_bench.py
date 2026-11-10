"""Microphysics engine A/B inside the production step loop.

`make_benchmarked(primary, reference, ...)` wraps two drop-ins for
`coupling.run_microphysics_full` into one. The PRIMARY engine advances the
run; every MICRO_REF_EVERY steps the REFERENCE engine is also run on the SAME
input state and the two outputs are compared cell by cell. The run itself is
unchanged by the comparison (the reference output is discarded), so

    MICRO_ENGINE=fast     MICRO_REF=emulator   -- the physics trajectory, with the
        emulator's one-step error measured along it ("teacher-forced": every
        input the emulator sees is a physics state);
    MICRO_ENGINE=emulator MICRO_REF=fast       -- the emulator's own trajectory,
        with the physics' one-step answer on the emulator's own states (the
        drift test: are the states the emulator drives itself into still ones
        the physics agrees with?).

Run both and the two `coupled_timeseries_<TAG>.npz` give the trajectory-level
comparison (burdens, size, AOD, forcing); `micro_bench_<TAG>.npz` from each
gives the per-step one-step error and the timing of BOTH engines on the same
cells, same GPU, same process. scripts/validation/plot_micro_bench.py draws it.

WHAT IS COMPARED (all per-box, i.e. mixing ratio x air density, so the
comparison is in the units both engines actually integrate):
  totals   per cell: N = sum_k Nk, M = sum_k Mk, SO2, H2SO4 -- relative error
           |pred - ref| / ref, percentiles over ALL cells, over ACTIVE cells
           (ref N > 1 cm-3, the emulator test-set convention), and over the
           PLUME (the top 1 % of cells by reference SO2+H2SO4+M, i.e. the SAI
           cells, since there is no other source of gas in the band).
  global   burden-weighted signed error sum(w (pred-ref)) / sum(w ref) with
           w = wgt3d (the model's own A_j burden weight) -- the number the
           budget would see.
  In every formula "ref" is the TRUTH engine (`truth`, the fast physics by
  default) whichever role it plays, so the two run directions are comparable:
  the error is always |emulator - physics| / physics.
  per-bin  number rel. error on OCCUPIED bins (ref Nk > 1e-2 cm-3): median and
           p90 per bin, plus the emulator-literature "burst" subset (bins 0..15
           going from <= 1e3 to > 1e3 cm-3 within the step).
  per-lev  median / p90 of the total-N and total-M rel. error per model level.
  zonal    (nlev, nlat) zonal means of N and M for both engines, so a lat-height
           picture of the disagreement can be drawn without storing 3-D fields.
  timing   wall seconds of each engine on this step (step 0 carries JIT /
           torch.compile warm-up for whichever engine runs first; report the
           steady state, as MANIFEST does for the physics).

The npz is rewritten atomically every step (temp + os.replace, like the
checkpoints), so a killed run still leaves a readable file.
"""
import os
import time

import numpy as np

RD = 287.05
BOX_CM3 = 1.0e6


def _pct(v, qs=(50, 90, 99)):
    v = np.asarray(v)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.full(len(qs), np.nan)
    return np.percentile(v, qs)


def _rel(pred, ref, eps=1e-25):
    return np.abs(pred - ref) / np.maximum(ref, eps)


def compare(pred, ref, rho_f, w_f, shape, nbin):
    """pred, ref: tuples (num, mas, so2, h2so4) of mixing-ratio arrays.
    Returns a dict of numpy arrays (all small)."""
    nlev, nlat, nlon = shape
    ncell = nlev * nlat * nlon
    out = {}

    def perbox(t):
        num, mas, so2, h2so4 = t
        N = np.asarray(num).reshape(nbin, ncell).T * rho_f[:, None]
        M = np.asarray(mas).reshape(nbin, ncell).T * rho_f[:, None]
        S = np.asarray(so2).reshape(ncell) * rho_f
        H = np.asarray(h2so4).reshape(ncell) * rho_f
        return N, M, S, H

    Np, Mp, Sp, Hp = perbox(pred)
    Nr, Mr, Sr, Hr = perbox(ref)
    tNp, tNr = Np.sum(1), Nr.sum(1)
    tMp, tMr = Mp.sum(1), Mr.sum(1)

    active = tNr / BOX_CM3 > 1.0
    gas_r = Sr + Hr + tMr
    thr = np.percentile(gas_r, 99.0)
    plume = gas_r >= thr
    out['n_active'] = int(active.sum()); out['n_plume'] = int(plume.sum())

    for name, p, r in (('totN', tNp, tNr), ('totM', tMp, tMr),
                       ('so2', Sp, Sr), ('h2so4', Hp, Hr)):
        e = _rel(p, r)
        out[f'{name}_all'] = _pct(e)
        out[f'{name}_active'] = _pct(e[active])
        out[f'{name}_plume'] = _pct(e[plume])
        # burden-weighted signed global error
        out[f'{name}_global'] = float(((p - r) * w_f).sum() / max((r * w_f).sum(), 1e-300))

    # per-bin on occupied bins
    occ = Nr / BOX_CM3 > 1e-2
    eb = _rel(Np, Nr)
    pb50 = np.full(nbin, np.nan); pb90 = np.full(nbin, np.nan); pbn = np.zeros(nbin, int)
    for k in range(nbin):
        m = occ[:, k]
        pbn[k] = int(m.sum())
        if pbn[k]:
            pb50[k], pb90[k] = np.percentile(eb[m, k], (50, 90))
    out['bin_p50'] = pb50; out['bin_p90'] = pb90; out['bin_nocc'] = pbn
    # the per-bin error of the whole size distribution as one number per cell:
    # Wasserstein-1 along the bin axis of the number-fraction CDFs (bins units)
    act = active
    if act.any():
        cp = np.cumsum(Np[act] / np.maximum(tNp[act], 1e-30)[:, None], 1)
        cr = np.cumsum(Nr[act] / np.maximum(tNr[act], 1e-30)[:, None], 1)
        out['W1_active'] = _pct(np.abs(cp - cr).sum(1))
    else:
        out['W1_active'] = np.full(3, np.nan)

    # burst subset (the emulator's hardest regime): we only see the post-step
    # state here, so "burst" = ref bin 0..15 > 1e3 cm-3 while the cell's N was
    # dominated by larger bins -- approximated by ref N(bins 0..15) > 1e3 cm-3.
    fine_r = Nr[:, :16] / BOX_CM3
    burst = (fine_r > 1e3)                                  # (ncell, 16)
    bcells = burst.any(1)
    out['n_burst'] = int(bcells.sum())
    out['burst_p'] = _pct(eb[:, :16][burst]) if burst.any() else np.full(3, np.nan)

    # per level
    eN = _rel(tNp, tNr).reshape(nlev, nlat * nlon)
    eM = _rel(tMp, tMr).reshape(nlev, nlat * nlon)
    actl = active.reshape(nlev, nlat * nlon)
    lev50N = np.full(nlev, np.nan); lev90N = np.full(nlev, np.nan)
    lev50M = np.full(nlev, np.nan); lev90M = np.full(nlev, np.nan)
    for k in range(nlev):
        m = actl[k]
        if m.any():
            lev50N[k], lev90N[k] = np.percentile(eN[k, m], (50, 90))
            lev50M[k], lev90M[k] = np.percentile(eM[k, m], (50, 90))
    out['lev_totN_p50'] = lev50N; out['lev_totN_p90'] = lev90N
    out['lev_totM_p50'] = lev50M; out['lev_totM_p90'] = lev90M

    # zonal means (per-box units, cm-3 and kg m-3); 'pred' = the non-truth engine
    out['zm_N_pred'] = (tNp / BOX_CM3).reshape(nlev, nlat, nlon).mean(2)
    out['zm_N_ref']  = (tNr / BOX_CM3).reshape(nlev, nlat, nlon).mean(2)
    out['zm_M_pred'] = tMp.reshape(nlev, nlat, nlon).mean(2)
    out['zm_M_ref']  = tMr.reshape(nlev, nlat, nlon).mean(2)
    out['zm_S_pred'] = Sp.reshape(nlev, nlat, nlon).mean(2)
    out['zm_S_ref']  = Sr.reshape(nlev, nlat, nlon).mean(2)
    return out


def make_benchmarked(primary, primary_name, reference, reference_name,
                     every=1, tag='bench', out_path=None, truth='fast'):
    """Return a drop-in for coupling.run_microphysics_full that advances with
    `primary` and, every `every` steps, also runs `reference` on the same input
    and records the comparison. `truth` names the engine that is the ground
    truth of the comparison (the denominator of every relative error and the
    reference of every signed one), whichever of the two roles it plays."""
    out_path = out_path or f'micro_bench_{tag}.npz'
    if truth not in (primary_name, reference_name):
        raise ValueError(f"truth={truth!r} is neither {primary_name!r} nor {reference_name!r}")
    truth_is_primary = (truth == primary_name)
    other = reference_name if truth_is_primary else primary_name
    hist = {}          # key -> list over logged steps
    step = [0]
    t_primary = []; t_reference = []; step_idx = []     # compared steps only
    t_primary_all = []                                  # every step

    def _append(d):
        for k, v in d.items():
            hist.setdefault(k, []).append(v)

    def _save():
        arrs = {k: np.asarray(v) for k, v in hist.items()}
        arrs['step'] = np.asarray(step_idx)
        arrs['t_primary'] = np.asarray(t_primary)
        arrs['t_reference'] = np.asarray(t_reference)
        arrs['t_primary_all'] = np.asarray(t_primary_all)
        arrs['primary'] = np.asarray(primary_name)
        arrs['reference'] = np.asarray(reference_name)
        arrs['truth'] = np.asarray(truth)
        arrs['every'] = np.asarray(every)
        tmp = out_path + '.tmp.npz'
        np.savez(tmp, **arrs)
        os.replace(tmp, out_path)

    def run(num, mas, so2, h2so4, temp3d, pres3d, rh3d, oh3d, wgt3d):
        s = step[0]; step[0] += 1
        t0 = time.time()
        res = primary(num, mas, so2, h2so4, temp3d, pres3d, rh3d, oh3d, wgt3d)
        tp = time.time() - t0
        t_primary_all.append(tp)
        if s % every == 0:
            t1 = time.time()
            ref = reference(num, mas, so2, h2so4, temp3d, pres3d, rh3d, oh3d, wgt3d)
            tr = time.time() - t1
            nbin, nlev, nlat, nlon = num.shape
            ncell = nlev * nlat * nlon
            rho_f = (np.asarray(pres3d) / (RD * np.asarray(temp3d))).reshape(ncell)
            w_f = np.asarray(wgt3d).reshape(ncell)
            tc = time.time()
            pred, tru = (ref, res) if truth_is_primary else (res, ref)
            d = compare(pred[:4], tru[:4], rho_f, w_f, (nlev, nlat, nlon), nbin)
            _append(d); step_idx.append(s); t_primary.append(tp); t_reference.append(tr)
            _save()
            print(f"  [bench] s={s} {primary_name} {tp:.1f}s vs {reference_name} {tr:.1f}s "
                  f"(compare {time.time() - tc:.1f}s) | one-step |{other}-{truth}|/{truth} p50/p90/p99 % "
                  f"active({d['n_active']}): N {d['totN_active'][0]*100:.1f}/{d['totN_active'][1]*100:.1f}/"
                  f"{d['totN_active'][2]*100:.0f} M {d['totM_active'][0]*100:.1f}/{d['totM_active'][1]*100:.1f}/"
                  f"{d['totM_active'][2]*100:.0f} | plume({d['n_plume']}): N {d['totN_plume'][0]*100:.1f}/"
                  f"{d['totN_plume'][1]*100:.1f} M {d['totM_plume'][0]*100:.1f}/{d['totM_plume'][1]*100:.1f} "
                  f"SO2 {d['so2_plume'][0]*100:.1f}/{d['so2_plume'][1]*100:.1f} | global N {d['totN_global']*100:+.2f}% "
                  f"M {d['totM_global']*100:+.2f}% | burst({d['n_burst']}) bin-N {d['burst_p'][0]*100:.0f}/"
                  f"{d['burst_p'][1]*100:.0f}", flush=True)
        return res

    return run
