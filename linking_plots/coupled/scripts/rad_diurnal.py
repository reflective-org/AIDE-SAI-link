"""Diurnal-mean TOA SW from the model's own radiation on saved states. One process per GPU.
usage: rad_diurnal.py GPU_INDEX NGPU   (tasks = (pair, day) for day in DAYS, round-robin)
For each day d, frames s=4d..4d+3 (UTC 12,18,00,06) are re-solved with the sun shifted by +1.5,+3,+4.5 h -> with the
saved frame values (shift 0) 16 solar times per day, 1.5 h apart. Frame 4d is also re-solved at shift 0 as a
reproduction check against the saved toa_sw_up."""
import os, sys, time
import numpy as np
sys.argv = sys.argv  # noqa
import rad_offline as R
import jax, jax.numpy as jnp
from jax.scipy.special import erf

OUTD = '/data/AIDE-SAI-link/linking_plots/coupled/cache/diurnal'
PAIRS = {'emu': ('r1_tomas_emu', 'r1c_tomas_emu_ctl'), 'num': ('r2_tomas_fast', 'r2c_tomas_fast_ctl'),
         'mam': ('r3_mam4', 'r3c_mam4_ctl')}
DAYS = list(range(int(os.environ.get('DAY0', '2')), int(os.environ.get('LP_NDAYS', '365')), int(os.environ.get('DSTEP', '5'))))
SHIFTS = (1.5, 3.0, 4.5)
SIG = (1.8, 1.6, 1.8, 1.6)

@jax.jit
def modes_to_bins_j(num_m, so4_m, lnxk):
    phi = lambda z: 0.5 * (1.0 + erf(z / jnp.sqrt(2.0)))
    nb = 0.0
    for m in range(4):
        n = num_m[m]; s_ = so4_m[m]; valid = (n > 0) & (s_ > 0)
        s = 3.0 * jnp.log(SIG[m])
        lnmg = jnp.log(jnp.where(valid, s_ / jnp.maximum(n, 1e-300), 1.0)) - 0.5 * s * s
        fn = jnp.diff(phi((lnxk[:, None, None, None] - lnmg[None]) / s), axis=0)
        nb = nb + jnp.where(valid[None], fn * n[None], 0.0)
    return nb

def get_num(z):
    num = z['num']
    if num.shape[0] == R.C.NBINS:
        return np.asarray(num, np.float64)
    mmid = np.asarray(R.C.MMID); lnxk = jnp.asarray(np.log(np.asarray(R.C.XK_NP)))
    return np.asarray(modes_to_bins_j(jnp.asarray(num, jnp.float64), jnp.asarray(z['mas'], jnp.float64), lnxk))

if __name__ == '__main__':
    gi, ng = int(sys.argv[1]), int(sys.argv[2])
    os.makedirs(OUTD, exist_ok=True)
    tasks = [(p, d) for d in DAYS for p in PAIRS]
    tasks = tasks[gi::ng]
    rad, Tb = R.build()
    t0 = time.time()
    for p, d in tasks:
        out = f'{OUTD}/{p}_{d:03d}.npz'
        if os.path.exists(out):
            continue
        tt = time.time(); res = {}
        for role, run in zip(('inj', 'ctl'), PAIRS[p]):
            up16 = 0; up8 = 0; dn16 = 0; dn8 = 0; olr = 0
            for s in range(4 * d, 4 * d + 4):
                z = R.np.load(R.frame_file(run, s))
                num = get_num(z)
                up0 = z['toa_sw_up'].astype(np.float64); dn0 = z['toa_sw_dn'].astype(np.float64)
                up16 = up16 + up0; dn16 = dn16 + dn0; up8 = up8 + up0; dn8 = dn8 + dn0; olr = olr + z['toa_olr']
                if s == 4 * d:
                    r0 = R.call(rad, Tb, s, num, 0.0, split=False)
                    dd = np.abs(r0['sw_up_toa'].astype(np.float32) - z['toa_sw_up'])
                    res[f'{role}_repro_maxabs'] = float(dd.max()); res[f'{role}_repro_bitfrac'] = float((dd == 0).mean())
                    res[f'{role}_repro_gmdiff'] = float((r0['sw_up_toa'] - up0).mean())
                for sh in SHIFTS:
                    r = R.call(rad, Tb, s, num, sh, split=False)
                    up16 = up16 + r['sw_up_toa']; dn16 = dn16 + r['sw_dn_toa']
                    if sh == 3.0:
                        up8 = up8 + r['sw_up_toa']; dn8 = dn8 + r['sw_dn_toa']
            res[f'{role}_swup16'] = (up16 / 16).astype(np.float32); res[f'{role}_swdn16'] = (dn16 / 16).astype(np.float32)
            res[f'{role}_swup8'] = (up8 / 8).astype(np.float32); res[f'{role}_swdn8'] = (dn8 / 8).astype(np.float32)
            res[f'{role}_olr4'] = (olr / 4).astype(np.float32)
        np.savez(out + '.tmp.npz', day=d, **res); os.replace(out + '.tmp.npz', out)
        print(f'gpu{gi} {p} day {d} {time.time() - tt:.0f}s  repro inj {res["inj_repro_maxabs"]:.2e} ctl {res["ctl_repro_maxabs"]:.2e}  '
              f'elapsed {time.time() - t0:.0f}s', flush=True)
