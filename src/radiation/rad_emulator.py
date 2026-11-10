"""The radiation emulator at the two-stream seam: RAD_ENGINE=emulator.

Replaces the RRTMGP solve of the EVOLVED radiation call (RadiationDriver.heating
with the model's own bins at T_eff = T_CESM + dT_rad) with two per-column
networks on top of the exact, cached clear-sky fields of the archive hour
(rad_cache.py, filled by scripts/radiation/build_ref_cache.py or on the fly):

    hr1  =  hr_clear(T_CESM)                     cached, exact
          + PlanckNet(gas column, dT)            the solver's response to dT
          + AerNet(gas column, sun, band optics, dT)   the aerosol-induced heating
    TOA fluxes the same way; sw_dn_toa and the surface fluxes are the cached
    clear-sky values (the surface fluxes are not consumed by the model).

Everything around the solve is the physics, unchanged: the Mie optics
(bins -> band tau/ssa/g on the band levels, 0.03 s), the 550 nm AOD, the
archive reads, the solar geometry. The reference call of RAD_MODE=anomaly is
either the exact cached physics (RAD_EMU_ANOM=phys) or the same networks on
the reference aerosol (RAD_EMU_ANOM=net, default: the networks' errors are
correlated between the two calls and cancel in the background, where the
two aerosols are nearly the same; the per-hour network output on the
reference aerosol is itself cached, keyed by the checkpoint).

Why a Planck NETWORK and not a cached Jacobian: the solve's response to dT is
exactly linear for a uniform dT but strongly non-linear (at the 0.5 K scale)
for the sharp vertical dT profiles the injection makes, because RRTMGP's
face temperatures come from a WENO5 reconstruction whose smoothness weights
move with the local profile (scripts/radiation/check_planck_jacobian.py: a
full 24x24 FD Jacobian leaves a 35 % rms error). The network is trained on
exactly that response.

Networks: scripts/radiation/train_rad_mlp.py, exported as .npz; evaluated
here in JAX (jit, float32) so the run has no torch dependency at this seam.
Features: rad_features.py (one definition for labels and inference).

Knobs: RAD_EMU_CKPT (directory with aer.npz + planck.npz), RAD_EMU_ANOM
(net|phys), RAD_REF=rrtmgp + RAD_REF_EVERY (A/B against the physics on the
same input, rad_bench_<TAG>.npz; the run advances with the emulator),
RAD_EMU_CLIP (output sanity clamp, K/day; default 20, 0 = off).
"""
import os
import sys
import time
import json
import glob, hashlib

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _paths                      # noqa: E402,F401

import rad_cache                   # noqa: E402
import rad_cache_build             # noqa: E402
import rad_features as F           # noqa: E402
import radiation as R              # noqa: E402

RAD_EMU_CKPT = os.environ.get('RAD_EMU_CKPT',
                              os.path.join(_paths.INPUTS, 'rad_emulator', 'current'))
RAD_EMU_ANOM = os.environ.get('RAD_EMU_ANOM', 'net').lower()
# The TOA forcing DIAGNOSTIC (arf_toa): 'emu' (default) takes it from the
# networks every step, 'exact' from the physics A/B samples (RAD_REF=rrtmgp
# every RAD_REF_EVERY steps; the other steps report NaN, which coupling.py's
# trailing window skips). 'exact' was the default while the networks' ARF bias
# was ~1 % offline (a 10-30 % forcing error early in a run), but with
# RAD_REF_EVERY=3 the 4-FINITE-SAMPLE ARF_AVG_H window then spans 72 h instead
# of the physics run's 24 h, and while the forcing is still growing the
# reported arf_toa_avg reads ~12 % weaker than a physics run's at the same
# hour although the samples themselves agree to 0.5 % (2026-10-16,
# docs/RADIATION.md "The forcing diagnostic"). Since v6 the networks' sampled
# ARF is within 1 % of the physics, so the default is the networks, sampled
# every step like a physics run; the exact samples stay in rad_bench_<TAG>.npz
# (arf_emu / arf_phys) for validation. RAD_EMU_ARF=exact restores the old
# behaviour (coupling.py then warns about the window).
RAD_EMU_ARF = os.environ.get('RAD_EMU_ARF', 'emu').lower()
# Output sanity clamp [K/day]: the networks' heating terms are clipped to
# +-RAD_EMU_CLIP (TOA deltas to +-25x that in W/m2) and NaN -> 0, and the run
# logs how many cells it touched. A residual MLP without normalisation can
# blow up on an input far outside its training set (2026-10-14: an aerosol
# net trained at lr 1e-3 returned 674 K/day in a column whose upper-
# troposphere humidity sat at -16 sigma, and dT_rad went to NaN in two steps);
# the physics never exceeds ~8 K/day even in the x10-augmented labels, so a
# clip at 20 is far outside anything legitimate and is a no-op on a healthy
# run. 0 disables the clamp (byte-for-byte the earlier behaviour).
RAD_EMU_CLIP = float(os.environ.get('RAD_EMU_CLIP', '20'))
if RAD_EMU_ANOM not in ('net', 'phys'):
    raise SystemExit(f"RAD_EMU_ANOM={RAD_EMU_ANOM!r} is not valid: net | phys")


class ColumnNet:
    """One exported network (train_rad_mlp.py .npz) as a jitted JAX function
    on raw (n, NF) features -> (n, NT) outputs in K/day and W/m2."""

    def __init__(self, path):
        z = np.load(path)
        self.path = path
        self.net = str(z['net']); self.nzb = int(z['nzb']); self.nf = int(z['nf'])
        self.nblocks = int(z['blocks']); self.width = int(z['width'])
        self.cols = jnp.asarray(z['cols'])
        p = dict(x_mean=z['x_mean'], x_std=z['x_std'], y_scale=z['y_scale'], sw_mask=z['sw_mask'],
                 W0=z['inp.W'], b0=z['inp.b'], Wo=z['out.W'], bo=z['out.b'])
        for j in range(self.nblocks):
            for k in ('W1', 'b1', 'W2', 'b2'):
                p[f'b{j}.{k}'] = z[f'b{j}.{k}']
        self.p = {k: jnp.asarray(np.asarray(v, np.float32)) for k, v in p.items()}
        self.g_scale = float(z['g_scale'])
        self.dt_shape = int(z['dt_shape']) if 'dt_shape' in z.files else 0
        self.head = str(z['head']) if 'head' in z.files else 'gate'
        self.sw_cosz = int(z['sw_cosz']) if 'sw_cosz' in z.files else 0
        self.tau_w = float(z['tau_w']) if 'tau_w' in z.files else 0.0
        self.dt_eps = float(z['dt_eps']) if 'dt_eps' in z.files else 1e-6
        self.x_clip = float(z['x_clip']) if 'x_clip' in z.files else 0.0
        self.tau_ref = float(z['tau_ref']) if 'tau_ref' in z.files else 1e-2
        self.taucol_ref = float(z['taucol_ref']) if 'taucol_ref' in z.files else 1e-1
        self.val = json.loads(str(z['val'])) if 'val' in z.files else {}
        self.epoch = int(z['epoch']) if 'epoch' in z.files else -1
        with open(path, 'rb') as f:
            self.digest = hashlib.sha1(f.read()).hexdigest()[:10]
        L = F.layout(int(z['nzc']), self.nzb, 14, 16)
        self._tau_col = (L['tau_col'][0], L['tau_col'][1])
        self._tau = (L['tau'][0], L['tau'][1])
        self._ssa = (L['ssa'][0], L['ssa'][1]); self._g = (L['g'][0], L['g'][1])
        self.nb_sw, self.nb_lw = 14, 16
        self._dT = (L['dT'][0], L['dT'][1])
        self._cosz = L['sfc'][0] + 2
        nblk = self.nblocks
        aer = self.net == 'aer'
        tc0, tc1 = self._tau_col; d0, d1 = self._dT; cz = self._cosz; gs = self.g_scale
        dt_shape = self.dt_shape; head = self.head; t0, t1 = self._tau; nzb = self.nzb; sw_cosz = self.sw_cosz
        nb_sw, nb_lw = self.nb_sw, self.nb_lw; nb = nb_sw + nb_lw; tau_ref, taucol_ref = self.tau_ref, self.taucol_ref
        tau_w, dt_eps, x_clip = self.tau_w, self.dt_eps, self.x_clip; s0, s1 = self._ssa; g0, g1 = self._g

        def fwd(p, x):
            if aer:
                gate = x[:, tc0:tc1].mean(1) / gs
            else:
                gate = jnp.sqrt((x[:, d0:d1] ** 2).mean(1)) / gs
            day = jnp.maximum(x[:, cz], 0.0) if sw_cosz else (x[:, cz] > 0).astype(jnp.float32)
            ssa_raw = x[:, s0:s1]; g_raw = x[:, g0:g1]     # the cum head needs the unweighted values
            if aer and tau_w > 0:                      # optically empty layers carry no albedo information
                tau_raw = jnp.expm1(x[:, t0:t1]) * F.TAU0; w = tau_raw / (tau_raw + tau_w)
                x = x.at[:, s0:s1].multiply(w).at[:, g0:g1].multiply(w)
            xi = x[:, self.cols]
            if dt_shape:
                d = x[:, d0:d1]
                xi = jnp.concatenate([xi, d / (jnp.sqrt((d ** 2).mean(1, keepdims=True)) + dt_eps)], axis=1)
            xn = (xi - p['x_mean']) / p['x_std']
            if x_clip > 0:
                xn = jnp.clip(xn, -x_clip, x_clip)
            h = jax.nn.silu(xn @ p['W0'] + p['b0'])
            for j in range(nblk):
                h = h + (jax.nn.silu(h @ p[f'b{j}.W1'] + p[f'b{j}.b1']) @ p[f'b{j}.W2'] + p[f'b{j}.b2'])
            if head == 'mixed2':
                e = h @ p['Wo'] + p['bo']; n1 = nb_lw * nzb; n2 = n1 + nb_lw
                tau = jnp.expm1(x[:, t0:t1]) * F.TAU0 / tau_ref
                tc = jnp.expm1(x[:, tc0:tc1]) * F.TAU0_COL / taucol_ref
                hr_lw = (e[:, :n1] * tau[:, nb_sw*nzb:]).reshape(-1, nb_lw, nzb).sum(1)
                olr = (e[:, n1:n2] * tc[:, nb_sw:]).sum(1, keepdims=True)
                ys = p['y_scale']
                hr_sw = e[:, n2:n2+nzb] * gate[:, None] * ys[:nzb]
                swup = (e[:, n2+nzb:n2+nzb+nb_sw] * tc[:, :nb_sw]).sum(1, keepdims=True)
                y = jnp.concatenate([hr_sw, hr_lw, swup, olr], axis=1)
                y = y * (1.0 - p['sw_mask']) + y * p['sw_mask'] * day[:, None]
                return y
            if head == 'mixed':
                e = h @ p['Wo'] + p['bo']; n1 = nb_lw * nzb; n2 = n1 + nb_lw
                tau = jnp.expm1(x[:, t0:t1]) * F.TAU0 / tau_ref
                tc = jnp.expm1(x[:, tc0:tc1]) * F.TAU0_COL / taucol_ref
                hr_lw = (e[:, :n1] * tau[:, nb_sw*nzb:]).reshape(-1, nb_lw, nzb).sum(1)
                olr = (e[:, n1:n2] * tc[:, nb_sw:]).sum(1, keepdims=True)
                ys = p['y_scale']
                hr_sw = e[:, n2:n2+nzb] * gate[:, None] * ys[:nzb]
                swup = e[:, n2+nzb:n2+nzb+1] * gate[:, None] * ys[2*nzb:2*nzb+1]
                y = jnp.concatenate([hr_sw, hr_lw, swup, olr], axis=1)
                y = y * (1.0 - p['sw_mask']) + y * p['sw_mask'] * day[:, None]
                return y
            if head == 'cum':                          # aer: LW as mixed; SW from local + cumulative absorption/backscatter depths
                e = h @ p['Wo'] + p['bo']; n1 = nb_lw * nzb; n2 = n1 + nb_lw; n3 = n2 + 5 * nb_sw * nzb; n4 = n3 + nzb; n5 = n4 + nb_sw
                tau = jnp.expm1(x[:, t0:t1]) * F.TAU0 / tau_ref
                tc = jnp.expm1(x[:, tc0:tc1]) * F.TAU0_COL / taucol_ref
                hr_lw = (e[:, :n1] * tau[:, nb_sw*nzb:]).reshape(-1, nb_lw, nzb).sum(1)
                olr = (e[:, n1:n2] * tc[:, nb_sw:]).sum(1, keepdims=True)
                ys = p['y_scale']; m = nb_sw * nzb
                t_sw = tau[:, :m].reshape(-1, nb_sw, nzb); ssa = ssa_raw[:, :m].reshape(-1, nb_sw, nzb); gg_ = g_raw[:, :m].reshape(-1, nb_sw, nzb)
                A = t_sw * (1.0 - ssa); B = t_sw * ssa * (1.0 - gg_) * 0.5
                cA = jnp.cumsum(A, axis=2); cB = jnp.cumsum(B, axis=2)
                Cc = jnp.stack([A, cA - A, cA[:, :, -1:] - cA, cB - B, cB[:, :, -1:] - cB], 1)
                hr_sw = (e[:, n2:n3].reshape(-1, 5, nb_sw, nzb) * Cc).sum((1, 2)) + e[:, n3:n4] * gate[:, None] * ys[:nzb]
                swup = (e[:, n4:n5] * B.sum(2)).sum(1, keepdims=True) + e[:, n5:n5+1] * gate[:, None] * ys[2*nzb:2*nzb+1]
                y = jnp.concatenate([hr_sw, hr_lw, swup, olr], axis=1)
                y = y * (1.0 - p['sw_mask']) + y * p['sw_mask'] * day[:, None]
                return y
            if head == 'jac':                          # planck: Jacobian(state, dT) . dT + second-order residual
                e = h @ p['Wo'] + p['bo']; NT = p['y_scale'].shape[0]; n1 = NT * nzb
                J = e[:, :n1].reshape(-1, NT, nzb); d = x[:, d0:d1] / gs
                y = jnp.einsum('nok,nk->no', J, d) + e[:, n1:] * (gate * gate / (1.0 + gate))[:, None]
                y = y * (1.0 - p['sw_mask']) + y * p['sw_mask'] * day[:, None]
                return y * p['y_scale']
            if head.startswith('tau'):
                e = h @ p['Wo'] + p['bo']                                  # (n, nb*nzb + nb [+ NT])
                tau = jnp.expm1(x[:, t0:t1]) * F.TAU0 / tau_ref           # (n, nb*nzb) band-major
                tc = jnp.expm1(x[:, tc0:tc1]) * F.TAU0_COL / taucol_ref   # (n, nb)
                prod = (e[:, :nb*nzb] * tau).reshape(-1, nb, nzb)
                pc = e[:, nb*nzb:nb*nzb+nb] * tc
                y = jnp.concatenate([prod[:, :nb_sw].sum(1), prod[:, nb_sw:].sum(1),
                                     pc[:, :nb_sw].sum(1, keepdims=True), pc[:, nb_sw:].sum(1, keepdims=True)], axis=1)
                if head == 'tau+gate':
                    y = y + e[:, nb*nzb+nb:] * gate[:, None] * p['y_scale']
            else:
                y = (h @ p['Wo'] + p['bo']) * gate[:, None]
            y = y * (1.0 - p['sw_mask']) + y * p['sw_mask'] * day[:, None]
            return y * p['y_scale']
        self._fwd = jax.jit(fwd)

    def __call__(self, x):
        return self._fwd(self.p, x.astype(jnp.float32))


class EnsembleNet:
    """Mean of several exported networks of the same kind (seed ensemble): the
    members' outputs are averaged in K/day and W/m2.  Presents the ColumnNet
    attributes the adapter reads (path, digest, width, blocks, epoch, val, head...)."""

    def __init__(self, paths):
        self.members = [ColumnNet(p) for p in paths]
        m0 = self.members[0]
        for m in self.members[1:]:
            assert (m.net, m.nzb, m.nf) == (m0.net, m0.nzb, m0.nf), f'ensemble member {m.path} differs in net/nzb/nf'
        self.path = ','.join(os.path.basename(p) for p in paths)
        self.net, self.nzb, self.nf = m0.net, m0.nzb, m0.nf
        self.width, self.nblocks, self.head, self.tau_w = m0.width, m0.nblocks, m0.head, m0.tau_w
        self.epoch = m0.epoch; self.val = m0.val
        self.digest = hashlib.sha1('|'.join(m.digest for m in self.members).encode()).hexdigest()[:10]
        self.n_members = len(self.members)

    def __call__(self, x):
        x = x.astype(jnp.float32)
        y = self.members[0]._fwd(self.members[0].p, x)
        for m in self.members[1:]:
            y = y + m._fwd(m.p, x)
        return y / self.n_members


def load_net(ckpt, name):
    """<ckpt>/<name>.npz alone -> ColumnNet; with <name>_*.npz members beside it
    (e.g. aer.npz + aer_s1.npz) -> EnsembleNet over all of them."""
    paths = sorted(glob.glob(os.path.join(ckpt, f'{name}.npz')) + glob.glob(os.path.join(ckpt, f'{name}_*.npz')))
    if not paths:
        raise SystemExit(f"RAD_ENGINE=emulator: no {name}.npz under RAD_EMU_CKPT={ckpt}\n"
                         "  train one with scripts/radiation/train_rad_mlp.py (docs/RADIATION.md)")
    return ColumnNet(paths[0]) if len(paths) == 1 else EnsembleNet(paths)


class RadEmulator:
    """Drop-in for RadiationDriver at the call sites coupling.py uses
    (heating, heating_anomaly, ref_ready), the physics driver underneath for
    everything that is not the solve."""

    def __init__(self, drv, ckpt=None, ref_engine='', ref_every=1, tag='run', anom=None):
        self.drv = drv
        if drv.ref_cache is None:
            raise SystemExit("RAD_ENGINE=emulator needs the reference cache (RAD_REF_CACHE != 0): "
                             "the clear-sky fields it builds on live there")
        ckpt = ckpt or RAD_EMU_CKPT
        self.aer = load_net(ckpt, 'aer')          # ColumnNet, or EnsembleNet when aer_*.npz members sit beside aer.npz
        self.planck = load_net(ckpt, 'planck')
        assert self.aer.nzb == drv.nzb == self.planck.nzb, 'checkpoint band levels != driver band'
        self.anom = anom or RAD_EMU_ANOM
        self.net_key = f'net_{self.aer.digest}'
        self.ref_engine = ref_engine; self.ref_every = max(1, int(ref_every)); self.tag = tag
        self.arf_exact = RAD_EMU_ARF == 'exact'
        if self.arf_exact and ref_engine != 'rrtmgp':
            raise SystemExit('RAD_EMU_ARF=exact needs RAD_REF=rrtmgp (the physics samples it reports)')
        self.ncall = 0; self.bench = []; self.n_clamped = 0
        self._w_lat = np.cos(np.deg2rad(drv.lat))[:, None] * np.ones((drv.nlat, drv.nlon))
        self.nzb = drv.nzb
        self.layout = F.layout(drv.nzc, drv.nzb, drv.nb_sw, drv.nb_lw)
        def _desc(c):
            n = getattr(c, 'n_members', 1)
            return f"{c.path} ({c.width}x{c.nblocks}, ep {c.epoch}" + (f", {n}-member ensemble)" if n > 1 else ")")
        print(f"  radiation emulator: aer {_desc(self.aer)}, planck {_desc(self.planck)}; "
              f"anomaly reference = {self.anom}; A/B vs {ref_engine or '-'} every {self.ref_every}; "
              f"TOA forcing diagnostic = {'exact physics samples' if self.arf_exact else 'networks'}", flush=True)
        for n, c in (('aer', self.aer), ('planck', self.planck)):
            if c.val:
                print(f"    {n} offline val (err/signal): " + '  '.join(
                    f"{k} {v[0]:.2e}/{v[1]:.2e}" for k, v in c.val.items() if k in ('SW', 'LW', 'swup', 'olr')), flush=True)

    def __getattr__(self, name):          # everything else is the physics driver's
        return getattr(self.drv, name)

    # ------------------------------------------------------------------
    def ref_ready(self, t):
        keys = rad_cache.REF_KEYS + rad_cache.CLEAR_KEYS
        if self.anom == 'net':
            keys = keys + (self.net_key,)
        return self.drv.ref_cache.has(t, keys)

    def _split(self, y, nx):
        """(nx*nlon, NT) -> sw (nzb,nx,nlon), lw (nzb,nx,nlon), up (nx,nlon), olr (nx,nlon)."""
        n = self.nzb; nlon = self.drv.nlon
        y = y.reshape(nx, nlon, -1)
        return (jnp.moveaxis(y[..., :n], -1, 0), jnp.moveaxis(y[..., n:2*n], -1, 0),
                y[..., 2*n], y[..., 2*n+1])

    def _evolved(self, t, num_band, T_band_override=None, with_planck=True):
        """The emulated heating() call. Returns (hr_band f64, diag, aer_parts)
        where aer_parts = (sw, lw, up, olr) of the aerosol network alone."""
        drv = self.drv
        h = F.prep_hour(drv, t)
        T_full = h['T'].copy()
        T_band = T_full[drv.band_pos].copy()
        if T_band_override is not None:
            T_full[drv.band_pos] = np.asarray(T_band_override)
            dT = (np.asarray(T_band_override) - T_band).astype(np.float64)
        else:
            dT = np.zeros_like(T_band)
        cl = rad_cache_build.ensure_clear(drv, t)
        wwt = F.wt_weights_for(drv, T_full, h['q'])
        if num_band is None:                      # the zero-aerosol call, as in heating()
            num_band = jnp.zeros((drv.nbins, drv.nzb, drv.nlat, drv.nlon))
        num_j = jnp.asarray(num_band)
        C = R.RAD_LAT_CHUNK
        nchunk = drv.ndev * int(np.ceil(drv.nlat / (drv.ndev * C)))
        C = int(np.ceil(drv.nlat / nchunk))
        ya_rows, yp_rows, aod_rows = [], [], []
        for i, a in enumerate(range(0, drv.nlat, C)):
            b = min(a + C, drv.nlat); k = i % drv.ndev
            put = lambda x: jax.device_put(x, drv.devices[k])
            nc = put(num_j[:, :, a:b])
            ww = put(wwt[:, :, a:b]) if wwt is not None else None
            tau_s, ssa_s, g_s, tau_l, ssa_l, g_l = F.optics_chunk(drv, nc, ww, k)
            x = F.column_features(put(T_full[:, a:b]), put(h['q'][:, a:b]), put(h['o3'][:, a:b]),
                                  put(h['ch4'][:, a:b]), put(h['ts'][a:b]), put(h['alb'][a:b]),
                                  put(h['zen'][a:b]), put(dT[:, a:b]),
                                  tau_s, ssa_s, g_s, tau_l, ssa_l, g_l)
            ya_rows.append(self._split(self.aer(x), b - a))
            if with_planck:
                yp_rows.append(self._split(self.planck(x), b - a))
            if drv.wet_optics:
                aod_rows.append(R._aod550_wet(nc, drv._dp_band_dev[k], drv._sig550_dev[k], ww))
            else:
                aod_rows.append(R._aod550(nc, drv._dp_band_dev[k], drv._sig550_dev[k]))
        d0 = drv.devices[0]
        cat = lambda rows, j, ax: jnp.concatenate([jax.device_put(r[j], d0) for r in rows], axis=ax)
        ya = (cat(ya_rows, 0, 1), cat(ya_rows, 1, 1), cat(ya_rows, 2, 0), cat(ya_rows, 3, 0))
        if with_planck:
            yp = (cat(yp_rows, 0, 1), cat(yp_rows, 1, 1), cat(yp_rows, 2, 0), cat(yp_rows, 3, 0))
        else:
            yp = tuple(jnp.zeros_like(v) for v in ya)
        if RAD_EMU_CLIP > 0:
            ya, n_a = self._clamp(ya); yp, n_p = self._clamp(yp)
            if n_a or n_p:
                self.n_clamped += n_a + n_p
                print(f"  [rad emulator] WARNING hour {t}: output clamp at +-{RAD_EMU_CLIP:g} K/day touched "
                      f"{n_a} aerosol-net and {n_p} Planck-net cells (run total {self.n_clamped}); "
                      f"an input far outside the training set -- see docs/RADIATION.md", flush=True)
        aod550 = jnp.concatenate([jax.device_put(x, d0) for x in aod_rows], axis=0)
        hr = (jnp.asarray(cl['hr_clear'], jnp.float64)
              + (ya[0] + ya[1] + yp[0] + yp[1]).astype(jnp.float64) / F.SEC_PER_DAY)
        diag = {'sw_dn_toa': jnp.asarray(cl['dc_sw_dn_toa'], jnp.float64),
                'sw_up_toa': jnp.asarray(cl['dc_sw_up_toa'], jnp.float64) + (ya[2] + yp[2]).astype(jnp.float64),
                'olr': jnp.asarray(cl['dc_olr'], jnp.float64) + (ya[3] + yp[3]).astype(jnp.float64),
                'sw_dn_sfc': jnp.asarray(cl['dc_sw_dn_sfc'], jnp.float64),
                'lw_dn_sfc': jnp.asarray(cl['dc_lw_dn_sfc'], jnp.float64),
                'aod550': aod550}
        return hr, diag, ya, cl

    def _clamp(self, y):
        """(sw, lw, up, olr) -> the same with NaN -> 0 and |.| clipped to the
        sanity bound (heating in K/day, TOA deltas in W/m2); plus the count of
        cells that were outside it."""
        lim = (RAD_EMU_CLIP, RAD_EMU_CLIP, 25.0 * RAD_EMU_CLIP, 25.0 * RAD_EMU_CLIP)
        out = []; n = 0
        for v, l in zip(y, lim):
            bad = ~(jnp.abs(v) <= l)                       # NaN counts as bad
            n += int(bad.sum())
            out.append(jnp.clip(jnp.nan_to_num(v, nan=0.0, posinf=l, neginf=-l), -l, l))
        return tuple(out), n

    def heating(self, t, num_band, T_band_override=None):
        hr, diag, _, _ = self._evolved(t, num_band, T_band_override)
        return hr, diag

    # ------------------------------------------------------------------
    def reference_net(self, t, num_ref):
        """The aerosol network on the reference aerosol at T_CESM, from the
        cache (keyed by the checkpoint) or computed from num_ref and stored.
        Also makes sure the exact reference entry exists (for d0's AOD)."""
        c = self.drv.ref_cache
        e = c.get(t, (self.net_key, 'd0_aod550', 'd0_sw_dn_toa'))
        if e is None:
            if num_ref is None:
                raise RuntimeError(f'rad_emulator: hour {t} has no cached reference and no num_ref was given')
            _, _, ya, _ = self._evolved(t, num_ref, None, with_planck=False)
            if not self.drv.ref_ready(t):
                self.drv.reference(t, num_ref)          # exact physics reference, stored
            arr = np.concatenate([np.asarray(ya[0]), np.asarray(ya[1]),
                                  np.asarray(ya[2])[None], np.asarray(ya[3])[None]], axis=0).astype(np.float32)
            c.put(t, **{self.net_key: arr})
            e = c.get(t, (self.net_key, 'd0_aod550', 'd0_sw_dn_toa'))
            if e is None:
                # concurrent runs computing the same uncached hour: rad_cache.put is a
                # read-merge-replace, so another run's put can replace the entry
                # without our key (2026-10-21, val_legacy_L2 step 1599). Use the
                # array just computed; take d0's AOD from the cache (retry) or the
                # reference itself, and re-store our key.
                for _ in range(5):
                    e = c.get(t, ('d0_aod550',))
                    if e is not None:
                        break
                    time.sleep(1.0)
                if e is None:
                    _, d0 = self.drv.reference(t, num_ref)
                    e = {'d0_aod550': np.asarray(d0['aod550'])}
                e = dict(e); e[self.net_key] = arr
                c.put(t, **{self.net_key: arr})
        n = self.nzb; arr = e[self.net_key]
        return (arr[:n], arr[n:2*n], arr[2*n], arr[2*n+1]), e['d0_aod550']

    def heating_anomaly(self, t, num_evolved, num_ref, dT_band, T_band=None):
        """Same contract as RadiationDriver.heating_anomaly. T_band (the
        band temperature of both calls when it is not the archive's,
        coupling.py MET_SRC=file) is only supported with RAD_EMU_ANOM=phys:
        the network reference (RAD_EMU_ANOM=net) is built on the cached
        clear-sky solve at the ARCHIVE temperature, so it cannot represent a
        reference at another one. The Planck network still sees
        dT = T_band - T_archive + dT_rad, i.e. a temperature difference it was
        not trained on; RAD_REF=rrtmgp is the instrument that measures that."""
        drv = self.drv
        t_e = time.time()
        if T_band is not None and self.anom != 'phys':
            raise SystemExit('RAD_ENGINE=emulator with a non-archive band temperature '
                             '(MET_SRC=file) needs RAD_EMU_ANOM=phys')
        T_cesm_band = (drv._read3d(drv.dT, 'T', t)[drv.band_pos] if T_band is None
                       else np.asarray(T_band, np.float64))
        hr1, d1, ya1, cl = self._evolved(t, num_evolved, T_cesm_band + np.asarray(dT_band))
        if self.anom == 'phys':
            hr0, d0 = drv.reference(t, num_ref, T_band=T_band)
        else:
            ya0, aod0 = self.reference_net(t, num_ref)
            hr0 = (jnp.asarray(cl['hr_clear'], jnp.float64)
                   + jnp.asarray(ya0[0] + ya0[1], jnp.float64) / F.SEC_PER_DAY)
            d0 = {'sw_dn_toa': d1['sw_dn_toa'],
                  'sw_up_toa': jnp.asarray(cl['dc_sw_up_toa'], jnp.float64) + jnp.asarray(ya0[2], jnp.float64),
                  'olr': jnp.asarray(cl['dc_olr'], jnp.float64) + jnp.asarray(ya0[3], jnp.float64),
                  'sw_dn_sfc': d1['sw_dn_sfc'], 'lw_dn_sfc': d1['lw_dn_sfc'],
                  'aod550': jnp.asarray(aod0)}
        hr_anom = hr1 - hr0
        jax.block_until_ready(hr_anom); t_e = time.time() - t_e
        self.ncall += 1
        if self.ref_engine == 'rrtmgp' and (self.ncall - 1) % self.ref_every == 0:
            d1p, d0p = self._ab(t, num_evolved, num_ref, dT_band, T_cesm_band, hr1, d1, hr0, d0, hr_anom, t_e,
                                T_band=T_band)
            if self.arf_exact:                 # the diagnostic fluxes are the physics' on this step
                for k in ('sw_up_toa', 'olr'):
                    d1[k] = d1p[k]; d0[k] = d0p[k]
        elif self.arf_exact:                   # no exact sample this step: report none
            nan = jnp.full_like(d1['olr'], jnp.nan)
            d1 = dict(d1); d0 = dict(d0)
            for k in ('sw_up_toa', 'olr'):
                d1[k] = nan; d0[k] = nan
        return hr_anom, d1, d0

    # ------------------------------------------------------------------
    def _ab(self, t, num_evolved, num_ref, dT_band, T_cesm_band, hr1, d1, hr0, d0, hr_anom, t_e,
            T_band=None):
        """Physics on the same input; per-step comparison to rad_bench_<TAG>.npz."""
        drv = self.drv
        t_p = time.time()
        hr1p, d1p = drv.heating(t, num_evolved, T_band_override=T_cesm_band + np.asarray(dT_band))
        hr0p, d0p = drv.reference(t, num_ref, T_band=T_band)
        anp = hr1p - hr0p
        jax.block_until_ready(anp); t_p = time.time() - t_p
        D = F.SEC_PER_DAY; w = self._w_lat
        e1 = (np.asarray(hr1) - np.asarray(hr1p)) * D; ea = (np.asarray(hr_anom) - np.asarray(anp)) * D
        sig1 = (np.asarray(hr1p) - np.asarray(drv.ref_cache.get(t, ('hr_clear',))['hr_clear'])) * D   # aerosol+Planck part
        siga = np.asarray(anp) * D
        def gm(x):
            return float((np.asarray(x) * w).sum() / w.sum())
        up1 = np.asarray(d1['sw_up_toa']) + np.asarray(d1['olr']); up0 = np.asarray(d0['sw_up_toa']) + np.asarray(d0['olr'])
        up1p = np.asarray(d1p['sw_up_toa']) + np.asarray(d1p['olr']); up0p = np.asarray(d0p['sw_up_toa']) + np.asarray(d0p['olr'])
        rec = dict(step=self.ncall - 1, hour=int(t), t_emu=t_e, t_phys=t_p,
                   hr1_rms=np.sqrt((e1**2).mean()), hr1_max=np.abs(e1).max(),
                   hr1_sig_rms=np.sqrt((sig1**2).mean()),
                   anom_rms=np.sqrt((ea**2).mean()), anom_max=np.abs(ea).max(), anom_sig_rms=np.sqrt((siga**2).mean()),
                   anom_lvl_rms=np.sqrt((ea**2).mean((1, 2))), anom_lvl_sig=np.sqrt((siga**2).mean((1, 2))),
                   anom_zm_err=ea.mean(2), anom_zm_sig=siga.mean(2),
                   anom_zm_abs_err=np.abs(ea).mean(2),
                   arf_emu=gm(up0 - up1), arf_phys=gm(up0p - up1p),
                   olr1_err_gm=gm(np.asarray(d1['olr']) - np.asarray(d1p['olr'])),
                   swup1_err_gm=gm(np.asarray(d1['sw_up_toa']) - np.asarray(d1p['sw_up_toa'])),
                   aod_gm=gm(d1['aod550']))
        self.bench.append(rec)
        print(f"  [rad A/B] s={rec['step']} hour {t}: emu {t_e:.2f}s phys {t_p:.2f}s | hr1 err rms {rec['hr1_rms']:.2e} "
              f"max {rec['hr1_max']:.2e} K/d (aer+Planck signal rms {rec['hr1_sig_rms']:.2e}) | anomaly err rms "
              f"{rec['anom_rms']:.2e} max {rec['anom_max']:.2e} (signal rms {rec['anom_sig_rms']:.2e}, "
              f"{rec['anom_rms']/max(rec['anom_sig_rms'],1e-30):.1%}) | ARF_toa emu {rec['arf_emu']:+.4f} phys {rec['arf_phys']:+.4f} W/m2",
              flush=True)
        keys = [k for k in rec if k not in ('anom_zm_err', 'anom_zm_sig', 'anom_zm_abs_err', 'anom_lvl_rms', 'anom_lvl_sig')]
        out = {k: np.array([r[k] for r in self.bench]) for k in keys}
        for k in ('anom_zm_err', 'anom_zm_sig', 'anom_zm_abs_err', 'anom_lvl_rms', 'anom_lvl_sig'):
            out[k] = np.stack([r[k] for r in self.bench])
        out['plev_pa'] = drv.p_lay[drv.band_pos]; out['lat'] = drv.lat
        out['anom_mode'] = self.anom; out['ckpt_aer'] = self.aer.path; out['ckpt_planck'] = self.planck.path
        tmp = f'rad_bench_{self.tag}.tmp.npz'
        np.savez(tmp, **out); os.replace(tmp, f'rad_bench_{self.tag}.npz')
        return d1p, d0p
