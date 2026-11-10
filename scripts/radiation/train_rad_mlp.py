#!/usr/bin/env python3
"""Train the radiation emulator's two column networks on label shards from
scripts/radiation/label_columns.py (docs/RADIATION_EMULATOR_SCOPE.md §4.5).

  --net aer     the AEROSOL network: hr1(optics, T_eff) - hr_clear(T_eff) on the
                band levels (SW, LW) and the two TOA deltas. Inputs: every
                feature of rad_features.py. Output multiplied by a column-
                optical-depth gate, so an aerosol-free column gives EXACTLY 0.
  --net planck  the PLANCK network: hr_clear(T_eff) - hr_clear(T_CESM) and the
                TOA deltas, i.e. the solver's response to the prognostic dT
                (non-linear at the 0.5 K scale through RRTMGP's WENO5 face
                temperatures -- scripts/radiation/check_planck_jacobian.py --
                hence a network and not a cached Jacobian). Inputs: the gas
                column, surface, dT. Output multiplied by rms(dT), so dT = 0
                gives EXACTLY 0.
Both: shortwave rows forced to 0 at night. Residual MLP, per-feature
standardisation, per-output rms scaling, MSE, AdamW + cosine. Exports a .npz
the JAX adapter (src/radiation/rad_emulator.py) evaluates: weights, norms,
layout, gate spec. The whole training set lives on the GPU, or in pageable host
memory under --x-cpu (standardised in place; nothing is bulk-pinned -- see load()).

  python3 train_rad_mlp.py --net aer --train DIR [DIR ...] --val DIR --out CKPT.npz
"""
import os, sys, glob, time, json, argparse, hashlib
import numpy as np
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, 'src'))
import _paths                      # noqa: E402
import rad_features as F

ap = argparse.ArgumentParser()
ap.add_argument('--net', required=True, choices=['aer', 'planck'])
ap.add_argument('--train', nargs='+', required=True)
ap.add_argument('--val', nargs='+', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--width', type=int, default=None); ap.add_argument('--blocks', type=int, default=None)
ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--bs', type=int, default=4096)
ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--wd', type=float, default=1e-5)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--max-shards', type=int, default=None)
ap.add_argument('--loss', default='rms', choices=['rms', 'abs'], help='rms: per-output rms-scaled MSE; abs: K/day + W/m2 (level-uniform)')
ap.add_argument('--max-dt', type=float, default=3.0, help='drop columns whose |dT| exceeds this [K] (a 30-day run reaches 0.8 K)')
ap.add_argument('--head', default=None, choices=['gate', 'tau', 'tau+gate', 'mixed', 'mixed2', 'jac', 'cum'], help='output head: cum (aer) = mixed with the SW heating as per-band, per-level efficiencies contracted with the local absorption and the CUMULATIVE absorption / backscatter optical depths above and below the level (the thin-plume SW response is linear in those) plus the gated free term; jac (planck) = state-dependent Jacobian contracted with dT plus a second-order gated residual (the solve is 96 %% linear in dT at the 30-day scale); gate = MLP x scalar gate; tau = per-band-level efficiency contracted with the exact band optical depths (default: tau for aer, gate for planck)')
ap.add_argument('--sw-cosz', type=int, default=0, help='mixed head: multiply the SW outputs by cos(zenith) as well (the available solar flux)')
ap.add_argument('--tau-w', type=float, default=3e-7, help='aer: weight the ssa/g inputs by tau/(tau+TAU_W) so optically empty layers carry no albedo information (a polar nucleation burst put the LW ssa 100 sigma out of distribution at tau ~ 1e-10); 0 = off')
ap.add_argument('--dt-eps', type=float, default=0.02, help='planck: dT shape = dT/(rms dT + DT_EPS) [K], so noise-like tiny profiles do not become unit shapes')
ap.add_argument('--bias-w', type=float, default=0.0, help='weight of a penalty on the batch-mean error per output (kills global-mean bias, e.g. of the TOA deltas)')
ap.add_argument('--x-clip', type=float, default=10.0, help='clip standardised inputs at +-X_CLIP sigma (0 = off)')
ap.add_argument('--max-aod', type=float, default=None, help='drop TRAINING columns whose (augmented) 550 nm AOD exceeds this (the x10 scaling of a year-long state makes AOD > 1 columns whose tau-contracted outputs blow the loss up and whose heating dominates the per-output rms scale)')
ap.add_argument('--x-cpu', action='store_true', help='keep the TRAINING inputs in PAGEABLE host memory and ship each batch through a one-batch pinned staging buffer (a set larger than the GPU; exact). Never bulk-pins: pinned pages are unreclaimable and outside cgroup accounting -- two trainers pinning 160+93 GB each hung the 1 TB node on 2026-10-14 and 2026-10-15')
ap.add_argument('--dt-shape', type=int, default=None, help='1: add dT/rms(dT) (the profile SHAPE) to the inputs (default: 1 for planck, 0 for aer)')
args = ap.parse_args()
import torch, torch.nn as nn
torch.manual_seed(args.seed); np.random.seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
dev = torch.device(os.environ.get('RAD_TRAIN_DEVICE', 'cuda'))   # RAD_TRAIN_DEVICE=cpu for a code check without a GPU

def _mem_available_gb():
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'): return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return float('inf')

def load(dirs, max_shards=None, max_dt=None, max_aod=None, what='train', g_cols=None):
    """Stream the label shards into ONE preallocated array per field, applying the row filters
    (|dT| <= max_dt, AOD <= max_aod) as each shard arrives. g_cols(L, nzb) -> columns of the
    per-row gate features that will also live on the host, for the memory guard.
    2026-10-16: the old list -> np.concatenate -> boolean-index version held the data 3x over
    (3 x 160 GB on the v1-v3 set) before training even started; with the pinned copies on top,
    two trainers took the 1 TB node down (2026-10-14 SIGBUS at 181 GB, 2026-10-15 hard hang).
    Now: one pass over the small `meta` members to size the arrays, a refusal if the result
    would not fit in MemAvailable, then one streaming pass over `x`."""
    files = sorted(sum((glob.glob(os.path.join(d, 'g*.npz')) for d in dirs), []))
    if max_shards: files = files[:max_shards]
    rows, good = [], []
    for f in files:                                   # pass 1: row counts + layout (x untouched)
        try:
            z = np.load(f); n = z['meta'].shape[0]; nzb = int(z['nzb']); nf = int(z['nf'])
            nraw = z['raw'].shape[1]; nmeta = z['meta'].shape[1]
        except Exception as e:                        # a shard still being written by the labeler
            print(f'  skipping {f}: {e}', flush=True); continue
        rows.append(n); good.append(f)
    N = sum(rows)
    nzc = (nf - 3 - nzb - 30 * nzb * 3 - 30) // 4
    L_ = F.layout(nzc, nzb, 14, 16); dT = L_['dT']
    need = (N * (nf + nraw) * 4 + N * nmeta * 8 + (N * g_cols(L_, nzb) * 4 if g_cols else 0)) / 1e9
    avail = _mem_available_gb()
    print(f'  {what}: {len(good)} shards, {N} rows -> {need:.0f} GB host; {avail:.0f} GB available', flush=True)
    if need > 0.8 * avail:
        raise SystemExit(f'  REFUSING to load {what}: {need:.0f} GB needed > 80 % of the {avail:.0f} GB available. '
                         f'The node has 1 TB and no swap; exceeding it hangs the machine. Wait for other jobs or use --max-shards.')
    x = np.empty((N, nf), np.float32); raw = np.empty((N, nraw), np.float32); meta = np.empty((N, nmeta), np.float64)
    n = n_dt = n_aod = n_all = 0
    for f in good:                                    # pass 2: stream the rows in, keeping the filtered ones
        try:
            z = np.load(f); xs = z['x']; rs = z['raw']; ms = z['meta']
        except Exception as e:
            print(f'  skipping {f}: {e}', flush=True); continue
        k = np.ones(len(xs), bool)
        if max_dt is not None: k &= np.abs(xs[:, dT[0]:dT[1]]).max(1) <= max_dt
        n_dt += int(k.sum())
        if max_aod is not None:
            ka = ms[:, 6] <= max_aod; n_aod += int(ka.sum()); k &= ka
        m = int(k.sum()); n_all += len(xs)
        x[n:n+m] = xs[k]; raw[n:n+m] = rs[k]; meta[n:n+m] = ms[k]; n += m
    if max_dt is not None: print(f'|dT| <= {max_dt} K keeps {n_dt/max(n_all,1):.1%} of {what}', flush=True)
    if max_aod is not None: print(f'AOD <= {max_aod} keeps {n_aod/max(n_all,1):.1%} of {what}', flush=True)
    return x[:n], raw[:n], meta[:n], nzb, nf, len(good)

t0 = time.time()
def _g_cols(L_, nzb_):   # gate features that stay on the host under --x-cpu (only the cum head's G; the others fit the GPU)
    if not (args.x_cpu and args.net == 'aer' and args.head == 'cum'): return 0
    nb = L_['nb_sw'] + L_['nb_lw']
    return nb * nzb_ + nb + 1 + 2 * L_['nb_sw'] * nzb_
xtr, rtr, mtr, nzb, nf, nftr = load(args.train, args.max_shards, max_dt=args.max_dt, max_aod=args.max_aod, what='train', g_cols=_g_cols)
xva, rva, mva, _, _, nfva = load(args.val, max_dt=args.max_dt, what='val')      # val: |dT| filter only (AOD untouched)
print(f'train {xtr.shape} from {nftr} shards, val {xva.shape} from {nfva} shards; NF={nf} nzb={nzb}; load {time.time()-t0:.0f}s', flush=True)
nzc = (nf - 3 - nzb - 30 * nzb * 3 - 30) // 4
L = F.layout(nzc, nzb, 14, 16); assert L['NF'] == nf, (L['NF'], nf)

def targets(raw):
    n = nzb
    sw1, lw1, swe, lwe, swc, lwc = (raw[:, i*n:(i+1)*n] for i in range(6))
    up1, olr1, upe, olre, upc, olrc = (raw[:, 144 + i] for i in range(6))
    if args.net == 'aer':
        return np.concatenate([sw1 - swe, lw1 - lwe, (up1 - upe)[:, None], (olr1 - olre)[:, None]], 1)
    return np.concatenate([swe - swc, lwe - lwc, (upe - upc)[:, None], (olre - olrc)[:, None]], 1)
# the |dT| and AOD row filters are applied inside load() as the shards stream in (no boolean-index copy of the set)
ytr = targets(rtr).astype(np.float32); yva = targets(rva).astype(np.float32)
NT = ytr.shape[1]
# input columns per net
if args.net == 'aer':
    cols = np.arange(nf)
else:
    cols = np.concatenate([np.arange(*L[k]) for k in ('T', 'logq', 'logo3', 'logch4', 'sfc', 'dT')])
cosz_col = L['sfc'][0] + 2
tau_col = np.arange(*L['tau_col']); dT_col = np.arange(*L['dT'])
dt_shape = args.dt_shape if args.dt_shape is not None else int(args.net == 'planck')
head = args.head or ('tau+gate' if args.net == 'aer' else 'gate')
TAU_REF, TAUCOL_REF = 1e-2, 1e-1          # per-layer / column optical depth units of the tau head
nb_tot = L['nb_sw'] + L['nb_lw']
def tau_feats(x):
    """(n, nb*nzb + nb): band optical depths per level and per column, in head units."""
    t = np.expm1(x[:, L['tau'][0]:L['tau'][1]]) * F.TAU0 / TAU_REF
    tc = np.expm1(x[:, L['tau_col'][0]:L['tau_col'][1]]) * F.TAU0_COL / TAUCOL_REF
    gate = x[:, tau_col].mean(1, keepdims=True)                 # 0 iff tau == 0 everywhere
    return np.concatenate([t, tc, gate], 1).astype(np.float32)
def cum_feats(x):
    """tau_feats + the raw SW ssa and g per band level (n, 751 + 2*nb_sw*nzb): the cum head builds its cumulative
    absorption/backscatter depths from these on the fly."""
    s0 = L['ssa'][0]; g0 = L['g'][0]; m = L['nb_sw'] * nzb
    return np.concatenate([tau_feats(x), x[:, s0:s0+m], x[:, g0:g0+m]], 1).astype(np.float32)
def shape_feats(x):
    d = x[:, dT_col]; return d / (np.sqrt((d ** 2).mean(1, keepdims=True)) + args.dt_eps)
ssa_sl = slice(*L['ssa']); g_sl = slice(*L['g']); tau_sl = slice(*L['tau'])
def inputs_np(x):
    xi = x[:, cols].copy()
    if args.net == 'aer' and args.tau_w > 0:
        tau = np.expm1(x[:, tau_sl]) * F.TAU0; w = tau / (tau + args.tau_w)
        xi[:, ssa_sl] = x[:, ssa_sl] * w; xi[:, g_sl] = x[:, g_sl] * w     # cols == arange(nf) for aer
    return np.concatenate([xi, shape_feats(x)], 1) if dt_shape else xi

def gate_np(x):
    if args.net == 'aer':
        return x[:, tau_col].mean(1)                     # 0 iff tau == 0 everywhere
    return np.sqrt((x[:, dT_col] ** 2).mean(1))          # 0 iff dT == 0
gtr = gate_np(xtr); gva = gate_np(xva)
g_scale = float(gtr[gtr > 0].mean())
# standardisation
FRUGAL = args.net == 'aer' and not dt_shape     # inputs_np(x) is then x itself with ssa/g weighted: standardise IN PLACE, chunked
CH = 1 << 16
if not FRUGAL:
    _xi = inputs_np(xtr); x_mean = _xi.mean(0); x_std = _xi.std(0); x_std[x_std < 1e-6] = 1.0; del _xi
y_rms = np.sqrt((ytr ** 2).mean(0)); y_rms[y_rms < 1e-9] = 1e-9
y_scale = y_rms if args.loss == 'rms' else np.where(np.arange(NT) < 2 * nzb, y_rms[:2*nzb].mean(), y_rms[2*nzb:].mean())
print(f'targets: NT={NT}; signal rms per group: SW {y_rms[:nzb].mean():.4f} LW {y_rms[nzb:2*nzb].mean():.4f} K/d, '
      f'TOA sw_up {y_rms[2*nzb]:.4f} olr {y_rms[2*nzb+1]:.4f} W/m2; gate mean {g_scale:.3f}; '
      f'frac zero-gate {np.mean(gtr == 0):.3f}', flush=True)

def to_dev(x, y, g, x_host=False):
    xn = (inputs_np(x) - x_mean) / x_std
    if args.x_clip > 0: xn = np.clip(xn, -args.x_clip, args.x_clip)
    if head in ('tau', 'tau+gate', 'mixed', 'mixed2'):
        gg = torch.from_numpy(tau_feats(x)).to(dev)
    elif head == 'cum':
        gg = torch.from_numpy(cum_feats(x))
        gg = gg.pin_memory() if x_host else gg.to(dev)
    elif head == 'jac':                                  # dT in units of g_scale, then the gate
        gg = torch.from_numpy(np.concatenate([x[:, dT_col] / g_scale, (g / g_scale)[:, None]], 1).astype(np.float32)).to(dev)
    else:
        gg = torch.from_numpy((g / g_scale).astype(np.float32)).to(dev)
    X = torch.from_numpy(np.ascontiguousarray(xn, dtype=np.float32))     # no astype copy; pageable on the host under x_host (never pinned)
    if not x_host: X = X.to(dev)
    return (X, torch.from_numpy((y / y_scale).astype(np.float32)).to(dev),
            gg, torch.from_numpy((x[:, cosz_col] > 0).astype(np.float32)).to(dev))
if FRUGAL:
    # ---- memory-frugal path (2026-10-16): gate features and day mask are taken from the RAW x chunk-wise, then the
    # set is weighted + standardised + clipped IN PLACE and handed to torch without a copy. Same float32 arithmetic
    # as inputs_np/to_dev; only the mean/std are accumulated in float64 over chunks instead of one float32 temporary.
    def _feats_chunked(fn, x, ncols):
        out = np.empty((len(x), ncols), np.float32)
        for i in range(0, len(x), CH): out[i:i+CH] = fn(x[i:i+CH])
        return out
    def _gate_feats(x, g, host):
        if head in ('tau', 'tau+gate', 'mixed', 'mixed2'):
            return torch.from_numpy(_feats_chunked(tau_feats, x, nb_tot * nzb + nb_tot + 1)).to(dev)
        if head == 'cum':
            G = torch.from_numpy(_feats_chunked(cum_feats, x, nb_tot * nzb + nb_tot + 1 + 2 * L['nb_sw'] * nzb))
            return G if host else G.to(dev)
        return torch.from_numpy((g / g_scale).astype(np.float32)).to(dev)
    def _weight_inplace(x):              # == inputs_np for the aer net (ssa/g scaled by tau/(tau+tau_w)), chunked, in place
        if args.tau_w > 0:
            for i in range(0, len(x), CH):
                tau = np.expm1(x[i:i+CH, tau_sl]) * F.TAU0; w = tau / (tau + args.tau_w)
                x[i:i+CH, ssa_sl] *= w; x[i:i+CH, g_sl] *= w
    def _mean_std(x):                    # two-pass, float64 accumulation, no full-size temporary
        s = np.zeros(x.shape[1])
        for i in range(0, len(x), CH): s += x[i:i+CH].sum(0, dtype=np.float64)
        mean = (s / len(x)).astype(np.float32); v = np.zeros(x.shape[1])
        for i in range(0, len(x), CH): v += ((x[i:i+CH] - mean) ** 2).sum(0, dtype=np.float64)
        return mean, np.sqrt(v / len(x)).astype(np.float32)
    Gtr = _gate_feats(xtr, gtr, host=args.x_cpu)
    Dtr = torch.from_numpy((xtr[:, cosz_col] if args.sw_cosz else (xtr[:, cosz_col] > 0)).astype(np.float32)).to(dev)
    Ytr = torch.from_numpy((ytr / y_scale).astype(np.float32)).to(dev)
    _weight_inplace(xtr); x_mean, x_std = _mean_std(xtr); x_std[x_std < 1e-6] = 1.0
    for i in range(0, len(xtr), CH):
        xtr[i:i+CH] -= x_mean; xtr[i:i+CH] /= x_std
        if args.x_clip > 0: np.clip(xtr[i:i+CH], -args.x_clip, args.x_clip, out=xtr[i:i+CH])
    Xtr = torch.from_numpy(xtr) if args.x_cpu else torch.from_numpy(xtr).to(dev)
    Xva, Yva, Gva, Dva = to_dev(xva, yva, gva)                       # val: the small copying path
    if args.sw_cosz: Dva = torch.from_numpy(xva[:, cosz_col].astype(np.float32)).to(dev)
else:
    Xtr, Ytr, Gtr, Dtr = to_dev(xtr, ytr, gtr, x_host=args.x_cpu); Xva, Yva, Gva, Dva = to_dev(xva, yva, gva)
    if args.sw_cosz:   # replace the day mask by cos(zenith) itself (0 at night): the SW rows scale with it
        Dtr = torch.from_numpy(xtr[:, cosz_col].astype(np.float32)).to(dev); Dva = torch.from_numpy(xva[:, cosz_col].astype(np.float32)).to(dev)
del xtr; print(f'{"host (pageable)" if args.x_cpu else "on GPU"}: {Xtr.numel()*4/1e9:.1f} GB train; G {Gtr.numel()*4/1e9:.1f} GB on {Gtr.device.type}; '
               f'{_mem_available_gb():.0f} GB host still available', flush=True)
if args.x_cpu:      # one-batch PINNED staging buffers: the set itself stays pageable
    _pin = dev.type == 'cuda'
    _st_x = torch.empty((args.bs, Xtr.shape[1]), dtype=torch.float32, pin_memory=_pin)
    _st_g = torch.empty((args.bs, Gtr.shape[1]), dtype=torch.float32, pin_memory=_pin) if Gtr.device.type == 'cpu' else None
    def _host_batch(T, idx_h, buf):      # gather the batch rows into the staging buffer and ship them; loss.item() syncs before reuse
        out = buf.narrow(0, 0, idx_h.numel()); torch.index_select(T, 0, idx_h, out=out)
        return out.to(dev, non_blocking=True)

width = args.width or (768 if args.net == 'aer' else 512)
blocks = args.blocks or (4 if args.net == 'aer' else 3)
class ResMLP(nn.Module):
    def __init__(self, nin, nout, w, nb):
        super().__init__()
        self.inp = nn.Linear(nin, w)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(w, w), nn.SiLU(), nn.Linear(w, w)) for _ in range(nb)])
        self.out = nn.Linear(w, nout)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
    def forward(self, x):
        h = torch.nn.functional.silu(self.inp(x))
        for b in self.blocks:
            h = h + b(h)
        return self.out(h)
n_in = len(cols) + (nzb if dt_shape else 0)
nb_lw = L['nb_lw']; nb_sw = L['nb_sw']
n_out = {'gate': NT, 'tau': nb_tot * nzb + nb_tot, 'tau+gate': nb_tot * nzb + nb_tot + NT,
         'mixed': nb_lw * nzb + nb_lw + nzb + 1,
         'mixed2': nb_lw * nzb + nb_lw + nzb + nb_sw,
         'jac': NT * nzb + NT,
         'cum': nb_lw * nzb + nb_lw + 5 * nb_sw * nzb + nzb + nb_sw + 1}[head]                      # J' (NT x nzb) . dT/g_scale  +  residual x gate^2/(1+gate)      # SW TOA delta as a tau_col contraction too
model = ResMLP(n_in, n_out, width, blocks).to(dev)
y_scale_t = torch.from_numpy(y_scale.astype(np.float32)).to(dev)
nb_sw = L['nb_sw']
sw_mask = torch.zeros(NT, device=dev); sw_mask[:nzb] = 1; sw_mask[2*nzb] = 1       # rows zeroed at night
def predict(X, G, D):
    if head == 'mixed2':
        e = model(X); n1 = nb_lw * nzb; n2 = n1 + nb_lw
        tau_lw = G[:, nb_sw*nzb:nb_tot*nzb]; tc_sw = G[:, nb_tot*nzb:nb_tot*nzb+nb_sw]; tc_lw = G[:, nb_tot*nzb+nb_sw:nb_tot*nzb+nb_tot]
        hr_lw = (e[:, :n1] * tau_lw).view(-1, nb_lw, nzb).sum(1)
        olr = (e[:, n1:n2] * tc_lw).sum(1, keepdim=True)
        gate = G[:, -1:] / g_scale
        hr_sw = e[:, n2:n2+nzb] * gate * y_scale_t[:nzb]
        swup = (e[:, n2+nzb:n2+nzb+nb_sw] * tc_sw).sum(1, keepdim=True)
        y = torch.cat([hr_sw, hr_lw, swup, olr], 1) / y_scale_t
        return y * (1 - sw_mask) + y * sw_mask * D[:, None]
    if head == 'jac':                   # planck: y = J(state, dT) . dT + R . gate^2/(1+gate)  (linear at small dT, free at large)
        e = model(X); n1 = NT * nzb
        J = e[:, :n1].view(-1, NT, nzb); d = G[:, :nzb]; gg = G[:, nzb:nzb+1]
        y = torch.bmm(J, d[:, :, None])[:, :, 0] + e[:, n1:] * (gg * gg / (1.0 + gg))
        return y * (1 - sw_mask) + y * sw_mask * D[:, None]
    if head == 'cum':                   # LW as mixed; SW: local absorption + cumulative absorption/backscatter above and below, plus a gated free term
        e = model(X); n1 = nb_lw * nzb; n2 = n1 + nb_lw; n3 = n2 + 5 * nb_sw * nzb; n4 = n3 + nzb; n5 = n4 + nb_sw
        tau_lw = G[:, nb_sw*nzb:nb_tot*nzb]; tc_lw = G[:, nb_tot*nzb+nb_sw:nb_tot*nzb+nb_tot]
        hr_lw = (e[:, :n1] * tau_lw).view(-1, nb_lw, nzb).sum(1)
        olr = (e[:, n1:n2] * tc_lw).sum(1, keepdim=True)
        o = nb_tot*nzb + nb_tot; gate = G[:, o:o+1] / g_scale
        t_sw = G[:, :nb_sw*nzb].view(-1, nb_sw, nzb); ssa = G[:, o+1:o+1+nb_sw*nzb].view(-1, nb_sw, nzb); gg_ = G[:, o+1+nb_sw*nzb:o+1+2*nb_sw*nzb].view(-1, nb_sw, nzb)
        A = t_sw * (1.0 - ssa); B = t_sw * ssa * (1.0 - gg_) * 0.5
        cA = A.cumsum(2); cB = B.cumsum(2)
        C = torch.stack([A, cA - A, cA[:, :, -1:] - cA, cB - B, cB[:, :, -1:] - cB], 1)     # (n, 5, nb_sw, nzb)
        hr_sw = (e[:, n2:n3].view(-1, 5, nb_sw, nzb) * C).sum((1, 2)) + e[:, n3:n4] * gate * y_scale_t[:nzb]
        swup = (e[:, n4:n5] * B.sum(2)).sum(1, keepdim=True) + e[:, n5:n5+1] * gate * y_scale_t[2*nzb:2*nzb+1]
        y = torch.cat([hr_sw, hr_lw, swup, olr], 1) / y_scale_t
        return y * (1 - sw_mask) + y * sw_mask * D[:, None]
    if head == 'mixed':                 # SW: gated non-local MLP; LW: tau-contracted local efficiencies
        e = model(X); n1 = nb_lw * nzb; n2 = n1 + nb_lw
        tau_lw = G[:, nb_sw*nzb:nb_tot*nzb]; tc_lw = G[:, nb_tot*nzb+nb_sw:nb_tot*nzb+nb_tot]
        hr_lw = (e[:, :n1] * tau_lw).view(-1, nb_lw, nzb).sum(1)
        olr = (e[:, n1:n2] * tc_lw).sum(1, keepdim=True)
        gate = G[:, -1:] / g_scale
        hr_sw = e[:, n2:n2+nzb] * gate * y_scale_t[:nzb]; swup = e[:, n2+nzb:n2+nzb+1] * gate * y_scale_t[2*nzb:2*nzb+1]
        y = torch.cat([hr_sw, hr_lw, swup, olr], 1) / y_scale_t
        return y * (1 - sw_mask) + y * sw_mask * D[:, None]
    if head.startswith('tau'):
        e = model(X)                                           # (n, nb*nzb + nb [+ NT]) efficiencies
        n1 = nb_tot * nzb; n2 = n1 + nb_tot
        prod = (e[:, :n1] * G[:, :n1]).view(-1, nb_tot, nzb)
        hr_sw = prod[:, :nb_sw].sum(1); hr_lw = prod[:, nb_sw:].sum(1)
        pc = e[:, n1:n2] * G[:, n1:n2]
        y = torch.cat([hr_sw, hr_lw, pc[:, :nb_sw].sum(1, keepdim=True), pc[:, nb_sw:].sum(1, keepdim=True)], 1)
        if head == 'tau+gate':                                 # non-local part, gated by the column optical depth
            y = y + e[:, n2:] * (G[:, n2:n2+1] / g_scale) * y_scale_t
        y = y / y_scale_t
    else:
        y = model(X) * G[:, None]
    return y * (1 - sw_mask) + y * sw_mask * D[:, None]
nparam = sum(p.numel() for p in model.parameters())
print(f'{args.net} net: {n_in} (dt_shape={dt_shape}) -> {width} x {blocks} blocks -> {n_out} (head={head}), {nparam/1e6:.2f} M params', flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
steps_per_epoch = (Xtr.shape[0] + args.bs - 1) // args.bs
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.05, anneal_strategy='cos', final_div_factor=100)

aod_va = mva[:, 6]; src_va = mva[:, 8]
def evaluate():
    model.eval()
    with torch.no_grad():
        P = torch.cat([predict(Xva[i:i+16384], Gva[i:i+16384], Dva[i:i+16384]) for i in range(0, Xva.shape[0], 16384)])
        err = (P - Yva).cpu().numpy() * y_scale; sig = yva
    model.train()
    out = {}
    for name, sl in (('SW', slice(0, nzb)), ('LW', slice(nzb, 2*nzb)), ('swup', slice(2*nzb, 2*nzb+1)), ('olr', slice(2*nzb+1, 2*nzb+2))):
        e = err[:, sl]; s = sig[:, sl]
        out[name] = (np.sqrt((e**2).mean()), np.sqrt((s**2).mean()), np.abs(e).max())
    # by AOD bin (aerosol relevance) -- heating rows only
    e = err[:, :2*nzb]; s = sig[:, :2*nzb]
    for lo, hi, nm in ((0, 1e-3, 'aod<1e-3'), (1e-3, 1e-2, '1e-3..1e-2'), (1e-2, 9, 'aod>1e-2')):
        m = (aod_va >= lo) & (aod_va < hi)
        if m.any(): out[nm] = (np.sqrt((e[m]**2).mean()), np.sqrt((s[m]**2).mean()), np.abs(e[m]).max())
    lvl = np.sqrt((err[:, :2*nzb]**2).mean(0)); lvs = np.sqrt((sig[:, :2*nzb]**2).mean(0))
    return out, lvl, lvs
def fmt(o):
    return '  '.join(f'{k}: {v[0]:.2e}/{v[1]:.2e} ({v[0]/max(v[1],1e-12):.1%}, max {v[2]:.1e})' for k, v in o.items())

best = float('inf'); log = []
for ep in range(args.epochs):
    te = time.time(); perm = torch.randperm(Xtr.shape[0], device=dev); tot = 0.0
    perm_h = perm.cpu() if args.x_cpu else None
    for i in range(0, Xtr.shape[0], args.bs):
        idx = perm[i:i+args.bs]
        if args.x_cpu:
            idx_h = perm_h[i:i+args.bs]
            Xb = _host_batch(Xtr, idx_h, _st_x)
            Gb = _host_batch(Gtr, idx_h, _st_g) if _st_g is not None else Gtr[idx]
        else:
            Xb = Xtr[idx]; Gb = Gtr[idx]
        err = predict(Xb, Gb, Dtr[idx]) - Ytr[idx]
        loss = (err ** 2).mean()
        if args.bias_w > 0:
            loss = loss + args.bias_w * (err.mean(0) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); tot += loss.item() * idx.numel()
    tr_loss = tot / Xtr.shape[0]
    o, lvl, lvs = evaluate()
    va_loss = sum((v[0]/v[1])**2 for k, v in o.items() if k in ('SW', 'LW', 'swup', 'olr'))
    print(f'ep {ep+1:3d} {time.time()-te:5.0f}s lr {sched.get_last_lr()[0]:.1e} train {tr_loss:.4e} | val err/signal: {fmt(o)}', flush=True)
    log.append(dict(ep=ep+1, train=tr_loss, val={k: [float(x) for x in v] for k, v in o.items()}))
    if va_loss < best:
        best = va_loss
        params = {}
        params['inp.W'] = model.inp.weight.detach().cpu().numpy().T; params['inp.b'] = model.inp.bias.detach().cpu().numpy()
        for j, b in enumerate(model.blocks):
            params[f'b{j}.W1'] = b[0].weight.detach().cpu().numpy().T; params[f'b{j}.b1'] = b[0].bias.detach().cpu().numpy()
            params[f'b{j}.W2'] = b[2].weight.detach().cpu().numpy().T; params[f'b{j}.b2'] = b[2].bias.detach().cpu().numpy()
        params['out.W'] = model.out.weight.detach().cpu().numpy().T; params['out.b'] = model.out.bias.detach().cpu().numpy()
        np.savez(args.out, net=args.net, cols=cols, x_mean=x_mean.astype(np.float32), x_std=x_std.astype(np.float32),
                 y_scale=y_scale.astype(np.float32), g_scale=np.float32(g_scale), nzb=nzb, nzc=nzc, nf=nf, width=width, blocks=blocks,
                 sw_mask=sw_mask.cpu().numpy(), epoch=ep+1, val=json.dumps(log[-1]['val']), dt_shape=dt_shape, max_dt=args.max_dt, head=head, sw_cosz=args.sw_cosz, tau_w=args.tau_w, dt_eps=args.dt_eps, x_clip=args.x_clip, bias_w=args.bias_w, tau_ref=TAU_REF, taucol_ref=TAUCOL_REF,
                 level_err=lvl, level_sig=lvs, train_dirs=json.dumps(args.train), **params)
        tag = ' *'
    else:
        tag = ''
    if tag: print(f'      saved {args.out} (val score {va_loss:.4e}); per-level err K/d: ' + ' '.join(f'{v:.1e}' for v in lvl[:2*nzb:3]), flush=True)
with open(args.out[:-4] + '_log.json', 'w') as f:
    json.dump(log, f)
print('TRAIN_DONE', flush=True)
