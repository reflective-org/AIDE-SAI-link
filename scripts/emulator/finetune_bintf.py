#!/usr/bin/env python3
"""Fine-tune the bin-token transformer on physics-labelled chains from the
coupled model's own states (scripts/emulator/label_chains.py output), warm-
started from the handoff checkpoint, with the original populations mixed in
so nothing the handoff measured is unlearned.

Adapted from /data/tomas-emulator/frontier_bintf_kchain.py (the script that
trained the handoff checkpoint): same architecture, same loss (MSE + totals
+ BW x burst-bin MSE over every 30-min output of a K-block chain), same
K-curriculum shape. What differs:
  * training chains = AIDE chains (over-sampled, weight AIDE_W) + a subsample
    of the original long-window populations (v9_sai, v8_carma_2wk,
    v9_sai_cont, v10_ideal), all truncated to the AIDE chain length;
  * TF1 (teacher-forced one-step loss from every true chain state) is on by
    default, because the deployment is one 6 h window at a time on states the
    coupled model advected -- that is the regime the chains sample;
  * MIX keeps the plain 6 h objective on the original populations alive;
  * evaluation adds a held-out AIDE set (stratum-resolved 6 h totals error
    and the chain-length self-rollout) next to the handoff's own all-sets
    table, and every epoch is checkpointed so the A/B harness can judge.

Env (defaults are the production fine-tune):
  CHAINS   glob of aide_chains_*.npz                      (required)
  HOLDOUT  glob of chains held out for evaluation          (default: 8 % of CHAINS)
  WARM     checkpoint to start from                        (handoff bintf_big_rebal avg_top3.pt)
  D, NL    model size (384, 6)                             TAG   output name suffix
  STAGES   "K:epochs:bs,..."                               (default "4:2:24,8:3:12")
  LR 3e-5  AIDE_W 3.0  ORIG_N 60000  TF1 0.5  MIX 0.5  BW 4  CAP (chains/epoch) 60000

    CUDA_VISIBLE_DEVICES=6 LD_LIBRARY_PATH=... CHAINS='/data/tomas-jax-output/aide_chains/*.npz' TAG=aide1 \
      /data/aide_emu_venv/bin/python scripts/emulator/finetune_bintf.py
Output: $EMU_ROOT/frontier_bintf_aide_<TAG>/{stage_K*_ep*.pt, stage_K*.pt, latest.pt, best_aide.pt}
"""
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

EMU_CODE = os.environ.get('EMU_CODE', '/data/tomas-emulator')
sys.path.insert(0, EMU_CODE)
from frontier_common import (norm, dev, BOX, EMU, N_BINS, Nsl, Msl, tot, totals_loss, log,      # noqa: E402
                             load_v8_set, load_multiset, MultiMixer, load_all_tests, eval_all_sets,
                             metrics6h, fmt, SMOKE)

CHAINS = os.environ.get('CHAINS', '')
HOLDOUT = os.environ.get('HOLDOUT', '')
WARM = os.environ.get('WARM', f"{EMU}/frontier_bintf_kchain_big_rebal/avg_top3.pt")
D_ = int(os.environ.get('D', '384')); NL_ = int(os.environ.get('NL', '6'))
TAG = os.environ.get('TAG', 'aide')
LR = float(os.environ.get('LR', '3e-5')); BW = float(os.environ.get('BW', '4'))
CW = float(os.environ.get('CW', '0'))      # extra weight on the COARSE bins (>= CB0) -- 2026-10-17: the day-30 forcing gap is the network's per-step bias in bins 21-32
CB0 = int(os.environ.get('CB0', '21'))
LAMM = float(os.environ.get('LAMM', '0.15')); LAMN = float(os.environ.get('LAMN', '0.35'))
AIDE_W = float(os.environ.get('AIDE_W', '3.0')); ORIG_N = int(os.environ.get('ORIG_N', '60000'))
TF1 = float(os.environ.get('TF1', '0.5')); MIX = float(os.environ.get('MIX', '0.5'))
CAP = int(os.environ.get('CAP', '60000')); SEED = int(os.environ.get('SEED', '0'))
STAGES_ENV = os.environ.get('STAGES', '4:2:24,8:3:12')
OUT = f"{EMU}/frontier_bintf_aide_{TAG}"; os.makedirs(OUT, exist_ok=True)
if not CHAINS:
    sys.exit("CHAINS=<glob of aide_chains_*.npz> is required")
log(f"TAG={TAG} WARM={WARM} D={D_} NL={NL_} LR={LR} AIDE_W={AIDE_W} ORIG_N={ORIG_N} TF1={TF1} MIX={MIX} BW={BW} CW={CW} CB0={CB0} STAGES={STAGES_ENV} CAP={CAP}")

src = open(f"{EMU_CODE}/frontier_transformers.py").read()
ns = {"torch": torch, "nn": nn, "N_BINS": N_BINS, "P": N_BINS + 2, "T": 12}
exec(src[src.index("def enc("):src.index("torch.manual_seed(0)")], ns); BinTF = ns["BinTF"]

# ---- data -----------------------------------------------------------------------
files = sorted(glob.glob(CHAINS))
if not files:
    sys.exit(f"no chain files match {CHAINS}")
rng = np.random.default_rng(SEED)
if HOLDOUT:
    hold = sorted(glob.glob(HOLDOUT)); files = [f for f in files if f not in hold]
else:
    k = max(1, int(round(0.08 * len(files)))); hold = list(rng.choice(files, k, replace=False)); files = [f for f in files if f not in hold]


def load_chains(fl, cap=None):
    S_l, C_l, st_l = [], [], []
    for f in fl:
        z = np.load(f); S_l.append(z['S']); C_l.append(z['C']); st_l.append(z['stratum'])
    S = np.concatenate(S_l); Cc = np.concatenate(C_l); st = np.concatenate(st_l)
    if cap and S.shape[0] > cap:
        i = rng.choice(S.shape[0], cap, replace=False); S, Cc, st = S[i], Cc[i], st[i]
    return torch.from_numpy(S), torch.from_numpy(Cc), torch.from_numpy(st.astype(np.int64))


S_a, C_a, st_a = load_chains(files)
S_h, C_h, st_h = load_chains(hold, cap=20000)
T_A = S_a.shape[1]; K_MAX = (T_A - 1) // 12
log(f"AIDE chains: train {S_a.shape[0]} ({len(files)} files), held-out {S_h.shape[0]} ({len(hold)} files); "
    f"T={T_A} -> K_max={K_MAX}; train strata {np.bincount(st_a.numpy(), minlength=4).tolist()}")


def load_orig(name, n, inj):
    S, Cc = load_v8_set(f"/data/tomas-jax-output/{name}", "train", n, verbose=False)
    S, Cc = torch.from_numpy(S[:, :T_A]).contiguous(), torch.from_numpy(Cc[:, :T_A]).contiguous()
    if inj:
        k = torch.exp(S[:, 0, :N_BINS]).sum(1) / BOX > 1.0; S, Cc = S[k], Cc[k]
    return S, Cc


no = max(ORIG_N, 4) if not SMOKE else 2000
parts = [load_orig("v9_sai", int(no * 0.45), True), load_orig("v8_carma_2wk", int(no * 0.15), False),
         load_orig("v9_sai_cont", int(no * 0.2), True), load_orig("v10_ideal", int(no * 0.2), True)]
S_o = torch.cat([p[0] for p in parts]); C_o = torch.cat([p[1] for p in parts]); del parts
if S_o.shape[1] < T_A:
    sys.exit(f"original chains shorter ({S_o.shape[1]}) than AIDE chains ({T_A})")
St = torch.cat([S_a, S_o]); Ct = torch.cat([C_a, C_o]); n = St.shape[0]
# STRATA_W = per-stratum multipliers (plume,active,quiet,empty) on the AIDE chain weight. 2026-10-17: the chain set is
# 40 % plume cells against ~1 % on the grid, and every fine-tune so far inflated the nucleation bins globally.
STRATA_W = torch.tensor([float(x) for x in os.environ.get('STRATA_W', '1,1,1,1').split(',')])
CHAIN_W = torch.cat([torch.full((S_a.shape[0],), AIDE_W) * STRATA_W[st_a], torch.ones(S_o.shape[0])])
log(f"STRATA_W={STRATA_W.tolist()} -> AIDE weight share by stratum {[(round(float(CHAIN_W[:S_a.shape[0]][st_a == k].sum() / CHAIN_W.sum()), 3)) for k in range(4)]}, original {round(float(CHAIN_W[S_a.shape[0]:].sum() / CHAIN_W.sum()), 3)}")
log(f"train chains {n}: AIDE {S_a.shape[0]} (weight {AIDE_W}) + original {S_o.shape[0]}")
TESTS = load_all_tests(2000)
SIX = load_multiset() if MIX > 0 else None
mix6 = MultiMixer(SIX, 64, seed=5) if MIX > 0 else None

# ---- model ------------------------------------------------------------------------
m = BinTF(D_, NL_).to(dev); m.load_state_dict(torch.load(WARM, map_location=dev)["model"]); m.train()
log(f"warm from {WARM}")


def chain(model, s0, cond, K, ck=False):
    cur, outs = s0, []
    for b in range(K):
        cb = cond[:, b * 12:b * 12 + 13]
        o = checkpoint(model, cur, cb, use_reentrant=False) if ck else model(cur, cb); outs.append(o); cur = o[:, -1]
    return torch.cat(outs, 1)


@torch.no_grad()
def eval_aide(tag):
    """Held-out AIDE chains: 6 h totals error per stratum, and K_MAX-block self-rollout."""
    m.eval(); P6, T6, S0 = [], [], []; PK, TK = [], []
    for a in range(0, S_h.shape[0], 512):
        s = S_h[a:a + 512].to(dev); c = C_h[a:a + 512].to(dev)
        sn = norm.norm_state(s); cn = norm.norm_cond(c)
        P6.append(norm.denorm_state(m(sn[:, 0], cn[:, :13])[:, -1])); T6.append(s[:, 12]); S0.append(s[:, 0])
        cur = sn[:, 0]
        for b in range(K_MAX):
            cur = m(cur, cn[:, b * 12:b * 12 + 13])[:, -1]
        PK.append(norm.denorm_state(cur)); TK.append(s[:, K_MAX * 12])
    P6, T6, S0, PK, TK = (torch.cat(x) for x in (P6, T6, S0, PK, TK))
    rel = lambda p, t, sl: (tot(p, sl) - tot(t, sl)).abs() / tot(t, sl).clamp(min=1e-25) * 100
    out = {}
    for s_id, name in ((0, 'plume'), (1, 'active'), (2, 'quiet'), (3, 'empty')):
        mk = (st_h == s_id).to(dev)
        if mk.sum() < 10:
            continue
        rN6, rM6 = rel(P6[mk], T6[mk], Nsl), rel(P6[mk], T6[mk], Msl)
        rNK, rMK = rel(PK[mk], TK[mk], Nsl), rel(PK[mk], TK[mk], Msl)
        out[name] = (rN6.median().item(), rN6.quantile(.9).item(), rM6.median().item(), rM6.quantile(.9).item(),
                     rNK.median().item(), rNK.quantile(.9).item(), rMK.median().item(), rMK.quantile(.9).item())
        log(f"{tag} AIDE {name:6s} ({int(mk.sum())}): 6h N {out[name][0]:.2f}/{out[name][1]:.1f} M {out[name][2]:.2f}/{out[name][3]:.1f} | "
            f"{K_MAX*6}h N {out[name][4]:.1f}/{out[name][5]:.1f} M {out[name][6]:.2f}/{out[name][7]:.1f}  (p50/p90 %)")
    m.train()
    # the score used to pick best_aide.pt: plume + active 6 h number and mass p90, mean
    keys = [k for k in ('plume', 'active') if k in out]
    return float(np.mean([out[k][1] + out[k][3] for k in keys])) if keys else float('nan')


f6 = lambda s, c: norm.denorm_state(m(s, c)[:, -1])
best = eval_aide("[start]")
m.eval(); eval_all_sets(f6, TESTS, "[start] 6h"); m.train()
log(f"[start] score {best:.2f}")

bi = list(range(16)) + list(range(N_BINS, N_BINS + 16))
ci = list(range(CB0, N_BINS)) + list(range(N_BINS + CB0, 2 * N_BINS))
STAGES = [(int(a), int(b), int(c)) for a, b, c in (x.split(":") for x in STAGES_ENV.split(","))]
if SMOKE:
    STAGES = [(2, 1, 16)]
for si, (K, eps, bs) in enumerate(STAGES):
    if K > K_MAX:
        log(f"stage K={K} exceeds chain length (K_max={K_MAX}); clipped"); K = K_MAX
    STEPS = K * 12; ck = K >= 8
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, eps)
    log(f"\n=== STAGE {si}: K={K} ({K*6}h chains), {eps}ep, bs={bs}, checkpointing={ck} ===")
    for ep in range(eps):
        n_ep = 300 if SMOKE else min(CAP, n)
        g_ = torch.Generator().manual_seed(1000 * si + 17 * ep + SEED)
        perm = torch.multinomial(CHAIN_W, n_ep, replacement=False, generator=g_)
        t0 = time.time(); nb = 0; lsum = 0.0
        for b in range(0, len(perm) - bs, bs):
            idx = perm[b:b + bs]; s = St[idx].to(dev); c = Ct[idx].to(dev)
            sn = norm.norm_state(s[:, :STEPS + 1]); cn = norm.norm_cond(c[:, :STEPS + 1])
            pred = chain(m, sn[:, 0], cn, K, ck); tgt = sn[:, 1:STEPS + 1]
            se = (pred - tgt) ** 2
            loss = se.mean() + totals_loss(pred, tgt, LAMM, LAMN) + BW * se[..., bi].mean() + CW * se[..., ci].mean()
            if TF1 > 0:
                Bn = sn.shape[0]; s_tf = sn[:, 0:STEPS:12].reshape(Bn * K, -1)
                widx = (torch.arange(K, device=dev) * 12).view(K, 1) + torch.arange(13, device=dev).view(1, 13)
                c_tf = cn[:, widx].reshape(Bn * K, 13, -1); t_tf = sn[:, 1:STEPS + 1].reshape(Bn, K, 12, -1).reshape(Bn * K, 12, -1)
                p_tf = m(s_tf, c_tf)
                loss = loss + TF1 * (((p_tf - t_tf) ** 2).mean() + totals_loss(p_tf, t_tf, LAMM, LAMN) + BW * ((p_tf[..., bi] - t_tf[..., bi]) ** 2).mean() + CW * ((p_tf[..., ci] - t_tf[..., ci]) ** 2).mean())
            if MIX > 0:
                sn6, cn6 = mix6.next(); p6 = m(sn6[:, 0], cn6); t6 = sn6[:, 1:]
                loss = loss + MIX * (((p6 - t6) ** 2).mean() + totals_loss(p6, t6, LAMM, LAMN) + BW * ((p6[..., bi] - t6[..., bi]) ** 2).mean() + CW * ((p6[..., ci] - t6[..., ci]) ** 2).mean())
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            nb += 1; lsum += loss.item()
        sched.step()
        log(f"  K={K} ep{ep+1}: {nb} batches, mean loss {lsum / max(nb, 1):.4f}, {time.time() - t0:.0f}s")
        score = eval_aide(f"  K={K} ep{ep+1}:")
        m.eval(); eval_all_sets(f6, {"v9_sai": TESTS["v9_sai"], "v8_carma": TESTS["v8_carma"]}, f"  K={K} ep{ep+1}: 6h"); m.train()
        torch.save({"model": m.state_dict(), "K": K, "ep": ep + 1}, f"{OUT}/stage_K{K}_ep{ep+1}.pt")
        torch.save({"model": m.state_dict(), "K": K, "ep": ep + 1}, f"{OUT}/latest.pt")
        if score < best:
            best = score; torch.save({"model": m.state_dict(), "K": K, "ep": ep + 1, "score": score}, f"{OUT}/best_aide.pt")
            log(f"  -> best_aide.pt (score {score:.2f})")
    torch.save({"model": m.state_dict(), "K": K}, f"{OUT}/stage_K{K}.pt")
m.eval(); eval_all_sets(f6, TESTS, "[final] 6h"); eval_aide("[final]")
log("DONE")
