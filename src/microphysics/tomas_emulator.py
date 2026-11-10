"""The TOMAS-emulator adapter: coupling.py's per-cell state <-> the bin-token
transformer's (B, 82) log-state, one 6-hour step per call.

`run_microphysics_full_emulator` is a drop-in for `coupling.run_microphysics_full`
with the SAME signature and return contract as `tomas_fast.run_microphysics_full_fast`
(mixing-ratio <-> per-box conversion, two-moment consistency clip + clip
diagnostic). The difference is the engine: instead of 60 x 360 s of
`tomas_jax.fast`, ONE forward pass of the learned 6-hour operator from
/data/tomas-emulator (EMULATOR_HANDOFF.md, 2026-10-09) per cell.

Where the emulator came from, in one paragraph: it was trained on tomas-jax FAST
trajectories (the v8_carma global sampling + v9_sai injection scenarios) at a
30-minute grid, so the "truth" it approximates is EXACTLY the engine
driver_fast.py runs by default -- which is what makes an A/B against
`MICRO_ENGINE=fast` a clean benchmark (micro_bench.py) rather than a comparison
across two physics.

INTERFACE CONTRACT (handoff section 3), restated in this tree's units:
  state (82)  = log of per-BOX quantities, box = BOXVOL = 1e6 cm3 = 1 m3 -- the
                same box coupling.py already uses, so Nk [#/m3] and Mk [kg/m3]
                go in with no rescaling. [0:40] log Nk, [40:80] log Mk (dry SO4),
                [80] log H2SO4 [kg/box], [81] log SO2 [kg/box]. Floor 1e-20 before
                the log (LOG_EPS of dataset_v8 is 1e-30; the model saw >= 1e-5 as
                "occupied", so anything below that is "empty" either way).
  forcing (13, 6) at t0, t0+30min, ..., t0+6h: [log P(Pa), T(K), RH(0-1),
                log OH(molec/cm3), log SO2 source(kg/box/30min), log kdil(1/s)].
                The model reads points 1..12 (the END of each 30-min interval).
  T, P, RH are held constant over the coupling step by coupling.py (T is the
  it0 snapshot + accumulated radiative dT), so they are repeated 13x. OH is the
  diurnal SZA parabola (OH_SZA=1) sampled by coupling.oh_sza at substep CENTERS;
  _oh_to_endpoints below interpolates those onto the 13 endpoints. coupling.py
  applies the SAI SO2 release to the gas field at the START of the step, before
  advection, so from the microphysics' point of view there is no in-step source:
  channel 4 is log(1e-30) everywhere, as is kdil (no dilution in a grid cell).

PRECISION: fp16 autocast + torch.compile(mode='reduce-overhead') (CUDA graphs),
the handoff's deployment path, measured lossless vs fp32. NOT bf16 (widens the
14-day tails). The compiled graph has a FIXED batch size, so the last batch is
padded with copies of cell 0 and trimmed.

TWO FRAMEWORKS ON ONE GPU. The run process now holds both JAX (advection,
radiation, settling, and the fast engine when it is the benchmark reference)
and torch. Each preallocates nothing by default here (run_prod.py exports
XLA_PYTHON_CLIENT_PREALLOCATE=false; torch's caching allocator grows on
demand), so they coexist; they do NOT share a stream, so every hand-off goes
through host numpy -- ~430 MB each way per step at 1.3 M cells, milliseconds
against the forward pass.

IMPORT ORDER MATTERS, as for tomas_fast.py: this module imports coupling, which
reads the environment at import. driver_fast.py imports it below its
os.environ.setdefault block.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _paths                      # noqa: E402,F401 -- puts src/ subdirs on sys.path

import coupling as C               # noqa: E402 -- see IMPORT ORDER above

# ---- emulator knobs -----------------------------------------------------------
# Where the emulator code and weights live. EMU_CODE is the tomas-emulator tree
# (frontier_bintf_infer.py, frontier_common.py, dataset_v8.py); it is NOT a
# submodule of this repo (yet), so it is located by env, like TOMAS_JAX_PATH.
EMU_CODE  = os.environ.get('EMU_CODE', '/data/tomas-emulator')
EMU_ROOT  = os.environ.get('EMU_ROOT', '/data/tomas-jax-output/emulator_v1')
# The handoff's recommended checkpoint: BinTF d=384, 6 layers, 7.2 M params,
# 6 h totals 3.1 % / 2.6 % (N/M, injection cells), 2.9 us/cell fp16 compiled.
# The lighter d=256 x 4 (1.1 us/cell, 4.7 % / 4.9 %) is EMU_CKPT=.../frontier_bintf_kchain_rebal/avg_top3.pt EMU_D=256 EMU_NL=4.
EMU_CKPT  = os.environ.get('EMU_CKPT', f'{EMU_ROOT}/frontier_bintf_kchain_big_rebal/avg_top3.pt')
EMU_D     = int(os.environ.get('EMU_D', '384'))
EMU_NL    = int(os.environ.get('EMU_NL', '6'))
EMU_BS    = int(os.environ.get('EMU_BS', '16384'))      # fixed batch per compiled graph
EMU_DTYPE = os.environ.get('EMU_DTYPE', 'fp16')         # fp16 | fp32  (bf16 deliberately refused)
EMU_COMPILE = os.environ.get('EMU_COMPILE', '1') != '0' # torch.compile reduce-overhead
# The OH diurnal field is built by coupling.oh_sza at C.OH_SUBSTEPS substep
# CENTERS. 24 centers (every 15 min at 7.5, 22.5, ...) put every 30-min endpoint
# exactly midway between two samples, so the interpolation below is a plain
# average there. An explicit OH_SUBSTEPS in the environment is respected and
# resampled onto the endpoints the same way.
if 'OH_SUBSTEPS' not in os.environ:
    C.OH_SUBSTEPS = 24

N_END = 13                 # forcing points per 6 h window (30-min grid incl. both ends)
# Floors before the log, read off the training cache (v9_sai_test_S.npy): an
# empty bin is Nk = 1e-20 (TOMAS's own floor, log = -46.05) with Mk = 1e-30
# (dataset_v8 LOG_EPS, log = -69.08); the gases floor at 1e-30 too. Getting
# these wrong moves the "empty" token by 23 log units.
N_FLOOR = 1e-20
M_FLOOR = 1e-30
# SO2 is the one state channel whose training floor is NOT the generator's
# LOG_EPS: the prognostic SO2 of v8/v9 never fell below ~1e-17 kg/box (log -39;
# 0.1 molecule cm-3), so a cell whose SO2 was advected to exactly zero and then
# logged at 1e-30 (log -69) is 7 sigma outside the training set. Found on the
# first emulator-driven run (runs/emu_dumps_emu7/ab_step7_noproj.txt): the worst
# cell had SO2 = 0.00 cm-3 and was answered with 1660x the physics' mass. The
# emulator-driven trajectory makes such cells itself (its SO2 output in a
# depleted plume is tiny; the next advection floors it to 0), which is why the
# physics-driven A/B did not show it. EMU_SO2_FLOOR overrides.
S_FLOOR = float(os.environ.get('EMU_SO2_FLOOR', '1e-17'))     # kg SO2 per box
NEPS_EMU = 1e-5            # the model saw >= 1e-5 per box as occupied; below -> empty
# FORCING FLOORS -- keep the inputs inside the training support. Found on the
# first coupled A/B (2026-10-09, runs/emu_dumps/ab_step0.txt): coupling.oh_sza's
# parabola in cos(SZA) is EXACTLY zero at night, and log(max(0, 1e-30)) = -69 is
# 17 standard deviations below the training mean of log OH (11.2 +- 4.6; the
# generator's diurnal OH never reached zero, its minimum is 0.006 cm-3). Every
# cell that saw a zero OH sample -- the band around the terminator, |lat| > 60
# in January -- was answered with a spurious 1e8 cm-3 nucleation burst from a
# 13 cm-3 background, and 1000 such cells carried 99.8 % of a +5000 % global
# number error while the per-cell median was 0.3 %. The floor value was swept on
# that state (runs/emu_dumps/ab_step0_ohfloor*.txt): active-cell totN p90 is
# 38.5 % at 0.01 cm-3 (the training p0.1), 20.5 % at 1 (p1), 16.3 % at 100 (p5),
# with the global error unchanged -- the thinner the training coverage of the
# night value, the noisier the answer. 100 cm-3 is a value the generator's own
# nights reached routinely and is chemically nothing (k_OH*OH*6h ~ 2e-6 of the
# SO2). RH is clamped to the generator's [0.01, 1] for the same reason (minor:
# the normalized RH moves by 0.04 sigma).
OH_FLOOR = float(os.environ.get('EMU_OH_FLOOR', '100'))       # molec/cm3
RH_MIN = 0.01
# OUTPUT PROJECTION. Two constraints every box-model microphysics step
# satisfies exactly and the network only approximately:
#   (a) two-moment consistency: Mk in [Nk*xk_lo, Nk*xk_hi] (TOMAS's mnfix);
#   (b) total sulfur: SO2 + H2SO4 + SO4 sulfur out == in (no source in the box).
# EMU_PROJECT = clip applies (a) as mnfix would (mass clipped to the
# number); 'sulfur' applies (a) and (b); '0' neither; 'chem' (the default since
# 2026-10-13, see below) applies (a) and then the exact gas chemistry.
# (b) is OFF by default, and the record of why is runs/emu_dumps/ab_step0_final.txt
# vs ab_step0_gpuprep.txt: enforcing it raised the active-cell one-step error
# from 16 % to 72 % (p90, both N and M). The emulator's SO2 answer is good to
# ~1 % per step, but the physics only converts ~1-2 % of the SO2 per step, so
# the gas error is the SIZE of the signal, and crediting it to the aerosol --
# which in a clean cell holds far less sulfur than the gas -- turns a 1 % gas
# error into a 10-1e7x aerosol error. (b) is implemented with the two
# monotonicities of a stratospheric step (gas sulfur only falls, aerosol
# sulfur only rises) and kept for an emulator whose gas channel is better.
# The mass fabrication that motivated it (runs/bench_emu30d, 137x the physics'
# mass at step 7, burden 50x by day 4) turned out to be the SO2 input floor
# above, not a conservation problem.
# 'chem' (2026-10-13) is (b) done the other way round. The SO2 chemistry in the
# physics is pseudo-first-order in OH with a known rate (Troe k1(T, P, RH),
# tomas_jax.physics.so2_chemistry) and no other sink, so the SO2 the physics
# consumes in a step is computable EXACTLY per cell from the inputs -- which
# is what the 'sulfur' mode lacked: it took the network's SO2 answer as the
# truth. 'chem' overwrites the network's SO2 with the exact decay, caps its
# H2SO4 at what is available, and moves the sulfur the gas lost into the
# aerosol: the network's per-bin mass change is kept (coagulation shifts,
# which it gets right) and the shortfall or excess against the sulfur budget
# is deposited where condensation would put it, by surface area, then
# re-clipped. Why it matters: every 30-day rollout, hybrid or not, drifted
# through this seam -- on the step-7 state the handoff network oxidises
# 0.38 % of the SO2 per step where the physics oxidises 0.76 %, while adding
# 0.10 % of aerosol mass against the physics' 0.057 % (SO2 burden +29-40 %
# at day 30, mass and number bent the other way).
# EMU_CHEM_NCLIP (2026-10-17): how the chem projection restores the two-moment
# bounds AFTER giving the aerosol exactly the sulfur the gas lost.
#   0 (legacy, the default until 2026-10-18): re-clip MASS -- creates sulfur where
#     the network's number implies a higher minimum mass (M < N xk_lo) and
#     destroys it where M > N xk_hi; net +0.018 % of the network cells' sulfur
#     per step on the day-30 state, which compounds to the +22 % mass / +25 %
#     AOD / -24 % forcing drift of the one-year run (runs/micro1yr_c5 vs
#     ffsl1yr_K24) while the 30-day state looks fine.
#   1: conserving -- where M < N xk_lo reduce NUMBER to M/xk_lo (that spurious
#     number is also the +1.4 %/step global N bias); where M > N xk_hi clip
#     the mass and redistribute the removed mass to the cell's bins with room
#     below their own maximum (two passes; the remainder is logged as
#     chem_lost). A first attempt that RAISED number to M/xk_hi instead blew
#     up a 30-day run (NaN by day 10): it injected huge counts into nearly
#     empty bins, which the network then amplified; a second attempt turned the
#     bins the additive deposit had driven negative into negative number ->
#     NaN -- hence the multiplicative removal of negative residuals below.
#     VALIDATED 2026-10-18 over ONE YEAR (runs/micro1yr_nclip1b vs the legacy
#     runs/micro1yr_c5, both vs ffsl1yr_K24, day 365): mass -0.1 % (legacy
#     +22.1 %), SO2 +0.3 % (same), number -16.3 % (-22.6), AOD +19.3 % (+25.1),
#     forcing -19.1 % (-24.1), r_eff +6.9 % (+5.9); at day 30 number is -2.2 %
#     against -0.7 % (the spurious number removed), everything else equal or
#     better. What remains at a year is the size distribution (number deficit
#     above the plume, coarse-bin sawtooth), i.e. the network, not bookkeeping.
EMU_CHEM_NCLIP = int(os.environ.get('EMU_CHEM_NCLIP', '1'))     # default 1 since 2026-10-18 (one-year validation below)
EMU_PROJECT = os.environ.get('EMU_PROJECT', 'chem').lower()        # production default since 2026-10-13 (was clip)
if EMU_PROJECT not in ('clip', 'sulfur', 'chem', '0'):
    sys.exit(f"tomas_emulator: EMU_PROJECT must be clip|sulfur|chem|0, got {EMU_PROJECT!r}")
_S_SO4, _S_SO2, _S_H2SO4 = 32.0 / 96.0, 32.0 / 64.0, 32.0 / 98.0

if EMU_DTYPE not in ('fp16', 'fp32'):
    sys.exit(f"tomas_emulator: EMU_DTYPE must be fp16 or fp32 (bf16 widens the "
             f"14-day tails -- handoff section 2); got {EMU_DTYPE!r}")
if C.NBINS != 40:
    sys.exit(f"tomas_emulator: emulator is 40-bin; coupling NBINS={C.NBINS}")
if abs(C.DT_MICRO - 6 * 3600.0) > 1e-6:
    sys.exit("tomas_emulator: the emulator is a 6-hour operator; STEP_HOURS must be 6")
if abs(C.BOXVOL - 1.0e6) > 1e-6:
    sys.exit(f"tomas_emulator: trained at BOX=1e6 cm3, coupling BOXVOL={C.BOXVOL}")

# ---- load the model (torch) --------------------------------------------------
try:
    import torch
except ImportError as _e:
    raise SystemExit(f"tomas_emulator: torch is required ({_e}); the run environment "
                     "must carry BOTH jax[cuda] and torch -- see docs/EMULATOR.md")
if EMU_CODE not in sys.path:
    sys.path.insert(0, EMU_CODE)
try:
    from frontier_bintf_infer import load_bintf_fast     # noqa: E402
    from dataset_v8 import NormStatsV8, N_STATE, N_COND  # noqa: E402
except ImportError as _e:
    raise SystemExit(f"tomas_emulator: cannot import the emulator code from "
                     f"EMU_CODE={EMU_CODE} ({_e})")
assert N_STATE == 2 * C.NBINS + 2 and N_COND == 6

_DEV = 'cuda'
_norm = NormStatsV8.load(f'{EMU_ROOT}/foundation_mlp180/norm_stats.npz')
_t0 = time.time()
_step6h = load_bintf_fast(EMU_CKPT, EMU_D, EMU_NL,
                          dtype=torch.float16 if EMU_DTYPE == 'fp16' else torch.float32,
                          compile_mode='reduce-overhead' if EMU_COMPILE else None)
T_LOAD = time.time() - _t0
# a constant forcing row for the two channels this model never sees vary here
_LOG_ZERO = float(np.log(1e-30))



# ---- triage: which cells need the network at all ------------------------------
# Measured on the dumped states (runs/emu_dumps*/ab_*_final.npz, 2026-10-13):
# the physics changes number or mass by >1 % in only 45-49 % of cells per 6 h
# step, and those cells hold 4-5 % of the mass. A cell is QUIET when
#   (a) the sulfur that could reach the aerosol in 6 h -- all its H2SO4 plus
#       the SO2 that k1*OH can oxidise -- is < EMU_TRIAGE_POT of its aerosol
#       mass (mass cannot move by more than that),
#   (b) its number is < EMU_TRIAGE_N cm-3 (coagulation timescale 1/(K N) ~ days),
#   (c) the steady-state H2SO4 the chemistry can sustain against the
#       condensation sink is < EMU_TRIAGE_HSS cm-3 (no nucleation), and
#   (d) less than EMU_TRIAGE_FINE of its number is below ~10 nm (the one
#       population that still moves at these N: bin-0 scavenging).
# On the three test states the physics moved such cells by 0.13 % (median) and
# 0.6-0.9 % (p99) in number and 0.02-0.03 % (p99) in mass -- the same size as
# the emulator's own error on them (0.10-0.15 % median). They get the exact
# gas chemistry (pseudo-first-order SO2 + OH) with the acid condensed onto the
# existing bins in proportion to their surface area, and no change in number.
# 31-37 % of the grid, 80 % of the mass, and the network runs on the rest.
# EMU_BIN_CORR = path of an .npz with cN, cM (40,) per-bin multiplicative
# factors applied to the network's raw per-bin output before the projection
# (default off). Why (2026-10-17): on the day-30 state the network's one-step
# per-bin bias against tomas_jax.fast is only 0.1-0.3 % in bins 17-28 and
# -0.8/-1.2 % in bins 31/32, but sign-stable, and its 120-step integral IS the
# +-15 % even/odd sawtooth of the 30-day state in bins 22-32 (the forcing
# lever: equal AOD, coarser tail, 7.5 % weaker forcing). The factors are the
# burden-weighted fast/emulator ratios of the one-step answer, calibrated by
# scripts/validation/calibrate_bin_corr.py on dumped pre-micro states; the
# chem projection afterwards still sets the total aerosol sulfur, so this
# only reshapes the distribution.
EMU_BIN_CORR = os.environ.get('EMU_BIN_CORR', '')
_BIN_CORR = None
if EMU_BIN_CORR:
    _bc = np.load(EMU_BIN_CORR)
    _BIN_CORR = (torch.as_tensor(np.asarray(_bc['cN'], np.float64), device=_DEV),
                 torch.as_tensor(np.asarray(_bc['cM'], np.float64), device=_DEV))
# EMU_NYQ_DAMP -- INERT on real states (2026-10-17, keep 0): the day-30
# pattern phase-slips every few bins, so a single windowed (-1)^b mode captures
# almost none of it (global per-bin totals unchanged to 3 decimals at 0.5).
# EMU_NYQ_DAMP = lambda in [0, 1]: each step, remove lambda x the 2-bin
# alternating (Nyquist) mode of the network's per-bin N and M over bins
# EMU_NYQ_B0..39 (default off). The mode is a Hann-windowed (-1)^b pattern
# with zero mean, so the per-cell totals of N and M are conserved and a smooth
# distribution is left untouched (no diffusion: only that one mode is
# projected out). Why (2026-10-17): the 30-day emulator state carries a
# +-15 % even/odd sawtooth in bins 22-32 that the physics never produces; the
# network's one-step bias on a sawtoothed input is 3-10x its bias on a smooth
# one in bins 29-32, i.e. it feeds the mode back on itself. Bins that cannot
# go negative scale the correction down per cell.
EMU_NYQ_DAMP = float(os.environ.get('EMU_NYQ_DAMP', '0'))
EMU_NYQ_B0 = int(os.environ.get('EMU_NYQ_B0', '21'))
_NYQ_P = None
if EMU_NYQ_DAMP > 0:
    _nb = 40 - EMU_NYQ_B0
    _w = np.sin(np.pi * (np.arange(_nb) + 0.5) / _nb) ** 2
    _p = ((-1.0) ** np.arange(EMU_NYQ_B0, 40)) * _w
    _p -= _p.mean()
    _NYQ_P = torch.as_tensor(_p / np.sqrt((_p ** 2).sum()), dtype=torch.float64, device=_DEV)
def _nyq_damp(X):
    seg = X[:, EMU_NYQ_B0:]
    corr = EMU_NYQ_DAMP * (seg * _NYQ_P).sum(1, keepdim=True) * _NYQ_P
    ratio = torch.where(corr > seg, seg / corr.clamp(min=1e-300), torch.full_like(seg, 2.0)).amin(1, keepdim=True)
    X[:, EMU_NYQ_B0:] = seg - torch.clamp(ratio, max=1.0) * corr
    return X
# EMU_SAW_DAMP -- REFUTED 2026-10-17, keep 0. On the real day-30 cells the
# per-cell bin distributions are spiky (isolated occupied bins between empty
# ones), so the alternation test fires on genuine peaks: the one-step
# network-class bias went from +-0.5 % to -9..-15 % with +92 % dumped into
# bin 30. Per-cell shape filters on bin distributions are not viable; the
# sawtooth is a training-coverage problem of the network on mature coarse
# tails. Kept only as a record of the experiment.
# EMU_SAW_DAMP = lambda in [0, 1]: local sawtooth damper on the network's
# per-bin N and M over bins EMU_NYQ_B0..39. r_b = x_b - ([1,2,1]/4 * x)_b is the
# residual from a 3-point smooth; where r_b alternates in sign with BOTH
# neighbours (a 2-bin oscillation; a smooth curved tail has same-sign
# residuals and is untouched) lambda x r_b is removed, the per-cell total of
# the removed amount is redistributed over the touched bins (N and M
# conserved), negative bins are prevented by scaling the cell's correction.
# Why (2026-10-17): the global Nyquist projector (EMU_NYQ_DAMP) misses the real
# day-30 pattern, which phase-slips every few bins; the bias sits entirely in
# the network-routed cells (physics/quiet classes: 0.00 %), bin 22 -0.45,
# 23 +0.49, 24 +0.35, 29 -0.32, 31 -1.1, 32 -1.6 % per step.
EMU_SAW_DAMP = float(os.environ.get('EMU_SAW_DAMP', '0'))
def _saw_damp(X):
    seg = X[:, EMU_NYQ_B0:]
    pad = torch.cat([seg[:, :1], seg, seg[:, -1:]], 1)
    r = seg - 0.25 * (pad[:, :-2] + 2.0 * pad[:, 1:-1] + pad[:, 2:])
    sg = torch.sign(r)
    left = torch.cat([sg[:, 1:2], sg[:, :-1]], 1); right = torch.cat([sg[:, 1:], sg[:, -2:-1]], 1)
    alt = (sg * left < 0) & (sg * right < 0) & (seg > 0)
    corr = torch.where(alt, EMU_SAW_DAMP * r, torch.zeros_like(r))
    n_alt = alt.sum(1, keepdim=True).clamp(min=1)
    corr = corr - torch.where(alt, corr.sum(1, keepdim=True) / n_alt, torch.zeros_like(corr))   # conserve the total
    ratio = torch.where(corr > seg, seg / corr.clamp(min=1e-300), torch.full_like(seg, 2.0)).amin(1, keepdim=True)
    X[:, EMU_NYQ_B0:] = seg - torch.clamp(ratio, max=1.0) * corr
    return X
EMU_TRIAGE = os.environ.get('EMU_TRIAGE', '1') != '0'
TRIAGE_POT = float(os.environ.get('EMU_TRIAGE_POT', '1e-3'))
TRIAGE_N = float(os.environ.get('EMU_TRIAGE_N', '300'))        # cm-3
TRIAGE_HSS = float(os.environ.get('EMU_TRIAGE_HSS', '1e5'))    # molec cm-3
TRIAGE_FINE = float(os.environ.get('EMU_TRIAGE_FINE', '0.1'))  # number fraction in bins 0..9

# ---- hybrid: the physics on the plume ------------------------------------------
# PRODUCTION DEFAULTS for MICRO_ENGINE=emulator (2026-10-13, docs/EMULATOR.md
# "Second pass"): EMU_PHYS_TOP=0.01, EMU_PHYS_NUM=0.04, EMU_PROJECT=chem,
# EMU_TRIAGE=1. 30 days of the production scenario against the physics run:
# number burden -1.2 %, mass +0.1 %, SO2 0.0 %, AOD 0.0 %, r_eff -1.1 %;
# microphysics stage 4.9 s against 14.5 s. The network alone (all three off)
# was -35 % in number and +10 % in AOD at 3.4 s. Set EMU_PHYS_TOP=0
# EMU_PHYS_NUM=0 EMU_PROJECT=clip to reproduce the first-pass benchmark.
# Measured on the emulator-driven step-7 state (scripts/validation/micro_ab_offline.py,
# 2026-10-13): the top 1 % of cells by sulfur (SO4 + SO2 + H2SO4 per box) hold
# 80 % of all particles and carry 88 % of the emulator's signed global number
# error (-9.4 of -10.7 %); the next 9 % of cells add 0.6 %. Those are the
# fresh-injection cells where the network's burst error lives (plume totN p50
# 18 %, p90 51 %). With EMU_PHYS_TOP > 0 that fraction of cells is integrated
# by tomas_jax.fast exactly as MICRO_ENGINE=fast would (same knobs, through
# tomas_fast), and the network / quiet step take the rest. The subset is
# padded with empty cells to a multiple of EMU_PHYS_PAD so run_fast's jit sees
# a stable shape; EMU_PHYS_CHUNKS > 1 stiffness-sorts it into that many
# sequential chunks (tomas_fast's FAST_SORT) so one violent cell does not set
# the substep count for all.
# EMU_PHYS_NUM > 0 routes a second set to the physics: that fraction of the
# remaining cells ranked by total particle number per box. Found on the
# 30-day closure runs (2026-10-13): once EMU_PROJECT=chem hands the network
# the physics' H2SO4 instead of its own 50 %-low estimate, the per-step global
# number error along its trajectory grows +0.6 -> +18 % over the month, all
# positive, in bins 0-6 at 36-143 hPa, 30 S-60 N. On the day-30 state the
# cells carrying it are dense fine populations (~1e4 cm-3 against a grid
# median of 70) in which the physics halves the number by coagulation per
# step and the network removes a quarter: under-coagulation, not
# over-nucleation. Ranked by number the top 4 % of cells carry 89 % of that
# error and the top 8 % 98 % (by steady-state H2SO4, the first ranking
# tried, 73 % and 82 %); the sulfur ranking does not see them because they
# hold little mass. Cost is EMU_PHYS_TOP's per cell.
EMU_PHYS_TOP = float(os.environ.get('EMU_PHYS_TOP', '0.01'))    # fraction of cells; 0 = off (default 1 % since 2026-10-13)
EMU_PHYS_NUM = float(os.environ.get('EMU_PHYS_NUM', '0.04'))    # fraction of cells; 0 = off (default 4 % since 2026-10-13)
# EMU_PHYS_COARSE = fraction of the remaining cells, ranked by their mass in
# bins >= EMU_COARSE_B0 (Dp > 240 nm), that also go to the physics. Why
# (2026-10-17): the day-30 forcing of the emulator run is 7.4 % weaker than the
# physics' at equal AOD, and swapping ONLY bins >= 21 to the physics' values
# closes that to +1.8 % (RRTMGP on the day-30 states); the per-step bias in
# those bins (an even/odd sawtooth, 0.3-1.6 %/step) sits entirely in the
# network-routed cells, which hold 66 % of the coarse mass. That mass is not
# concentrated: beyond the sulfur/number classes the top 4/10/15 % of cells
# hold 23/44/57 % more of it, at ~0.22 s per 1 % of cells. Default 0.
EMU_PHYS_COARSE = float(os.environ.get('EMU_PHYS_COARSE', '0'))
EMU_COARSE_B0 = int(os.environ.get('EMU_COARSE_B0', '21'))
EMU_PHYS_PAD = int(os.environ.get('EMU_PHYS_PAD', '4096'))
EMU_PHYS_CHUNKS = int(os.environ.get('EMU_PHYS_CHUNKS', '1'))
# EMU_PHYS_CHUNK_MIN = n > 0 (2026-11-06, plume agent; default 0 = off): a physics_cells call uses at most
# padded_cells // n of the EMU_PHYS_CHUNKS chunks (at least 1), so a small call (the ~6.6k-cell EMU_PHYS_BURST
# re-run) stays one chunk: per-chunk launch overhead there costs 1-2 s/step against 0.1 s unchunked.
EMU_PHYS_CHUNK_MIN = int(os.environ.get('EMU_PHYS_CHUNK_MIN', '0'))
# ---- drift control (2026-10-21, micro-spec campaign agent B; all default OFF) ----
# The one-year error of the hybrid is a slow size-distribution DRIFT (number
# -17 %, AOD +19 % at day 365 against <1 % at day 30) built from small
# sign-stable per-step biases of the network in the cells it handles.
# EMU_PHYS_EVERY = k > 0: every k-th call is the plain physics step
#   (tomas_fast.run_microphysics_full_fast on all cells), the others are the hybrid.
# EMU_PHYS_ROT = k > 0: each call also sends a rotating 1/k strided slice of
#   the grid (flat index % k == call % k) to the physics, so every cell gets a
#   physics step every k calls.
# EMU_ONLINE_CORR = f > 0: ONLINE per-(region, bin) bias correction of the
#   network. Each call a random fraction f of the network-routed cells is ALSO
#   run through the physics (their physics answer is kept); the ratio of the
#   physics' to the network's raw per-bin number and mass, summed over those
#   cells per region (level x EMU_CORR_NLAT latitude bands), is tracked as an
#   exponential moving average of numerator and denominator (weight
#   EMU_CORR_EMA per call) and multiplies the network's raw output in every
#   network cell of that region before the chem projection (which keeps
#   sulfur exact). Factors are clamped to [1/EMU_CORR_CLAMP, EMU_CORR_CLAMP]
#   and left at 1 where the denominator holds < EMU_CORR_MINFRAC of the
#   bin's grid-wide sample. Unlike EMU_BIN_CORR (static, refuted 2026-10-17:
#   the bias is state-dependent) the factors follow the state as it evolves.
EMU_PHYS_EVERY = int(os.environ.get('EMU_PHYS_EVERY', '0'))
EMU_PHYS_ROT = int(os.environ.get('EMU_PHYS_ROT', '0'))
EMU_ONLINE_CORR = float(os.environ.get('EMU_ONLINE_CORR', '0'))
EMU_CORR_NLAT = int(os.environ.get('EMU_CORR_NLAT', '12'))
EMU_CORR_EMA = float(os.environ.get('EMU_CORR_EMA', '0.2'))
EMU_CORR_CLAMP = float(os.environ.get('EMU_CORR_CLAMP', '4'))
EMU_CORR_MINFRAC = float(os.environ.get('EMU_CORR_MINFRAC', '1e-4'))
EMU_CORR_SEED = int(os.environ.get('EMU_CORR_SEED', '0'))
EMU_CORR_B0 = int(os.environ.get('EMU_CORR_B0', '0'))          # correct only bins >= B0 (21 = coarse tail; 0 = all)
# Burst routing (agent D finding 2026-10-21: on the PHYSICS-trajectory day-365
# state 80 % of the network's -1.6 %/step global number bias sits in ~1000
# nucleation-burst cells (Asian-monsoon UTLS, 30-37 N, levels 17-18) with low
# pre-step number and sulfur, which the sulfur/number routing never sees; the
# network grows bin 0 there 1e6 -> 1e10 where the physics reaches 8e10).
# EMU_PHYS_HSS = f: also send the top f of the remaining (non-quiet) cells by
#   steady-state H2SO4 (production / free-molecular sink) to the physics.
# EMU_PHYS_BURST = f: AFTER the network, the top f*ncell network cells by
#   their predicted number increase in bins 0..EMU_BURST_B1-1 are recomputed
#   by the physics (the network flags its own bursts; costs one small extra
#   physics call).
EMU_PHYS_HSS = float(os.environ.get('EMU_PHYS_HSS', '0'))
EMU_PHYS_BURST = float(os.environ.get('EMU_PHYS_BURST', '0'))
EMU_BURST_B1 = int(os.environ.get('EMU_BURST_B1', '3'))
_ANY_PHYS = (EMU_PHYS_TOP > 0 or EMU_PHYS_NUM > 0 or EMU_PHYS_COARSE > 0 or EMU_PHYS_ROT > 0 or EMU_ONLINE_CORR > 0
             or EMU_PHYS_HSS > 0 or EMU_PHYS_BURST > 0)
if _ANY_PHYS or EMU_PHYS_EVERY > 0:
    import tomas_fast as TF                                       # noqa: E402  the fast engine + its knobs

from tomas_jax.physics.so2_chemistry import calc_k1_so2_oh   # noqa: E402  (jax; coupling already imports tomas_jax)
import jax                                                     # noqa: E402
import jax.numpy as jnp                                        # noqa: E402

_NA = 6.02214076e23; _KB = 1.380649e-23
_MW_SO2, _MW_H2SO4, _MW_SO4 = 64.066, 98.0, 96.0
_XK = torch.tensor(np.asarray(C.XK_NP), dtype=torch.float64, device=_DEV)         # (41,)
_MMID = torch.sqrt(_XK[:-1] * _XK[1:])
_DP = (6.0 * _MMID / (np.pi * 1770.0)) ** (1.0 / 3.0)                               # (40,) m
_DP2 = _DP ** 2


def _to_torch(a, dtype=torch.float64):
    """jax (zero-copy via DLPack) or numpy -> torch tensor on the GPU."""
    if torch.is_tensor(a):
        return a.to(_DEV, dtype)
    if isinstance(a, np.ndarray) or not hasattr(a, '__dlpack__'):
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float64)).to(_DEV, dtype)
    return torch.from_dlpack(a).to(dtype)


def _to_jax(t):
    """torch (GPU) -> jax, zero-copy; the tensor must not be modified afterwards."""
    return jax.dlpack.from_dlpack(t.contiguous())


def _oh_to_endpoints(oh_arr, ncell, n_end=N_END):
    """OH field -> (ncell, n_end) [molec/cm3] at the 30-min ENDPOINTS of the step.

    coupling.oh_sza samples substep CENTERS at (m+0.5)/nsub of the step; the
    emulator wants the values at k/12, k=0..12. Linear interpolation between the
    two neighbouring centers (exact for a line, second-order for the parabola:
    at 24 samples the neighbours are 7.5 min either side; at 60, 3 min), and
    linear EXTRAPOLATION from the first/last pair for the two endpoints that lie
    outside the centers (half a substep beyond), clamped at zero since OH is a
    concentration. A (nlev,nlat,nlon) field (OH_SZA=0, constant over the step)
    just repeats.
    """
    if oh_arr.ndim != 4:
        return np.repeat(np.asarray(oh_arr).reshape(ncell, 1), n_end, axis=1)
    nin = oh_arr.shape[0]
    oh_f = np.asarray(oh_arr).reshape(nin, ncell)
    if nin < 2:
        return np.repeat(oh_f.reshape(ncell, 1), n_end, axis=1)
    c_in = (np.arange(nin) + 0.5) / nin
    t_out = np.arange(n_end) / (n_end - 1)
    j = np.clip(np.searchsorted(c_in, t_out) - 1, 0, nin - 2)      # (n_end,)
    w = (t_out - c_in[j]) / (c_in[j + 1] - c_in[j])
    out = oh_f[j] * (1.0 - w)[:, None] + oh_f[j + 1] * w[:, None]  # (n_end, ncell)
    return np.maximum(out, 0.0).T


def _oh_endpoints_and_mean(oh3d, nlev, nlat, nlon):
    """(13, ncell) endpoints and (ncell,) step-mean OH as GPU tensors, without
    materializing the (nsub, nlev, nlat, nlon) broadcast coupling.oh_sza hands
    over: the SZA field is level-independent, so it is computed on (nsub, nlat,
    nlon) and expanded over levels on the device."""
    oh = np.asarray(oh3d)
    ncol = nlat * nlon
    if oh.ndim == 4 and oh.strides[1] == 0:               # broadcast over levels
        base = np.ascontiguousarray(oh[:, 0])              # (nsub, nlat, nlon)
        e = _oh_to_endpoints(base[:, None], ncol)          # (ncol, 13)
        m = base.reshape(oh.shape[0], ncol).mean(0)
        e_t = torch.from_numpy(e).to(_DEV).T[:, None, :].expand(N_END, nlev, ncol).reshape(N_END, -1)
        m_t = torch.from_numpy(m).to(_DEV)[None, :].expand(nlev, ncol).reshape(-1)
        return e_t, m_t
    ncell = nlev * ncol
    e = _oh_to_endpoints(oh, ncell)                        # (ncell, 13)
    m = oh.reshape(-1, ncell).mean(0) if oh.ndim == 4 else oh.reshape(ncell)
    return torch.from_numpy(e).to(_DEV).T.contiguous(), torch.from_numpy(np.ascontiguousarray(m)).to(_DEV)


@torch.no_grad()
def emulate_cells(Nk, Mk, H, S, T, P, RH, OH13, OHbar=None, raw_hook=None):
    """One 6 h emulator step over ncell boxes. Inputs per-box, either torch GPU
    tensors (float64) or numpy: Nk, Mk (ncell, 40); H, S, T, P, RH (ncell,);
    OH13 (ncell, 13). Returns Nk2, Mk2, H2, S2 in the same container type.
    OHbar (ncell,), the step-mean OH, is used by EMU_PROJECT=chem (default:
    the trapezoid of OH13).

    Everything but the network runs in float64 on the device; the floors are
    applied before the float32 cast (1e-30 is representable, a product like
    Nk * xk is not).
    """
    was_np = not torch.is_tensor(Nk)
    Nk, Mk, H, S, T, P, RH, OH13 = (_to_torch(a) for a in (Nk, Mk, H, S, T, P, RH, OH13))
    ncell = Nk.shape[0]
    tt = [time.time()]
    lin = torch.empty((ncell, N_STATE), dtype=torch.float32, device=_DEV)
    lin[:, 0:40] = Nk.clamp(min=N_FLOOR); lin[:, 40:80] = Mk.clamp(min=M_FLOOR)
    lin[:, 80] = H.clamp(min=M_FLOOR); lin[:, 81] = S.clamp(min=S_FLOOR)
    s_all = _norm.norm_state(torch.log(lin)); del lin
    c_all = torch.empty((ncell, N_END, N_COND), dtype=torch.float32, device=_DEV)
    c_all[:, :, 0] = torch.log(P).float()[:, None]; c_all[:, :, 1] = T.float()[:, None]
    c_all[:, :, 2] = RH.clamp(RH_MIN, 1.0).float()[:, None]
    c_all[:, :, 3] = torch.log(OH13.clamp(min=OH_FLOOR)).float()
    c_all[:, :, 4] = _LOG_ZERO; c_all[:, :, 5] = _LOG_ZERO
    c_all = _norm.norm_cond(c_all)
    torch.cuda.synchronize(); tt.append(time.time())                  # prep + normalize
    out = torch.empty((ncell, N_STATE), dtype=torch.float32, device=_DEV)
    for a in range(0, ncell, EMU_BS):                                 # fixed batch: pad the tail
        b = min(a + EMU_BS, ncell); n = b - a
        s = s_all[a:b]; c = c_all[a:b]
        if n < EMU_BS:
            s = torch.cat([s, s[:1].expand(EMU_BS - n, -1)], 0)
            c = torch.cat([c, c[:1].expand(EMU_BS - n, -1, -1)], 0)
        out[a:b] = _step6h(s, c)[:n]
    torch.cuda.synchronize(); tt.append(time.time())                  # forward
    E = torch.exp(torch.clamp(_norm.denorm_state(out), max=60.0)).double()
    # "empty" below the occupancy threshold the model was trained to represent;
    # a bin with no number carries no mass (self-consistent output, and the
    # budget attributes it to 'micro', where it belongs)
    empty = E[:, 0:40] < NEPS_EMU
    Nk2 = torch.where(empty, 0.0, E[:, 0:40]); Mk2 = torch.where(empty, 0.0, E[:, 40:80])
    H2, S2 = E[:, 80], E[:, 81]
    del s_all, c_all, out, E
    if raw_hook is not None:                                          # EMU_ONLINE_CORR: sees + corrects the raw answer
        Nk2, Mk2 = raw_hook(Nk2, Mk2)
    if _BIN_CORR is not None:                                         # per-bin bias factors (EMU_BIN_CORR)
        Nk2 = Nk2 * _BIN_CORR[0]; Mk2 = Mk2 * _BIN_CORR[1]
    if _NYQ_P is not None:                                            # project out the 2-bin alternating mode (EMU_NYQ_DAMP)
        Nk2 = _nyq_damp(Nk2); Mk2 = _nyq_damp(Mk2)
    if EMU_SAW_DAMP > 0:                                              # local 2-bin oscillation damper (EMU_SAW_DAMP)
        Nk2 = _saw_damp(Nk2); Mk2 = _saw_damp(Mk2)
    if EMU_PROJECT != '0':
        Nk2, Mk2, H2, S2 = project(Nk, Mk, H, S, Nk2, Mk2, H2, S2, sulfur=(EMU_PROJECT == 'sulfur'))
    if EMU_PROJECT == 'chem':
        if OHbar is None:
            OHbar = (0.5 * (OH13[:, 0] + OH13[:, -1]) + OH13[:, 1:-1].sum(1)) / (N_END - 1)
        else:
            OHbar = _to_torch(OHbar)
        k1 = _to_torch(calc_k1_so2_oh(_to_jax(T.contiguous()), _to_jax(P.contiguous()), _to_jax(RH.contiguous())))
        Nk2, Mk2, H2, S2 = project_chem(Mk, H, S, Nk2, Mk2, H2, k1, OHbar, T, P)
    torch.cuda.synchronize(); tt.append(time.time())                  # denorm + project
    emulate_cells.last_timing = dict(zip(('prep_norm', 'forward', 'denorm_project'), np.diff(tt)))
    if was_np:
        return tuple(x.cpu().numpy() for x in (Nk2, Mk2, H2, S2))
    return Nk2, Mk2, H2, S2


emulate_cells.last_timing = {}


def project(Nk, Mk, H, S, Nk2, Mk2, H2, S2, sulfur=False):
    """Two-moment consistency (+ per-box sulfur conservation if `sulfur`) on an
    emulator answer (see EMU_PROJECT above). Torch float64 on the device."""
    Mk2 = torch.minimum(torch.maximum(Mk2, Nk2 * _XK[:-1]), Nk2 * _XK[1:])
    if not sulfur:
        project.last = {}
        return Nk2, Mk2, H2, S2
    gas_in = S * _S_SO2 + H * _S_H2SO4
    aer_in = Mk.sum(1) * _S_SO4
    gas_out = S2 * _S_SO2 + H2 * _S_H2SO4
    g = torch.clamp(gas_in / gas_out.clamp(min=1e-300), max=1.0)      # gas sulfur cannot rise
    S2 = S2 * g; H2 = H2 * g
    aer_target = aer_in + (gas_in - gas_out * g)                      # aerosol gets what the gas lost
    aer_out = Mk2.sum(1) * _S_SO4
    f = torch.where(aer_out > 0.0, aer_target / aer_out.clamp(min=1e-300), torch.ones_like(aer_out))
    Nk2 = Nk2 * f[:, None]; Mk2 = Mk2 * f[:, None]
    project.last = dict(frac_corr_1pct=float(((f - 1.0).abs() > 0.01).double().mean()),
                        f_p50=float(f.median()), f_p01=float(torch.quantile(f, 0.01)),
                        f_p99=float(torch.quantile(f, 0.99)), gas_capped=int((g < 1.0).sum()))
    return Nk2, Mk2, H2, S2


project.last = {}


from tomas_jax.fast.condensation import _condensation_sink_cell as _cs_cell   # noqa: E402
from tomas_jax.fast.config import SRTSO4 as _SRTSO4                              # noqa: E402
_cs_batched = jax.jit(jax.vmap(lambda n, m, t, p: _cs_cell(n, m, t, p, C.BOXVOL, jnp.asarray(C.XK), C.ALPHA_COND)))


def cond_sink(Nk, Mk, T, P):
    """The physics' own H2SO4 condensation sink (Fuchs-Sutugin, binary density,
    ALPHA_COND) per cell: (CS (ncell,) 1/s, sinkfrac (ncell, 40)). Torch in,
    torch out; the dry SO4 mass is used (water column 0)."""
    Mk3 = torch.zeros((Nk.shape[0], Nk.shape[1], 2), dtype=torch.float64, device=_DEV)
    Mk3[:, :, _SRTSO4] = Mk
    cs, sf = _cs_batched(_to_jax(Nk.contiguous()), _to_jax(Mk3), _to_jax(T.contiguous()), _to_jax(P.contiguous()))
    return _to_torch(cs), _to_torch(sf)


def project_chem(Mk, H, S, Nk2, Mk2, H2, k1, OHbar, T, P):
    """EMU_PROJECT=chem (see above): exact SO2 decay; H2SO4 from the exact
    production / condensation-sink relaxation (the physics' own sink on the
    mean of the input and output distributions) instead of the network's
    H2SO4 (keeping the network's H2SO4, capped at what exists, raised the
    active-cell mass p90 from 5.9 to 9.9 %: its under-predicted gas acid was
    being forced into the aerosol); the aerosol gets exactly the sulfur the
    gas lost -- the network's per-bin mass change is kept and the budget
    residual deposited by the physics' sink fraction, then the two-moment
    bounds are restored per EMU_CHEM_NCLIP (0 legacy mass re-clip, 1 conserving).
    Torch float64 on the device."""
    S2 = S * torch.exp(-k1 * OHbar * C.DT_MICRO)
    produced = (S - S2) * (_MW_H2SO4 / _MW_SO2)                     # kg H2SO4 made in the step
    avail = H + produced
    CS, sf = cond_sink(Nk2, Mk2, T, P)
    hss = (produced / C.DT_MICRO) / CS.clamp(min=1e-30)
    H2 = torch.where(CS > 0.0, hss + (H - hss) * torch.exp(-CS * C.DT_MICRO), avail)
    H2 = torch.minimum(H2, avail)
    dM_target = (avail - H2) * (_MW_SO4 / _MW_H2SO4)                # kg SO4 the aerosol must gain (>= 0)
    resid = dM_target - (Mk2.sum(1) - Mk.sum(1))
    area = Nk2 * _DP2
    wgt = torch.where(sf.sum(1, keepdim=True) > 0.0, sf, area / area.sum(1, keepdim=True).clamp(min=1e-300))
    lost = torch.zeros_like(resid)
    if EMU_CHEM_NCLIP == 1:
        # a NEGATIVE residual (the network added more mass than the budget allows) is taken out
        # multiplicatively, never additively: the additive deposit drove bins below zero, which the
        # legacy mass clip silently lifted back (part of the leak) and which a number-side bound
        # turns into negative number -> NaN (both 2026-10-17 attempts failed that way at step 2)
        target_tot = Mk.sum(1) + dM_target
        scale = (target_tot / Mk2.sum(1).clamp(min=1e-300)).clamp(min=0.0)
        Mk2 = torch.where((resid > 0.0)[:, None], Mk2 + resid.clamp(min=0.0)[:, None] * wgt, Mk2 * scale[:, None]).clamp(min=0.0)
    else:
        Mk2 = Mk2 + resid[:, None] * wgt
    if EMU_CHEM_NCLIP == 1:                                         # conserving: number down where mass is short, mass redistributed where it is in excess
        Nk2 = torch.where(Mk2 < Nk2 * _XK[:-1], Mk2 / _XK[:-1], Nk2)
        hi = Nk2 * _XK[1:]
        excess = (Mk2 - hi).clamp(min=0.0); Mk2 = Mk2 - excess; extra = excess.sum(1)
        for _ in range(2):
            room = (hi - Mk2).clamp(min=0.0); tot = room.sum(1)
            add = torch.where(tot > 0.0, extra / tot.clamp(min=1e-300), torch.zeros_like(tot))[:, None] * room
            add = torch.minimum(add, room); Mk2 = Mk2 + add; extra = (extra - add.sum(1)).clamp(min=0.0)
        lost = extra
    else:                                                           # legacy: re-clip mass (creates/destroys sulfur)
        Mk2 = torch.minimum(torch.maximum(Mk2, Nk2 * _XK[:-1]), Nk2 * _XK[1:])
    rel = resid.abs() / Mk.sum(1).clamp(min=1e-30)
    project.last = dict(chem_resid_rel_p50=float(rel.median()), chem_resid_rel_p99=float(torch.quantile(rel, 0.99)),
                        chem_gain_ratio=float((Mk2.sum(1) - Mk.sum(1)).sum() / dM_target.sum().clamp(min=1e-300)),
                        chem_lost=float(lost.sum() / Mk.sum().clamp(min=1e-300)))
    return Nk2, Mk2, H2, S2


def cell_scores(Nk, Mk, H, S, T, P, RH, OHbar):
    """The per-cell quantities the triage and the nucleation routing rank on.
    All per-box torch float64 on the device; returns a dict with k1 (cm3/s),
    CS (1/s, free-molecular surface-area sink), Hss (molec cm-3, the larger of
    the cell's H2SO4 and its production / sink steady state), pot (sulfur the
    step can add relative to the aerosol), fine (number fraction in bins 0-9)."""
    k1 = _to_torch(calc_k1_so2_oh(_to_jax(T), _to_jax(P), _to_jax(RH)))   # cm3 molec-1 s-1
    Ntot = Nk.sum(1); Mtot = Mk.sum(1)
    dS = S * (1.0 - torch.exp(-k1 * OHbar * C.DT_MICRO))
    pot = (dS * (_MW_SO4 / _MW_SO2) + H * (_MW_SO4 / _MW_H2SO4)) / Mtot.clamp(min=1e-30)
    cbar = torch.sqrt(8.0 * _KB * T / (np.pi * _MW_H2SO4 * 1e-3 / _NA))         # m/s
    CS = (cbar / 4.0) * (np.pi * (Nk * _DP2).sum(1))                            # s-1 (free-molecular, alpha=1)
    prod = k1 * OHbar * (S / (_MW_SO2 * 1e-3) * _NA / 1e6)                      # molec cm-3 s-1
    Hss = torch.maximum(H / (_MW_H2SO4 * 1e-3) * _NA / 1e6, prod / CS.clamp(min=1e-30))
    fine = Nk[:, :10].sum(1) / Ntot.clamp(min=1e-30)
    return dict(k1=k1, CS=CS, Hss=Hss, pot=pot, fine=fine, Ntot=Ntot)


def quiet_mask(Nk, Mk, H, S, T, P, RH, OHbar, sc=None):
    """The triage test (see EMU_TRIAGE above). All per-box torch float64 on the
    device; returns (mask (ncell,) bool, k1 (ncell,) cm3/s, CS (ncell,) 1/s)."""
    if sc is None:
        sc = cell_scores(Nk, Mk, H, S, T, P, RH, OHbar)
    m = ((sc['pot'] < TRIAGE_POT) & (sc['Ntot'] / C.BOXVOL < TRIAGE_N)
         & (sc['Hss'] < TRIAGE_HSS) & (sc['fine'] < TRIAGE_FINE))
    return m, sc['k1'], sc['CS']


def quiet_step(Nk, Mk, H, S, T, OHbar, k1, CS):
    """The cheap exact step for quiet cells: pseudo-first-order SO2 + OH, the
    acid condensed onto the existing bins by surface area (free-molecular
    uptake), H2SO4 left at its production/sink steady state, number unchanged."""
    dec = torch.exp(-k1 * OHbar * C.DT_MICRO)
    S2 = S * dec
    produced = (S - S2) * (_MW_H2SO4 / _MW_SO2)
    avail = H + produced
    # dH/dt = Pr - CS*H with constant production Pr over the step (exact):
    #   H(dt) = Hss + (H0 - Hss) exp(-CS dt),  Hss = Pr / CS
    hss = (produced / C.DT_MICRO) / CS.clamp(min=1e-30)
    relax = torch.exp(-CS * C.DT_MICRO)
    H2 = torch.where(CS > 0.0, hss + (H - hss) * relax, avail)
    H2 = torch.minimum(H2, avail)
    deposit = (avail - H2) * (_MW_SO4 / _MW_H2SO4)
    area = Nk * _DP2
    wgt = area / area.sum(1, keepdim=True).clamp(min=1e-300)
    Mk2 = Mk + deposit[:, None] * wgt
    return Nk, Mk2, H2, S2


def _oh_substeps_for(oh3d, idx, nlev, nlat, nlon, nsub_out):
    """(nsub_out, k) per-inner-step OH for the cells `idx` (flat indices), from
    whatever coupling handed over (3-D constant, 4-D, or the level-broadcast
    4-D view), resampled onto the fast engine's inner grid as tomas_fast does."""
    oh = np.asarray(oh3d); ncol = nlat * nlon; idx = np.asarray(idx)
    if oh.ndim != 4:
        return np.repeat(oh.reshape(-1)[idx][None, :], nsub_out, axis=0)
    if oh.strides[1] == 0:                                   # broadcast over levels
        sub = oh[:, 0].reshape(oh.shape[0], ncol)[:, idx % ncol]
    else:
        sub = oh.reshape(oh.shape[0], -1)[:, idx]
    return np.ascontiguousarray(C._oh_to_substeps(sub[:, None, None, :], len(idx), nsub_out=nsub_out).T)


def physics_cells(Nk, Mk, H, S, T, P, RH, OHsub):
    """tomas_jax.fast on a SUBSET of cells, with the production knobs of
    tomas_fast.run_microphysics_full_fast. Per-box torch tensors in (Nk, Mk
    (k, 40); H, S, T, P, RH (k,)), OHsub (N_FAST_STEPS, k) numpy; per-box torch
    tensors out. The batch is padded with empty cells to a multiple of
    EMU_PHYS_PAD (adaptive substepping makes them free) so the jit cache hits."""
    k = Nk.shape[0]; kp = -(-k // EMU_PHYS_PAD) * EMU_PHYS_PAD
    f = lambda t: t.detach().cpu().numpy()
    Nk_np = np.zeros((kp, 40)); Nk_np[:k] = f(Nk)
    Mk3 = np.zeros((kp, 40, 2)); Mk3[:k, :, TF.SRTSO4] = f(Mk)
    Gc = np.zeros((kp, 2)); Gc[:k, TF.GH2SO4] = f(H); Gc[:k, TF.GSO2] = f(S)
    T_np = np.full(kp, 230.0); T_np[:k] = f(T)
    P_np = np.full(kp, 5000.0); P_np[:k] = f(P)
    RH_np = np.full(kp, 0.3); RH_np[:k] = f(RH)
    OH_np = np.zeros((OHsub.shape[0], kp)); OH_np[:, :k] = OHsub
    state = TF.FastState.create(Nk=Nk_np, Mk=Mk3, Gc=Gc, xk=np.asarray(C.XK), temp=T_np, pres=P_np,
                                boxvol=C.BOXVOL, rh=RH_np)
    state2, diags = TF.run_fast(state, TF.N_FAST_STEPS, dt=TF.FAST_DT, oh_conc=OH_np,
                                n_cell_chunks=(max(1, min(EMU_PHYS_CHUNKS, kp // EMU_PHYS_CHUNK_MIN)) if EMU_PHYS_CHUNK_MIN > 0 else EMU_PHYS_CHUNKS),
                                sort_by_coag_cost=TF.FAST_SORT,
                                alpha=C.ALPHA_COND, fn_scale=TF.FAST_FN_SCALE,
                                coag_sub_cap=TF.FAST_COAG_CAP, cond_sub_cap=TF.FAST_COND_CAP,
                                coag_c_max=TF.FAST_COAG_CMAX)
    g = lambda a: torch.from_numpy(np.ascontiguousarray(np.asarray(a)[:k], dtype=np.float64)).to(_DEV)
    physics_cells.last = {'cells': k, 'padded': kp,
                          'coag_cap_hit': int(np.asarray(diags['coag_cap_hit']).sum()),
                          'cond_cap_hit': int(np.asarray(diags['cond_cap_hit']).sum())}
    return g(state2.Nk), g(state2.Mk[:, :, TF.SRTSO4]), g(state2.Gc[:, TF.GH2SO4]), g(state2.Gc[:, TF.GSO2])


physics_cells.last = {}


class _DriftState:
    """Call counter (EMU_PHYS_EVERY / EMU_PHYS_ROT) and the EMU_ONLINE_CORR moving sums."""
    def __init__(self):
        self.calls = 0; self.A = None; self.B = None; self.last = {}

    def make_hook(self, act, samp, sampN, sampM, nlev, nlat, nlon, ncell):
        nreg = nlev * EMU_CORR_NLAT
        cell = torch.arange(ncell, device=_DEV)
        lat = (cell % (nlat * nlon)) // nlon
        region = (cell // (nlat * nlon)) * EMU_CORR_NLAT + (lat * EMU_CORR_NLAT) // nlat
        reg_act = region[act]
        pos = torch.full((ncell,), -1, dtype=torch.long, device=_DEV)
        pos[act] = torch.arange(act.numel(), device=_DEV)
        sp = pos[samp]                                # sample rows inside the act block
        rs = region[samp]
        st = self

        def hook(Nk2, Mk2):
            num = torch.zeros((2, nreg, 40), dtype=torch.float64, device=_DEV)
            den = torch.zeros_like(num)
            num[0].index_add_(0, rs, sampN); num[1].index_add_(0, rs, sampM)
            den[0].index_add_(0, rs, Nk2[sp]); den[1].index_add_(0, rs, Mk2[sp])
            if st.A is None:
                st.A, st.B = num, den
            else:
                a = EMU_CORR_EMA
                st.A = (1 - a) * st.A + a * num; st.B = (1 - a) * st.B + a * den
            tot = st.B.sum(1, keepdim=True)
            ok = st.B > EMU_CORR_MINFRAC * tot
            c = torch.where(ok, st.A / st.B.clamp(min=1e-300), torch.ones_like(st.A))
            c = c.clamp(1.0 / EMU_CORR_CLAMP, EMU_CORR_CLAMP)
            if EMU_CORR_B0 > 0:
                c[:, :, :EMU_CORR_B0] = 1.0
            cN = c[0][reg_act]; cM = c[1][reg_act]
            w = den[0].sum(0); gN = (c[0] * den[0]).sum(0) / w.clamp(min=1e-300)
            st.last = dict(n_samp=int(samp.numel()),
                           gN_b0_5=[round(float(x), 3) for x in gN[:6]],
                           gN_b21_29=[round(float(x), 3) for x in gN[21:30]],
                           raw_ratioN_global=round(float(num[0].sum() / den[0].sum().clamp(min=1e-300)), 4),
                           raw_ratioM_global=round(float(num[1].sum() / den[1].sum().clamp(min=1e-300)), 4))
            return Nk2 * cN, Mk2 * cM
        return hook


_drift = _DriftState()


def run_microphysics_full_emulator(num, mas, so2, h2so4, temp3d, pres3d, rh3d,
                                   oh3d, wgt3d):
    """Drop-in for coupling.run_microphysics_full backed by the learned 6 h operator.

    Same signature/return contract and the SAME mixing-ratio<->per-box
    conversion + two-moment consistency clip + clip diagnostic as
    tomas_fast.run_microphysics_full_fast; only the engine differs, and the
    bookkeeping runs on the GPU (the state arrives as jax device arrays and is
    handed to torch zero-copy; the result goes back the same way, so
    coupling's `jnp.asarray` on it is free).
    """
    t_all = time.time()
    nbin, nlev, nlat, nlon = num.shape
    assert nbin == 40, f"expected 40 bins, got {nbin}"
    ncell = nlev * nlat * nlon

    T_f = _to_torch(temp3d).reshape(ncell); P_f = _to_torch(pres3d).reshape(ncell)
    RH_f = _to_torch(rh3d).reshape(ncell)
    rho_f = P_f / (C.RD * T_f)
    inv_rho = 1.0 / rho_f.clamp(min=1e-30)
    OH13, OHbar = _oh_endpoints_and_mean(oh3d, nlev, nlat, nlon)     # (13, ncell), (ncell,)

    Nk_all = _to_torch(num).reshape(nbin, ncell).T * rho_f[:, None]   # (ncell, nbin) per box
    Mt_all = _to_torch(mas).reshape(nbin, ncell).T * rho_f[:, None]
    S_all = _to_torch(so2).reshape(ncell) * rho_f
    H_all = _to_torch(h2so4).reshape(ncell) * rho_f

    # ---- two-moment consistency clip, exactly as tomas_fast / _micro_cell ----
    Nk_eff = torch.where(Nk_all > C.NEPS_N, Nk_all, 0.0)
    Mt_clip = torch.minimum(torch.maximum(Mt_all, Nk_eff * _XK[:-1]), Nk_eff * _XK[1:])
    dM_mr = (Mt_clip - Mt_all) * inv_rho[:, None]
    w = _to_torch(wgt3d).reshape(ncell)[:, None]
    clip_add = float((dM_mr.clamp(min=0.0) * w).sum())
    clip_rem = float((dM_mr.clamp(max=0.0) * w).sum())
    del Nk_all, Mt_all, dM_mr

    t0 = time.time()
    out_N = torch.empty_like(Nk_eff); out_M = torch.empty_like(Mt_clip)
    out_H = torch.empty_like(H_all); out_S = torch.empty_like(S_all)
    rest = torch.ones(ncell, dtype=torch.bool, device=_DEV)
    n_phys = 0; dt_phys = 0.0
    sc = cell_scores(Nk_eff, Mt_clip, H_all, S_all, T_f, P_f, RH_f, OHbar) if EMU_TRIAGE else None
    call = _drift.calls; _drift.calls += 1
    if EMU_PHYS_EVERY > 0 and call % EMU_PHYS_EVERY == EMU_PHYS_EVERY - 1:
        tp = time.time()
        r = TF.run_microphysics_full_fast(num, mas, so2, h2so4, temp3d, pres3d, rh3d, oh3d, wgt3d)
        if os.environ.get('DEBUG') or os.environ.get('PROFILE'):
            print(f"  [emu] call {call}: EMU_PHYS_EVERY={EMU_PHYS_EVERY} refresh -> physics on all {ncell} cells in {time.time() - tp:.2f}s", flush=True)
        return r
    samp = None
    if EMU_PHYS_ROT > 0 or EMU_ONLINE_CORR > 0 or EMU_PHYS_HSS > 0:
        qm0 = quiet_mask(Nk_eff, Mt_clip, H_all, S_all, T_f, P_f, RH_f, OHbar, sc)[0] if EMU_TRIAGE else torch.zeros(ncell, dtype=torch.bool, device=_DEV)
    if _ANY_PHYS and (EMU_PHYS_TOP > 0 or EMU_PHYS_NUM > 0 or EMU_PHYS_COARSE > 0 or EMU_PHYS_ROT > 0
                      or EMU_ONLINE_CORR > 0 or EMU_PHYS_HSS > 0):
        # the plume: top fraction of cells by total sulfur per box -> the physics;
        # then the dense ones: top fraction of the rest by particle number per box
        sel = []
        if EMU_PHYS_TOP > 0:
            sulf = Mt_clip.sum(1) + S_all * (_MW_SO4 / _MW_SO2) + H_all * (_MW_SO4 / _MW_H2SO4)
            sel.append(torch.topk(sulf, max(1, int(round(EMU_PHYS_TOP * ncell)))).indices)
        if EMU_PHYS_NUM > 0:
            ntot = Nk_eff.sum(1)
            if sel:
                ntot[sel[0]] = -1.0
            sel.append(torch.topk(ntot, max(1, int(round(EMU_PHYS_NUM * ncell)))).indices)
        if EMU_PHYS_COARSE > 0:
            cmass = Mt_clip[:, EMU_COARSE_B0:].sum(1)
            for s_ in sel:
                cmass[s_] = -1.0
            sel.append(torch.topk(cmass, max(1, int(round(EMU_PHYS_COARSE * ncell)))).indices)
        taken = torch.zeros(ncell, dtype=torch.bool, device=_DEV)
        for s_ in sel:
            taken[s_] = True
        if EMU_PHYS_HSS > 0:
            sc_ = sc if sc is not None else cell_scores(Nk_eff, Mt_clip, H_all, S_all, T_f, P_f, RH_f, OHbar)
            hs = torch.where(taken | qm0, -1.0, sc_['Hss'])
            hsel = torch.topk(hs, max(1, int(round(EMU_PHYS_HSS * ncell)))).indices
            sel.append(hsel); taken[hsel] = True
        if EMU_PHYS_ROT > 0:
            rot = (torch.arange(ncell, device=_DEV) % EMU_PHYS_ROT == call % EMU_PHYS_ROT) & ~taken & ~qm0
            sel.append(torch.nonzero(rot, as_tuple=True)[0]); taken |= rot
        if EMU_ONLINE_CORR > 0:
            cand = torch.nonzero(~taken & ~qm0, as_tuple=True)[0]
            g_ = torch.Generator(device=_DEV); g_.manual_seed(EMU_CORR_SEED * 1000003 + call)
            ns = max(1, int(round(EMU_ONLINE_CORR * cand.numel())))
            samp = cand[torch.randperm(cand.numel(), generator=g_, device=_DEV)[:ns]]
            sel.append(samp)
        pi = torch.cat(sel)
        rest[pi] = False
        if samp is not None:
            rest[samp] = True                         # the sample also goes through the network (raw answer compared)
        OHsub = _oh_substeps_for(oh3d, pi.cpu().numpy(), nlev, nlat, nlon, TF.N_FAST_STEPS)
        tp = time.time()
        r = physics_cells(Nk_eff[pi], Mt_clip[pi], H_all[pi], S_all[pi], T_f[pi], P_f[pi], RH_f[pi], OHsub)
        r_phys = r
        out_N[pi], out_M[pi], out_H[pi], out_S[pi] = r
        n_phys = int(pi.numel()); dt_phys = time.time() - tp
        if samp is not None:
            ns = samp.numel()
            sampN = r[0][-ns:]; sampM = r[1][-ns:]    # physics answer of the sample (last block of pi)
    if EMU_TRIAGE:
        qm, k1, CS = quiet_mask(Nk_eff, Mt_clip, H_all, S_all, T_f, P_f, RH_f, OHbar, sc)
        qm &= rest
        act = torch.nonzero(rest & ~qm, as_tuple=True)[0]
        qi = torch.nonzero(qm, as_tuple=True)[0]
        if qi.numel():
            r = quiet_step(Nk_eff[qi], Mt_clip[qi], H_all[qi], S_all[qi], T_f[qi], OHbar[qi], k1[qi], CS[qi])
            out_N[qi], out_M[qi], out_H[qi], out_S[qi] = r
    else:
        act = torch.nonzero(rest, as_tuple=True)[0]
    if act.numel():
        hook = None
        if samp is not None:
            hook = _drift.make_hook(act, samp, sampN, sampM, nlev, nlat, nlon, ncell)
        r = emulate_cells(Nk_eff[act], Mt_clip[act], H_all[act], S_all[act],
                          T_f[act], P_f[act], RH_f[act], OH13.T[act], OHbar=OHbar[act], raw_hook=hook)
        out_N[act], out_M[act], out_H[act], out_S[act] = r
        if samp is not None:                          # the sample keeps its physics answer
            out_N[pi], out_M[pi], out_H[pi], out_S[pi] = r_phys
            if os.environ.get('DEBUG') or os.environ.get('PROFILE'):
                print(f"  [emu] online corr: {_drift.last}", flush=True)
    n_engine = int(act.numel())
    if EMU_PHYS_BURST > 0 and act.numel():
        grow = out_N[act, :EMU_BURST_B1].sum(1) - Nk_eff[act, :EMU_BURST_B1].sum(1)
        if samp is not None:
            grow[torch.isin(act, samp)] = -1e300      # already physics
        nb = min(act.numel(), max(1, int(round(EMU_PHYS_BURST * ncell))))
        bi = act[torch.topk(grow, nb).indices]
        OHb = _oh_substeps_for(oh3d, bi.cpu().numpy(), nlev, nlat, nlon, TF.N_FAST_STEPS)
        tb = time.time()
        rb = physics_cells(Nk_eff[bi], Mt_clip[bi], H_all[bi], S_all[bi], T_f[bi], P_f[bi], RH_f[bi], OHb)
        if os.environ.get('DEBUG') or os.environ.get('PROFILE'):
            bn = out_N[bi, :EMU_BURST_B1].sum(); bp = rb[0][:, :EMU_BURST_B1].sum()
            print(f"  [emu] burst routing: {nb} cells re-run by physics in {time.time() - tb:.2f}s; their bins 0-{EMU_BURST_B1 - 1} N network/physics {float(bn / bp.clamp(min=1e-300)):.3f}, global N change {float((rb[0].sum() - out_N[bi].sum()) / out_N.sum()) * 100:+.3f} %", flush=True)
        out_N[bi], out_M[bi], out_H[bi], out_S[bi] = rb
        n_phys += nb
    dt_emu = time.time() - t0

    num2 = _to_jax((out_N * inv_rho[:, None]).T.reshape(nbin, nlev, nlat, nlon))
    mas2 = _to_jax((out_M * inv_rho[:, None]).T.reshape(nbin, nlev, nlat, nlon))
    so2_2 = _to_jax((out_S * inv_rho).reshape(nlev, nlat, nlon))
    h2so4_2 = _to_jax((out_H * inv_rho).reshape(nlev, nlat, nlon))
    for a in (num2, mas2, so2_2, h2so4_2):
        a.block_until_ready()
    if os.environ.get('DEBUG') or os.environ.get('PROFILE'):
        lt = emulate_cells.last_timing
        print(f"  [emu] {ncell} cells, network on {n_engine} ({n_engine / ncell * 100:.1f} %"
              f"{', triage' if EMU_TRIAGE else ''}"
              + (f", physics on {n_phys} in {dt_phys:.2f}s" if n_phys else '')
              + f"), one 6 h step in {time.time() - t_all:.2f}s "
              f"total ({(time.time() - t_all) / ncell * 1e6:.2f} us/cell); engine {dt_emu:.2f}s: "
              f"forward {lt.get('forward', float('nan')):.2f}s = "
              f"{lt.get('forward', float('nan')) / max(n_engine, 1) * 1e6:.2f} us/cell, "
              + ", ".join(f"{k} {v:.2f}s" for k, v in lt.items() if k != 'forward') + ")",
              flush=True)
        if project.last and 'f_p50' in project.last:
            pl = project.last
            print(f"  [emu] sulfur projection: aerosol scale f p01/p50/p99 {pl['f_p01']:.3f}/"
                  f"{pl['f_p50']:.3f}/{pl['f_p99']:.3f}, cells corrected >1 %: "
                  f"{pl['frac_corr_1pct']*100:.2f} %, gas capped: {pl['gas_capped']}", flush=True)
        elif project.last and 'chem_resid_rel_p50' in project.last:
            pl = project.last
            print(f"  [emu] chem projection: |budget residual|/M p50/p99 {pl['chem_resid_rel_p50']:.2e}/"
                  f"{pl['chem_resid_rel_p99']:.2e}, aerosol gain after clip / target {pl['chem_gain_ratio']:.4f}", flush=True)
    return num2, mas2, so2_2, h2so4_2, clip_add, clip_rem


def describe():
    """One-line engine description for the run header."""
    return (f"tomas-emulator BinTF d={EMU_D} x {EMU_NL} ({EMU_DTYPE}"
            f"{', compiled' if EMU_COMPILE else ', eager'}, bs {EMU_BS}; "
            f"load {T_LOAD:.1f}s; OH floor {OH_FLOOR:g} cm-3, RH >= {RH_MIN}, "
            f"SO2 floor {S_FLOOR:g} kg/box; output projection {EMU_PROJECT}; "
            + (f"triage ON: pot<{TRIAGE_POT:g} N<{TRIAGE_N:g} Hss<{TRIAGE_HSS:g} fine<{TRIAGE_FINE:g}"
               if EMU_TRIAGE else "triage OFF")
            + (f"; physics on top {EMU_PHYS_TOP * 100:g} % of cells by sulfur" if EMU_PHYS_TOP > 0 else '')
            + (f"; physics on top {EMU_PHYS_NUM * 100:g} % of the rest by particle number" if EMU_PHYS_NUM > 0 else '')
            + (f"; physics on top {EMU_PHYS_COARSE * 100:g} % of the rest by mass in bins >= {EMU_COARSE_B0}" if EMU_PHYS_COARSE > 0 else '')
            + (f"; physics refresh every {EMU_PHYS_EVERY} calls" if EMU_PHYS_EVERY > 0 else '')
            + (f"; physics on a rotating 1/{EMU_PHYS_ROT} slice" if EMU_PHYS_ROT > 0 else '')
            + (f"; physics on top {EMU_PHYS_HSS * 100:g} % of the rest by steady-state H2SO4" if EMU_PHYS_HSS > 0 else '')
            + (f"; burst routing: physics re-run on the top {EMU_PHYS_BURST * 100:g} % of cells by network bin 0-{EMU_BURST_B1 - 1} N growth" if EMU_PHYS_BURST > 0 else '')
            + (f"; ONLINE bias correction (bins >= {EMU_CORR_B0}): physics on {EMU_ONLINE_CORR * 100:g} % of network cells, {EMU_CORR_NLAT} lat bands x lev, EMA {EMU_CORR_EMA:g}, clamp {EMU_CORR_CLAMP:g}" if EMU_ONLINE_CORR > 0 else '')
            + (f" ({EMU_PHYS_CHUNKS} physics chunk{'s' if EMU_PHYS_CHUNKS > 1 else ''})" if _ANY_PHYS else '')
            + ")\n"
            f"             ckpt {EMU_CKPT}")
