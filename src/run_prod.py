#!/usr/bin/env python3
"""Production launcher: resolve the run configuration, then exec driver_fast.py.

Was run_prod.sh until 2026-09-01. Same job, same defaults, same exec-at-the-end
shape -- but every knob the model reads is now a documented `--flag` instead of
an environment variable you had to already know about, and `--help` is the
inventory. `--dry-run` prints the resolved environment without launching.

    mkdir -p runs/prod90d && cd runs/prod90d
    python3 /path/to/repo/src/run_prod.py --inj-so2 10 --out-tag prod90d

WHY IT STILL EXECS. Two reasons, both load-bearing:
  * LD_LIBRARY_PATH must be set BEFORE the process starts. glibc reads it once
    at startup, so setting os.environ['LD_LIBRARY_PATH'] in THIS process and
    then importing jax here would not help it find libcuda.so.1 -- it would
    silently fall back to CPU, the exact failure this launcher exists to prevent.
  * coupling.py reads every knob once at module import with os.environ.get. The
    environment therefore has to be final before the import, which is what
    handing a fully-built env to execve guarantees.
Flags are translated to environment variables and nothing else: the model's
configuration remains env-driven and self-describing in the run header.

PRECEDENCE, highest first:  --flag  >  an env var already set  >  the
production preset below  >  coupling.py's own default. So `--fast-sort 0` and
`FAST_SORT=0 run_prod.py` both work, and a knob this launcher does not preset is
simply passed through untouched.

============================================================================
PRODUCTION 90-DAY RUN
============================================================================
360 steps x 6 h = 2160 h = 90 days, ~3.5 h wall clock on one H100 at the
34.9 s/step measured over a full 1-year run (2026-08-28, runs/prod1yr/):
  micro 16.5s (47%)  radiation 9.4s (27%)  advection 7.4s (21%)  rest 1.7s
This read '~33-36 h at ~340-370 s/step (microphysics ~94% of it)' from
2026-08-04 until 2026-08-28. That was true of the engine THEN. tomas-jax's
GPU-fast work on 2026-08-13 (547s -> 23.1s on its 1M-cell benchmark, 23.6x) cut
microphysics ~19x, and this repo pinned that submodule on 2026-08-27 -- so the
old figure survived nine days past the change that invalidated it, and a further
two weeks in the docs that quoted it. Microphysics is no longer the overwhelming
majority of a step; it is now under half.
NB cost is FLAT over a year: quarter means 35.8/34.1/35.5/34.2 s per step, no
stage drifting >15%. Burden growth does NOT make steps more expensive.

CONFIG = coupling.py / driver_fast.py DEFAULTS except the preset below.
For the record, what the defaults now give (validated by an 18 h A/B pair):
  AER_SRC=mam4        per-step dynamic IC + BC. No CARMA anywhere -- only ~1
                      week of CARMA output exists, so it could only ever be a
                      STATIC reservoir, and a frozen reservoir is the root cause
                      of the old mass leak.
  INIT_BIN=so4        NOW THE DEFAULT (flipped 2026-07-29). Preset explicitly
                      anyway so the log records it: the old dgnum default binned
                      MAM4 NUMBER by dgnumwet and set mas = num*MMID without ever
                      reading so4_a*, inflating sulfate mass 4.29x (mode 3 alone
                      6.68x) and pushing Dp(massw) to ~1920 nm instead of ~890.
  P_LO/HI = 1/150 hPa 24 levels, 1,327,104 cells. The 1 hPa top is effectively
                      aerosol-free inflow (MAM4 3.2e-17 kg/kg, 7 orders below
                      13 hPa) so no knob is needed there.
  OH                  diurnal parabola in cos(SZA), OH_PEAK=2.3e6, 60 samples
                      per step (6 min). NOT CESM's OH -- a parametrization
                      sitting in front of the whole SO2 -> H2SO4 -> aerosol chain.
  BC_BOT_AER=1.0      bottom-face aerosol inflow at full MAM4 strength.
  ADV_WCONT=1 -> BC_EDGE=open   both vertical faces are real FLUX boundaries
                      (inflow served at the reservoir value, free outflow), not
                      Dirichlet clamps. This is what removed the leaking clamp:
                      the budget's `bc` term is now +0.00e+00 exactly.
  -> BC_GAS=flux      NEW DEFAULT 2026-07-30, derived from BC_EDGE so the gases
                      cannot desync from the aerosol. The old clamp beside open
                      aerosol faces was an unbounded gas SOURCE at a level whose
                      particles were free to leave: the 13.3 hPa top level went
                      0.3% -> ~50% of the model's TOTAL number in 24 h as a
                      6-8 nm mode. Cost: the gases are no longer pinned to CESM.
  ADV_SCHEME=ffsl     NEW DEFAULT 2026-10-15: fct_lr's flux form at large Courant
                      number (ADV_FFSL_K=24 sub-steps on the two step snapshots,
                      ADV_SL_HOURLY=0) instead of lr's 70-250 CFL sub-steps.
                      advect 1.5 s/step instead of 7.0; 30 days within 0.2 % of
                      lr on the budgets, 2 % on every quantity (PROCESSES.md 1.9).
                      --adv-scheme lr is the previous default, byte for byte.
  MICRO               tomas_jax.fast.run_fast, 60 inner steps x 360 s. ~20x
                      faster than physical TOMAS and stable, but its nucleation
                      is BINARY, not ternary.
  RADIATION           physical RRTMGP + Mie, RAD_MODE=anomaly. Under AER_SRC=mam4
                      the anomaly BASELINE is time-varying, so the reported
                      forcing is defined differently than in zonal90d.

HOW TO READ THE OUTPUT -- four standing caveats:
  * the first ~week is cold-start spin-up (N/N0 was still climbing at 19.5 by 18 h);
  * 90 days is a TRANSIENT, not a steady state -- the sink is not
    burden-proportional (loss ~ M^0.35), so no plateau is readable from it;
  * there is NO WET REMOVAL anywhere. Settling and transport out of the band are
    the only sinks, so aerosol settling into the 100-150 hPa layer just lingers;
  * AOD550 came out 0.0039 in the 18 h test vs a ~0.005-0.01 quoted background.
    Uncalibrated -- fine for the run, matters before quoting absolute forcing.

MEMORY / SHARED-GPU SETTINGS (none can change results -- micro is per-cell
independent, so chunk grouping is numerically irrelevant):
  FAST_SORT=1         stiffness sort, worth ~27% of micro. It does one unchunked
                      ~8 GB allocation, so `--fast-sort 0` is the first thing to
                      try if the card is already loaded -- it OOMs when another
                      job holds most of the memory.
  --gpu 0             ONE card, pinned, rather than auto-selected: the card a
                      long job lands on should never be a function of what
                      happened to be idle at launch.
  FAST_CELL_CAP=50000 lower than the 250000 default, and FASTER, not slower:
                      micro went only 1.3x for 2.18x the cells, because the
                      vmapped adaptive coag while_loop runs every lane to the
                      slowest one, so 27 small chunks waste far less than 3 big.
  INJ_ZONAL=1         zonal ring rather than the default POINT source, which is
                      5.6x slower and drives runaway nucleation.

ADV_VPOS=1 -- VERTICAL POSITIVITY LIMITER. Now the default in fct_lr.py; preset
  explicitly so the log records it. Without it the number field is not usable:
  fct_lr's vertical remap is exactly CONSERVATIVE but not POSITIVE, so it
  undershoots negative on the steep ultrafine gradient at the injection ring and
  coupling.py's floor clips those negatives and CREATES number. Measured: 100% of
  the negatives come from that one operator. It injected ~3.3e-3 of the standing
  number burden per 6 h step; zonal90d accumulated a cumulative floor equal to
  35% of its day-90 standing N, 97.4% of it below 10 nm. MASS/AOD/ARF were always
  safe; TOTAL NUMBER and anything under ~10 nm were not.

TO RESUME after a crash/kill: re-run with --resume 1. It picks up
coupled_state_<TAG>_ckpt.npz (validated bit-exact). Kill by PID, not by script
name. A RESUME across an ADV_VPOS change is not meaningful -- the checkpoint
carries a floor-contaminated number field.
============================================================================
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # src/
REPO = HERE.parent                              # the checkout
DRIVER = HERE / "driver_fast.py"

# ---------------------------------------------------------------------------
# The knob table. One row per environment variable the model reads.
#   (group, cli-flag, ENV_VAR, preset, metavar, help)
# preset is None for "do not set it -- let coupling.py's own default stand".
# A non-None preset is this launcher's PRODUCTION choice and is what makes a
# bare run_prod.py the production run rather than a bare coupling.py run.
#
# Every one of these was reachable before only by knowing the variable name;
# several (INJ_*, FAST_SORT, STATE_CKPT, FRAME_EVERY, N_HOURS) were bare
# literals in the shell version at various times, which meant an override was
# accepted by the shell and then SILENTLY DISCARDED. A table makes that class of
# bug structurally impossible: presets and pass-throughs go through one path.
# ---------------------------------------------------------------------------
KNOBS = [
 # ---- run control ----
 ("run", "--n-hours",      "N_HOURS",      "2160", "H",   "run length in hours (2160 = 90 d; 8760 = 1 yr)"),
 ("run", "--n-days",       "N_DAYS",       None,   "D",   "run length in days (N_HOURS wins if both given)"),
 ("run", "--out-tag",      "OUT_TAG",      "prod90d", "TAG", "output tag; ALL outputs and checkpoints are keyed by it"),
 ("run", "--resume",       "RESUME",       "0",    "0|1", "resume from coupled_state_<TAG>_ckpt.npz"),
 ("run", "--h0",           "H0",           None,   "N",   "start hour index into the CESM h1 series (0 = 1996-01-01 00Z)"),
 ("run", "--step-hours",   "STEP_HOURS",   None,   "H",   "coupling step [h]; one advect+coag per step (default 6)"),
 ("run", "--state-ckpt",   "STATE_CKPT",   "1",    "0|1", "write the 3-D restart checkpoint (needed for --resume)"),
 ("run", "--frame-every",  "FRAME_EVERY",  "24",   "H",   "spatial frame every N hours (>= N_HOURS = final frame only)"),
 ("run", "--frame3d",      "FRAME3D",      None,   "0|1", "1 = write the full 3-D state + per-cell size statistics + TOA fields after EVERY step to frames3d/step_<s>.npz (~480 MB/step, ~700 GB/yr)"),
 ("run", "--frame-levels", "FRAME_LEVELS", None,   "probe|all",
                                                          "'probe' (default) = one PROBE_HPA level; 'all' adds a "
                                                          "zonal-mean frame at every level (+1.1 GB/yr) and unlocks "
                                                          "the lat-height figures and gifs"),
 ("run", "--log-every",    "LOG_EVERY",    None,   "N",   "progress line every N coupling steps"),
 ("run", "--probe-hpa",    "PROBE_HPA",    None,   "hPa", "diagnostic/frame level (default 50)"),
 ("run", "--debug",        "DEBUG",        "1",    "0|1", "verbose per-step diagnostics"),
 ("run", "--profile",      "PROFILE",      "1",    "0|1", "per-stage timing lines"),
 # ---- domain / grid ----
 ("grid", "--p-lo-hpa",    "P_LO_HPA",     None,   "hPa", "top of the band (default 1)"),
 ("grid", "--p-hi-hpa",    "P_HI_HPA",     None,   "hPa", "bottom of the band (default 100)"),
 ("grid", "--n-lev",       "N_LEV",        None,   "N",   "sub-sample the band to ~N levels (0 = full contiguous band)"),
 ("grid", "--n-bins",      "N_BINS",       None,   "N",   "TOMAS bin count (0 = native 40)"),
 ("grid", "--diag-core-hpa", "DIAG_CORE_HPA", None, "LO,HI", "core window for the 'int' burden diagnostics"),
 # ---- injection scenario ----
 ("inj", "--inj-so2",      "INJ_SO2_TG_YR", "0.0", "Tg/yr", "SO2 injection rate. DEFAULT 0 = no-injection control"),
 ("inj", "--inj-h2so4",    "INJ_H2SO4_TG_YR", "0.0", "Tg/yr", "direct H2SO4(g) injection rate"),
 ("inj", "--inj-hpa",      "INJ_HPA",      "55.0", "hPa", "injection altitude (snapped to the nearest model level)"),
 ("inj", "--inj-lat",      "INJ_LAT",      "0.0",  "deg", "injection latitude (snapped to the nearest row)"),
 ("inj", "--inj-lon",      "INJ_LON",      "180.0", "degE", "injection longitude (ignored when --inj-zonal 1)"),
 ("inj", "--inj-zonal",    "INJ_ZONAL",    "1",    "0|1", "1 = spread around the whole latitude ring, 0 = single cell"),
 ("inj", "--inj-mirror",   "INJ_MIRROR",   "0",    "0|1", "release at BOTH +lat and -lat, total split 50/50 (not doubled)"),
 # ---- aerosol source / initial condition ----
 ("aer", "--aer-src",      "AER_SRC",      None,   "mam4|carma", "IC/BC reservoir source (default mam4)"),
 ("aer", "--init-bin",     "INIT_BIN",     "so4",  "so4|dgnum", "how MAM4 modes are binned onto TOMAS"),
 ("aer", "--init-sigma",   "INIT_SIGMA",   None,   "S",   "override the log-normal mode width used for binning"),
 ("aer", "--carma-file",   "CARMA_FILE",   None,   "PATH", "CARMA reservoir file (AER_SRC=carma)"),
 ("aer", "--carma-frame",  "CARMA_FRAME",  None,   "N",   "time index into the CARMA file"),
 ("aer", "--carma-rho",    "CARMA_RHO",    None,   "kg/m3", "CARMA particle density"),
 ("aer", "--carma-subbin", "CARMA_SUBBIN", None,   "N",   "CARMA sub-binning factor"),
 # ---- boundaries ----
 ("bc", "--n-bc-top",      "N_BC_TOP",     None,   "N",   "band levels pinned/served at the top face (default 1)"),
 ("bc", "--n-bc-bot",      "N_BC_BOT",     None,   "N",   "band levels pinned/served at the bottom face (default 1)"),
 ("bc", "--bc-edge",       "BC_EDGE",      None,   "open|clamp", "aerosol edge treatment (derived from ADV_WCONT)"),
 ("bc", "--bc-gas",        "BC_GAS",       None,   "flux|clamp", "gas edge treatment (derived from BC_EDGE)"),
 ("bc", "--bc-bot-aer",    "BC_BOT_AER",   None,   "X",   "scale the bottom-face aerosol inflow (0 = clean upwelling)"),
 # ---- physics stages ----
 ("phys", "--micro",       "MICRO",        None,   "full|coag|off", "microphysics mode; 'off' is transport-only"),
 ("phys", "--micro-substeps", "MICRO_SUBSTEPS", None, "N", "micro substeps per coupling step (default 6)"),
 ("phys", "--settle",      "SETTLE",       None,   "0|1", "gravitational settling (default on)"),
 ("phys", "--wet-settling", "WET_SETTLING", None,  "0|1", "hygroscopic growth in the settling velocity"),
 ("phys", "--wet-optics",  "WET_OPTICS",   None,   "0|1", "hygroscopic growth in the optics"),
 ("phys", "--rad",         "RAD",          None,   "0|1", "radiation on/off"),
 ("phys", "--rad-every",   "RAD_EVERY",    None,   "N",   "coupling steps between radiation calls"),
 ("phys", "--rad-mode",    "RAD_MODE",     None,   "anomaly|full|flux", "heating-rate definition (flux = one solve, no feedback)"),
 ("phys", "--arf-avg-h",   "ARF_AVG_H",    None,   "H",   "trailing window for the reported TOA forcing"),
 ("phys", "--alpha-cond",  "ALPHA_COND",   None,   "X",   "H2SO4 accommodation coefficient"),
 # ---- chemistry / nucleation ----
 ("chem", "--oh-sza",      "OH_SZA",       None,   "0|1", "diurnal OH parabola in cos(SZA)"),
 ("chem", "--oh-peak",     "OH_PEAK",      None,   "cm-3", "noon OH peak (default 2.3e6)"),
 ("chem", "--oh-substeps", "OH_SUBSTEPS",  None,   "N",   "OH samples per coupling step"),
 ("chem", "--nuc-org",     "NUC_ORG",      None,   "cm-3", "organic concentration for Riccobono nucleation"),
 ("chem", "--nuc-nh3",     "NUC_NH3",      None,   "cm-3", "ammonia concentration"),
 ("chem", "--nuc-fion",    "NUC_FION",     None,   "X",   "ion-induced nucleation factor"),
 ("chem", "--nuc-fn-max",  "NUC_FN_MAX",   None,   "cm-3 s-1", "cap on the total nucleation rate"),
 ("chem", "--n-coag-substeps", "N_COAG_SUBSTEPS", None, "N", "forward-Euler coag substeps (legacy MICRO=coag path)"),
 ("chem", "--coag-max-substeps", "COAG_MAX_SUBSTEPS", None, "N", "ceiling on the adaptive coag substeps (speed-critical)"),
 # ---- microphysics engine selection / emulator ----
 ("engine", "--micro-engine", "MICRO_ENGINE", None, "fast|emulator",
                                                          "which batched engine advances the run (default fast = tomas_jax.fast)"),
 ("engine", "--micro-ref",  "MICRO_REF",    None,   "fast|emulator",
                                                          "ALSO run this engine on the same input and log the per-step "
                                                          "comparison to micro_bench_<TAG>.npz (the run still advances "
                                                          "with --micro-engine)"),
 ("engine", "--micro-ref-every", "MICRO_REF_EVERY", None, "N", "compare every N coupling steps (default 1)"),
 ("engine", "--emu-ckpt",   "EMU_CKPT",     None,   "PATH", "emulator checkpoint (default: bintf_big_rebal avg_top3.pt)"),
 ("engine", "--emu-d",      "EMU_D",        None,   "N",   "emulator width (384 for the default ckpt, 256 for the light one)"),
 ("engine", "--emu-nl",     "EMU_NL",       None,   "N",   "emulator depth (6 for the default ckpt, 4 for the light one)"),
 ("engine", "--emu-bs",     "EMU_BS",       None,   "N",   "cells per emulator forward pass (fixed per compiled graph; default 16384)"),
 ("engine", "--emu-dtype",  "EMU_DTYPE",    None,   "fp16|fp32", "emulator autocast precision (default fp16; bf16 refused)"),
 ("engine", "--emu-compile", "EMU_COMPILE", None,   "0|1", "torch.compile(reduce-overhead) the emulator (default 1)"),
 ("engine", "--emu-oh-floor", "EMU_OH_FLOOR", None,  "cm-3", "OH floor handed to the emulator (default 100; the SZA curve is 0 at night, the training set never was)"),
 ("engine", "--emu-so2-floor", "EMU_SO2_FLOOR", None, "kg/box", "SO2 floor handed to the emulator (default 1e-17, the training minimum)"),
 ("engine", "--emu-project", "EMU_PROJECT",  None,   "clip|chem|sulfur|0", "output projection: chem = two-moment clip + exact SO2/H2SO4 chemistry and sulfur budget (default); clip = two-moment consistency only (the first-pass benchmark); sulfur = conservation from the network's gas (worse, see docs/EMULATOR.md)"),
 ("engine", "--emu-triage", "EMU_TRIAGE",   None,   "0|1", "1 = analytic step for quiet cells, network for the rest (default); 0 = network everywhere"),
 ("engine", "--emu-phys-top", "EMU_PHYS_TOP", None,  "F",   "hybrid: fraction of cells (top by sulfur per box) integrated by tomas_jax.fast instead of the network (default 0.01, ~0.2 s/step, removes the plume burst error; 0 = off)"),
 ("engine", "--emu-phys-chunks", "EMU_PHYS_CHUNKS", None, "N", "hybrid: stiffness-sorted cell chunks for the physics subset (default 1)"),
 ("engine", "--emu-phys-num", "EMU_PHYS_NUM", None, "F", "hybrid: also hand this fraction of the remaining cells, ranked by particle number per box (the dense populations whose coagulation the network under-does), to the fast physics (default 0.04, ~1 s/step; 0 = off)"),
 ("engine", "--micro-sample-every", "MICRO_SAMPLE_EVERY", None, "N", "write a training-data sample of the engine's input every N steps (0 = off)"),
 ("engine", "--micro-sample-n", "MICRO_SAMPLE_N", None, "N", "cells per sample (default 5000; plus the plume stratum, capped at n/2)"),
 ("engine", "--emu-code",   "EMU_CODE",     None,   "PATH", "tomas-emulator code tree (default /data/tomas-emulator)"),
 ("engine", "--emu-root",   "EMU_ROOT",     None,   "PATH", "emulator checkpoints/norm_stats root (default /data/tomas-jax-output/emulator_v1)"),
 # ---- fast microphysics engine ----
 ("fast", "--fast-dt",     "FAST_DT",      None,   "s",   "inner timestep of run_fast (default 360)"),
 ("fast", "--fast-cell-cap", "FAST_CELL_CAP", "50000", "N", "cells per run_fast chunk; smaller is often FASTER"),
 ("fast", "--fast-sort",   "FAST_SORT",    "1",    "0|1", "stiffness sort (~27%% of micro; one ~8 GB alloc -- first to disable on a loaded card)"),
 ("fast", "--fast-fn-scale", "FAST_FN_SCALE", None, "X",  "nucleation-rate scale factor"),
 ("fast", "--fast-coag-sub-cap", "FAST_COAG_SUB_CAP", None, "N", "coag substep cap inside run_fast"),
 ("fast", "--fast-cond-sub-cap", "FAST_COND_SUB_CAP", None, "N", "condensation substep cap inside run_fast"),
 ("fast", "--fast-coag-cmax", "FAST_COAG_CMAX", None, "X", "coag Courant-like limit inside run_fast"),
 # ---- meteorology source ----
 ("met", "--met-src",      "MET_SRC",      None,   "cesm|file", "where U/V/OMEGA/T come from: cesm (default) = the hourly h1 archive; file = the step-end frames in MET_DIR (an emulator free run regridded by scripts/met/emu_to_met.py). Gases, reservoirs and the radiation column below the band stay archive-driven"),
 ("met", "--met-dir",      "MET_DIR",      None,   "PATH", "MET_SRC=file: directory holding u/v/omega/T.npy + meta.json"),
 ("met", "--met-rh",       "MET_RH",       None,   "q|cesm", "MET_SRC=file: RH = archive water vapour with the file's temperature, over ice (q, default) or the archive's RELHUM as is (cesm)"),
 ("met", "--bc-src",       "BC_SRC",       None,   "cesm|clim", "where every non-wind boundary field comes from: cesm (default) = the hourly archive; clim = the monthly climatology in CLIM_FILE (scripts/met/make_bc_climatology.py), so the run opens no archive file. Needs --met-src file"),
 ("met", "--clim-file",    "CLIM_FILE",    None,   "PATH", "BC_SRC=clim: the bc_clim.npz climatology"),
 # ---- advection ----
 ("adv", "--adv-scheme",   "ADV_SCHEME",   None,   "lr|fast|sl|ffsl", "transport form (default ffsl = lr at large Courant number, advection/ffsl_lr.py, the production scheme since 2026-10-15; lr = Lin-Rood flux form at CFL sub-steps, the previous default, byte for byte; sl = semi-Lagrangian split, docs/TRANSPORT_SL_HANDOFF.md)"),
 ("adv", "--adv-ffsl-k",   "ADV_FFSL_K",   None,   "int", "ffsl: sub-steps per coupling step (default 24, the production operating point; 6 = hourly; a convergence knob, each ~1/200 of an lr step)"),
 ("adv", "--adv-ffsl-split", "ADV_FFSL_SPLIT", None, "strang|lie", "ffsl: sweep order (strang = z/2, xy|yx alternating, z/2, default; lie = fct_lr's x y z)"),
 ("adv", "--adv-sl-k",     "ADV_SL_K",     None,   "N",   "SL: sub-steps per coupling step (default 6 = one per hourly wind)"),
 ("adv", "--adv-sl-fixer", "ADV_SL_FIXER", None,   "global|local|none", "SL: per-tracer mass fixer on the closed column (default global; local = Bermejo-Conde)"),
 ("adv", "--adv-sl-hourly", "ADV_SL_HOURLY", None, "0|1", "sl and ffsl: 1 = read the hourly CESM winds between the step snapshots; 0 (default) = the two snapshots interpolated in time, as lr does and as production winds arrive"),
 ("adv", "--adv-sl-vert",  "ADV_SL_VERT",  None,   "flux|sl", "SL: vertical treatment (default flux = fct_lr's open-face flux sweep; sl = closed-column 3-D gather, A/B-failed)"),
 ("adv", "--adv-sl-omega", "ADV_SL_OMEGA", None,   "cont|cesm", "SL flux vertical: face omega from continuity (lr's, default) or CESM centers averaged"),
 ("adv", "--adv-sl-rho",   "ADV_SL_RHO",   None,   "0|1", "SL flux vertical, lie split only: carry the air mass 1 - dt div(v) through the horizontal step (strang always does; default 0)"),
 ("adv", "--adv-sl-split", "ADV_SL_SPLIT", None,   "lie|strang", "SL flux vertical: strang (default) = V/2 H V/2 with the air mass carried (second-order face flux); lie = H then V (first order at the open face)"),
 ("adv", "--adv-sl-jac",   "ADV_SL_JAC",   None,   "0|1", "SL flux vertical: 1 (default) = air mass from the Jacobian of the departure map and the bookkeeping remainder returned through the faces; 0 = Eulerian 1 - dt div(v) and the global column fixer"),
 ("adv", "--adv-sl-vorder", "ADV_SL_VORDER", None, "linear|cubic", "SL: vertical gather order (default linear)"),
 ("adv", "--adv-cfl",      "ADV_CFL",      None,   "X",   "horizontal CFL target (production 0.5)"),
 ("adv", "--adv-passive",  "ADV_PASSIVE",  None,   "0|1", "TEST: advect ONE total-sulfate tracer (+ the two gases) and rescale every bin by the cell's mass ratio (frozen per-cell shape); score against the same run with 0"),
 ("adv", "--adv-f32",      "ADV_F32",      None,   "0|1", "float32 transport sweeps (production 1)"),
 ("adv", "--adv-vpos",     "ADV_VPOS",     "1",    "0|1", "VERTICAL POSITIVITY LIMITER -- see the header; do not turn off"),
 ("adv", "--adv-wcont",    "ADV_WCONT",    None,   "0|1", "rederive omega from discrete continuity (opens the faces)"),
 ("adv", "--adv-metric",   "ADV_METRIC",   None,   "0|1", "cos(phi) area metric in the y-sweep"),
 ("adv", "--adv-dxfix",    "ADV_DXFIX",    None,   "0|1", "true grid spacing in the x-sweep"),
 ("adv", "--adv-polar",    "ADV_POLAR",    None,   "zonal|freeze", "polar cap treatment (zonal = stirred, mass-conserving)"),
 # ---- radiation gases / chunking ----
 ("rad", "--co2-ppm",      "CO2_PPM",      None,   "ppm", "background CO2 for RRTMGP"),
 ("rad", "--n2o-ppb",      "N2O_PPB",      None,   "ppb", "background N2O for RRTMGP"),
 ("rad", "--rad-lat-chunk", "RAD_LAT_CHUNK", None, "N",   "latitude rows per radiation chunk (GPU memory)"),
 # ---- radiation engine / reference cache / A-B (docs/RADIATION.md) ----
 ("rad", "--rad-engine",   "RAD_ENGINE",   None,   "rrtmgp|emulator", "which engine solves the EVOLVED radiation call (default rrtmgp = the physics; emulator = the column network, src/radiation/rad_emulator.py)"),
 ("rad", "--rad-ref",      "RAD_REF",      None,   "rrtmgp", "with --rad-engine emulator: ALSO run the physics on the same input and log the per-step comparison to rad_bench_<TAG>.npz (the run advances with the engine)"),
 ("rad", "--rad-ref-every", "RAD_REF_EVERY", None, "N",   "compare every N radiation steps (default 1)"),
 ("rad", "--rad-ref-cache", "RAD_REF_CACHE", None, "PATH|0", "per-archive-hour cache of the reference-aerosol solve and the clear-sky fields (default inputs/cache/rad_ref; bit-identical on a hit, computes+stores on a miss; 0 = off)"),
 ("rad", "--rad-sample-every", "RAD_SAMPLE_EVERY", None, "N", "write rad_sample_<TAG>_s<step>.npz (evolved bins, dT_rad, hour) every N radiation steps -- emulator training data (0 = off)"),
 ("rad", "--rad-emu-ckpt", "RAD_EMU_CKPT", None,   "PATH", "radiation emulator weights (.npz exported by scripts/radiation/train_rad_mlp.py)"),
 ("rad", "--rad-emu-anom", "RAD_EMU_ANOM", None,   "net|phys", "emulator anomaly: net = network on BOTH evolved and reference aerosol (errors cancel in the background; default) | phys = network evolved minus the exact cached reference"),
 ("rad", "--rad-emu-arf",  "RAD_EMU_ARF",  None,   "exact|emu", "with --rad-engine emulator: report the TOA forcing from the physics A/B samples (default when --rad-ref rrtmgp; use --rad-ref-every 3 so the 4-sample diurnal window still covers the four local times) or from the networks"),
 ("rad", "--rad-emu-clip", "RAD_EMU_CLIP", None,   "K/day",     "with --rad-engine emulator: sanity clamp on the networks' heating outputs (TOA deltas at 25x), NaN -> 0, counted and logged; a residual MLP can blow up on an input far outside its training set (default 20; 0 = off)"),
 # ---- memory / performance (numerically irrelevant) ----
 ("perf", "--cell-chunk",  "CELL_CHUNK",   None,   "N",   "cells per microphysics vmap batch"),
 ("perf", "--tracer-chunk", "TRACER_CHUNK", None,  "N",   "tracers per advection batch (0 = all 82 at once)"),
 # ---- input data locations ----
 ("path", "--cesm-dir",    "CESM_DIR",     None,   "PATH", "root of the CESM h1 tseries archive"),
 ("path", "--cesm-prefix", "CESM_PREFIX",  None,   "STR",  "CESM case-name prefix of the h1 files"),
 ("path", "--cesm-suf",    "CESM_SUF",     None,   "STR",  "date-range suffix of the h1 files"),
 ("path", "--tomas-jax-path", "TOMAS_JAX_PATH", None, "PATH", "tomas-jax checkout (default models/tomas-jax)"),
 ("path", "--rrtmgp-path", "RRTMGP_PATH",  None,   "PATH", "jax-rrtmgp checkout (default models/jax-rrtmgp)"),
 # ---- debug dumps ----
 ("dbg", "--dump-premicro", "DUMP_PREMICRO", None, "PATH", "dump the pre-microphysics state to this file"),
 ("dbg", "--dump-premicro-step", "DUMP_PREMICRO_STEP", None, "N", "step at which to take that dump"),
]

GROUPS = [
    ("run",  "run control"),
    ("grid", "domain and grid"),
    ("inj",  "injection scenario (the knobs meant to change run to run)"),
    ("aer",  "aerosol source / initial condition"),
    ("bc",   "boundaries"),
    ("phys", "physics stages"),
    ("chem", "chemistry, nucleation and coagulation"),
    ("engine", "microphysics engine selection and the emulator (docs/EMULATOR.md)"),
    ("fast", "fast microphysics engine"),
    ("adv",  "advection"),
    ("met",  "meteorology source (MET_SRC; scripts/met/emu_to_met.py writes the file)"),
    ("rad",  "radiation gases, chunking, engine selection and the emulator (docs/RADIATION.md)"),
    ("perf", "memory / performance (cannot change results)"),
    ("path", "input data locations"),
    ("dbg",  "debug dumps"),
]


def build_parser():
    p = argparse.ArgumentParser(
        prog="run_prod.py",
        description=("Launch a coupled TOMAS-JAX SAI run. Outputs are written to "
                     "the CURRENT DIRECTORY, so launch from a run directory."),
        epilog=("Every flag below maps to the environment variable named in its "
                "help text; an already-set variable is used when the flag is "
                "omitted. Presets marked [prod] are this launcher's production "
                "choices, not coupling.py's defaults."),
        # NOT ArgumentDefaultsHelpFormatter: every knob flag defaults to None,
        # meaning 'not given', and it would append '(default: None)' to all 83.
        # The real default is in the help text as the [ENV; prod preset] tag.
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", default=os.environ.get("GPU", "0"), metavar="N",
                   help="CUDA device to pin the run to (GPU / CUDA_VISIBLE_DEVICES)")
    p.add_argument("--cuda-driver-lib", default=None, metavar="PATH",
                   help="directory holding libcuda.so.1; empty string disables "
                        "the search (CUDA_DRIVER_LIB)")
    p.add_argument("--preallocate", default=None, metavar="true|false",
                   help="XLA_PYTHON_CLIENT_PREALLOCATE (default false: grow on "
                        "demand, so the run is a good neighbour on a shared box)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the resolved environment and exit without launching")
    p.add_argument("--allow-source-tree", action="store_true",
                   help="override the refusal to write into the checkout (you "
                        "almost certainly do not want this -- see the guard)")
    for key, title in GROUPS:
        g = p.add_argument_group(title)
        for grp, flag, env, preset, metavar, helptext in KNOBS:
            if grp != key:
                continue
            g.add_argument(flag, default=None, metavar=metavar,
                           help=f"{helptext}  [{env}"
                                + (f"; prod preset {preset}]" if preset is not None else "]"))
    return p


def resolve_env(args):
    """Build the child environment. See PRECEDENCE in the module docstring."""
    env = dict(os.environ)
    resolved = {}
    for _grp, flag, name, preset, _mv, _h in KNOBS:
        attr = flag.lstrip("-").replace("-", "_")
        val = getattr(args, attr)
        if val is None:                       # no flag given
            if name in env:                   # an exported variable wins over the preset
                resolved[name] = (env[name], "env")
                continue
            if preset is None:                # nothing to say: coupling.py decides
                continue
            val, src = preset, "preset"
        else:
            src = "flag"
        env[name] = str(val)
        resolved[name] = (str(val), src)
    return env, resolved


def find_libcuda(explicit):
    """Locate libcuda.so.1.

    JAX needs it, and under a containerized driver mount it is not on the loader
    path. Without it JAX SILENTLY falls back to CPU, which turns a 3.5-hour run
    into weeks -- so a miss is a loud warning, never a quiet default.
    """
    if explicit == "":                        # explicitly disabled
        return None, None
    cand = explicit or "/run/nvidia/driver/usr/lib/x86_64-linux-gnu"
    if (Path(cand) / "libcuda.so.1").exists():
        return cand, None
    for root in ("/run/nvidia", "/usr/lib"):
        r = Path(root)
        if not r.exists():
            continue
        for hit in r.rglob("libcuda.so.1"):
            return str(hit.parent), f"libcuda.so.1 found at {hit.parent}"
    return None, ("WARNING: libcuda.so.1 not found. If the run reports no GPU it "
                  "has fallen back to CPU -- set --cuda-driver-lib.")


def guard_cwd(allow):
    """Refuse to write into the source tree.

    Outputs go to $PWD and they do not belong in the checkout. This fails
    INVISIBLY otherwise: .npz/.png are gitignored, so a run launched from the
    wrong directory looks completely normal while quietly filling the tree (this
    is how 29 GB accumulated before 2026-08-12). The rule is "anywhere inside the
    checkout except under runs/", which is what makes the documented
    `cd runs/<tag>` workflow legal.
    """
    if allow:
        return
    pwd = Path.cwd().resolve()
    runs = REPO / "runs"
    inside = pwd == REPO or REPO in pwd.parents
    in_runs = pwd == runs or runs in pwd.parents
    if inside and not in_runs:
        sys.exit(
            f"run_prod.py: refusing to run from inside the source tree.\n"
            f"             $PWD = {pwd}\n"
            f"             Outputs are written to $PWD, and they do not belong in\n"
            f"             the source tree. Launch from a run directory instead:\n"
            f"                 mkdir -p {REPO}/runs/<tag> && cd $_\n"
            f"                 python3 {Path(__file__).resolve()} --out-tag <tag>\n"
            f"             (a directory outside the checkout entirely also works)")


def main():
    args = build_parser().parse_args()
    guard_cwd(args.allow_source_tree)
    env, resolved = resolve_env(args)

    libdir, note = find_libcuda(args.cuda_driver_lib
                                if args.cuda_driver_lib is not None
                                else os.environ.get("CUDA_DRIVER_LIB"))
    if note:
        print(f"run_prod.py: {note}", file=sys.stderr)
    if libdir:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [libdir] + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = (
        args.preallocate if args.preallocate is not None
        else os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "false"))

    print(f"run_prod.py: CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}  "
          f"preallocate={env['XLA_PYTHON_CLIENT_PREALLOCATE']}", file=sys.stderr)
    width = max((len(k) for k in resolved), default=0)
    for k in sorted(resolved):
        val, src = resolved[k]
        print(f"run_prod.py:   {k:<{width}} = {val}   ({src})", file=sys.stderr)

    if args.dry_run:
        print("run_prod.py: --dry-run, not launching", file=sys.stderr)
        return 0
    if not DRIVER.exists():
        sys.exit(f"run_prod.py: driver not found at {DRIVER}")
    # exec, not import: see WHY IT STILL EXECS in the module docstring.
    os.execve(sys.executable, [sys.executable, str(DRIVER)], env)


if __name__ == "__main__":
    sys.exit(main())
