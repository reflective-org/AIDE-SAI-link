#!/usr/bin/env bash
# ============================================================================
# PRODUCTION 90-DAY RUN
# ============================================================================
# 360 steps x 6 h = 2160 h = 90 days, ~33-36 h wall clock on one H100 at the
# observed ~340-370 s/step (microphysics is ~94% of it).
#
# CONFIG = coupling.py / driver_fast.py DEFAULTS except the knobs below.
# For the record, what the defaults now give (validated by an 18 h A/B pair):
#   AER_SRC=mam4        per-step dynamic IC + BC. No CARMA anywhere -- only ~1
#                       week of CARMA output exists, so it could only ever be a
#                       STATIC reservoir, and a frozen reservoir is the root
#                       cause of the old mass leak.
#   INIT_BIN=so4        NOW THE DEFAULT (coupling.py:381, flipped 2026-07-29).
#                       Set explicitly here anyway so the log records it: the old
#                       dgnum default binned MAM4 NUMBER by dgnumwet and set
#                       mas = num*MMID without ever reading so4_a*, inflating
#                       sulfate mass 4.29x (mode 3 alone 6.68x) and pushing
#                       Dp(massw) to ~1920 nm instead of ~890 nm.
#   P_LO/HI = 1/150 hPa 24 levels, 1,327,104 cells. The 1 hPa top is effectively
#                       aerosol-free inflow (MAM4 3.2e-17 kg/kg, 7 orders below
#                       13 hPa) so no knob is needed there.
#   OH                  diurnal parabola in cos(SZA), OH_PEAK=2.3e6, 60 samples
#                       per step (6 min). NOT CESM's OH -- a parametrization
#                       sitting in front of the whole SO2 -> H2SO4 -> aerosol chain.
#   BC_BOT_AER=1.0      bottom-face aerosol inflow at full MAM4 strength.
#   ADV_WCONT=1 -> BC_EDGE=open   both vertical faces are real FLUX boundaries
#                       (inflow served at the reservoir value, free outflow), not
#                       Dirichlet clamps. This is what removed the leaking clamp:
#                       the budget's `bc` term is now +0.00e+00 exactly.
#   -> BC_GAS=flux      NEW DEFAULT 2026-07-30, derived from BC_EDGE so the gases
#                       cannot desync from the aerosol. SO2/H2SO4 now get the same
#                       open faces instead of a Dirichlet clamp. The old clamp beside
#                       open aerosol faces was an unbounded gas SOURCE at a level
#                       whose particles were free to leave: the 13.3 hPa top level
#                       went 0.3% -> ~50% of the model's TOTAL number in 24 h as a
#                       6-8 nm mode. Cost: the gases are no longer pinned to CESM.
#                       This run is therefore NOT comparable to the 2026-07-29
#                       prod90d, which is mid-trajectory on BC_GAS=clamp.
#   FRAME_EVERY=24      probe-level frame + full 3-D state checkpoint every 24 h.
#                       OVERRIDABLE as of 2026-08-14, and it has to be for any run
#                       much longer than 90 days: the whole frames history is
#                       REWRITTEN at every frame, so cost grows as (frames)^2. A
#                       2-year run at 24 h would write ~40 GB of frames and spend
#                       hours re-serialising them; FRAME_EVERY=120 makes the same
#                       run 147 frames and ~8 GB.
#   STATE_CKPT=1        set explicitly below; enables RESUME=1 restart.
#   MICRO               tomas_jax.fast.run_fast, 60 inner steps x 360 s. ~20x
#                       faster than physical TOMAS and stable, but its nucleation
#                       is BINARY, not ternary.
#   RADIATION           physical RRTMGP + Mie, RAD_MODE=anomaly. Under AER_SRC=mam4
#                       the anomaly BASELINE is time-varying, so the reported
#                       forcing is defined differently than in zonal90d and is not
#                       directly comparable to those numbers.
#
# HOW TO READ THE OUTPUT -- three standing caveats:
#   * the first ~week is cold-start spin-up (N/N0 was still climbing at 19.5 by
#     18 h and had not plateaued);
#   * 90 days is a TRANSIENT, not a steady state -- the sink is not
#     burden-proportional (loss ~ M^0.35), so no plateau is readable from it;
#   * there is NO WET REMOVAL anywhere. Settling and transport out of the band are
#     the only sinks, so aerosol settling into the 100-150 hPa layer just lingers.
#   * AOD550 came out 0.0039 in the 18 h test vs a ~0.005-0.01 quoted background.
#     Uncalibrated -- fine for the run, matters before quoting absolute forcing.
#
# MEMORY / SHARED-GPU SETTINGS (none can change results -- micro is per-cell
# independent, so chunk grouping is numerically irrelevant):
#   FAST_SORT=1         stiffness sort, worth ~27% of micro (~5 h over a 90-day run).
#                       It does one unchunked ~8 GB allocation, so it is the first
#                       thing to disable (FAST_SORT=0) if the card is already loaded
#                       -- it OOMs when another job holds most of the memory.
#   GPU=0               ONE card, pinned, rather than auto-selected: the card a
#                       33-hour job lands on should never be a function of what
#                       happened to be idle at launch. GPU=<n> overrides.
#   FAST_CELL_CAP=50000 lower than the 250000 default, and FASTER, not slower:
#                       micro went only 1.3x for 2.18x the cells, because the
#                       vmapped adaptive coag while_loop runs every lane to the
#                       slowest one, so 27 small chunks waste far less than 3 big
#                       ones. It also fits alongside another job on the card.
#   INJ_ZONAL=1         zonal ring, 10 Tg SO2/yr spread over all 288 longitudes at
#                       lat -0.5 / 51.7 hPa. The default POINT source is 5.6x
#                       slower and drives runaway nucleation.
#
# ADV_VPOS=1 -- VERTICAL POSITIVITY LIMITER, new 2026-07-29. Now the DEFAULT in
#   fct_lr.py; still set explicitly here so the log records it. Without it the
#   number field is not usable:
#   fct_lr's vertical remap is exactly CONSERVATIVE but not POSITIVE (the
#   Colella-Woodward limiter in _ppm_coeffs_nonper enforces monotonicity, which is
#   a different property), so it undershoots negative on the steep ultrafine
#   gradient at the injection ring and coupling.py's floor clips those negatives
#   and CREATES number. Measured: 100% of the negatives come from this one
#   operator -- the horizontal sweeps produce exactly zero because _lr_sweep's
#   Zalesak step is already positivity-preserving.
#   Scale of the problem it fixes (measured on real states):
#     * floor injected ~3.3e-3 of the standing number burden per 6h step;
#       zonal90d accumulated a cumulative floor equal to 35% of its day-90
#       standing N.
#     * 97.4% of it landed in bins below 10 nm (number-weighted mean Dp 3.3 nm)
#       and 0.0025% in the optically active 150-1200 nm bins, and that split does
#       NOT drift with run age. So MASS/AOD/ARF were always safe; TOTAL NUMBER
#       and anything under ~10 nm were not.
#   Validated: worst undershoot -8.5e10 -> roundoff, floor contribution to the
#   number budget -> exactly 0, conservation residual unchanged to 4 s.f.
#   (limiting only rescales shared face fluxes, so telescoping is preserved), and
#   inactive to roundoff on smooth fields so the LR/N2O accuracy result stands.
#   Over 8 steps of pure advection the unlimited scheme GREW the number burden
#   0.94% out of nothing; limited it sits at 0.99987.
#   NB it also fixes the MASS floor (2.4e6 negative cells in the mass field).
#
# NB every example below invokes the launcher as `$REPO/run_prod.sh`, where $REPO
# is wherever this checkout lives. That is not decoration: outputs are written to
# the CURRENT DIRECTORY and this script REFUSES to run with the repo as $PWD, so
# every real invocation is from a runs directory by path. See the block above
# `HERE=` further down for why.
#
# TO RESUME after a crash/kill: re-run this script with RESUME=1 prepended. It
# picks up coupled_state_prod90d_ckpt.npz (validated bit-exact). Kill it by PID,
# not by script name. NOTE a RESUME across an ADV_VPOS change is not meaningful --
# the checkpoint carries a floor-contaminated number field.
# ============================================================================
set -euo pipefail
# Run THE TREE THIS SCRIPT LIVES IN, derived from the script's own location rather
# than hardcoded. A launcher that cd's to a fixed path runs whatever code is at that
# path, not the code you just edited -- and no amount of output checking detects it,
# because both sides of any comparison execute the same files.
#
# Resolve that tree, but do NOT cd into it. Until 2026-08-12 this line was a `cd`,
# which put every output beside the source; the repo accumulated 29 GB of .npz that
# .gitignore hid rather than prevented. Outputs go to the CURRENT DIRECTORY instead
# -- every output path in coupling.py and plot_run.py is cwd-relative -- so launch
# from a runs directory outside the repo:
#   cd <runs dir> && INJ_SO2_TG_YR=10 OUT_TAG=prod90d /path/to/repo/run_prod.sh
# Nothing on the import path needs the cwd: coupling.py resolves tomas-jax and
# jax-rrtmgp from its own __file__ (_dep_path/_HERE), radiation.py resolves
# rad_data/ the same way, and python3 puts the script's own directory on sys.path.
#
# TRADEOFF, stated plainly: the old `cd` also meant two checkouts could not collide,
# because each wrote into itself. Now they collide if both are launched from the same
# runs directory with the same OUT_TAG. OUT_TAG discipline (see above) is what keeps
# runs apart -- it already had to, since one checkout can overwrite its own results.
HERE="$(dirname "$(readlink -f "$0")")"

# Refuse to write into the repo. This is the whole point of the change above, and
# it fails INVISIBLY otherwise: .npz/.png are gitignored, so a run launched from the
# wrong directory looks completely normal and just quietly re-pollutes the tree.
if [[ "$(readlink -f "$PWD")" == "$HERE" ]]; then
    echo "run_prod.sh: refusing to run with the repo itself as the working directory." >&2
    echo "             Outputs are written to \$PWD, and they do not belong in the" >&2
    echo "             source tree. Launch from a runs directory instead:" >&2
    echo "                 cd <a directory outside the repo>" >&2
    echo "                 $(readlink -f "$0")" >&2
    exit 2
fi

# N_HOURS and OUT_TAG are overridable so this same launcher can EXTEND the run
# instead of being copied into a near-duplicate script that then drifts. They were
# bare literals until 2026-07-31, which meant an outside `N_HOURS=8760` was
# accepted by the shell and then silently discarded -- the run would have quietly
# stopped at 90 days. Defaults are unchanged, so a bare run_prod.sh is the same
# run it has always been.
# To push the 90-day run out to a full year:
#   RESUME=1 N_HOURS=8760 OUT_TAG=prod1yr $REPO/run_prod.sh
# after seeding the prod1yr checkpoints from the prod90d ones.
#
# FAST_SORT is overridable for the same reason: an OOM fallback that relaunches with
# FAST_SORT=0 is useless if the launcher swallows the override and goes straight back
# into the same OOM.
# INJECTION SCENARIO -- the knobs meant to change run to run. Every default below is
# coupling.py's own default, so a bare run_prod.sh is byte-for-byte the run it
# always was (verified: step-1 prognostics bit-identical to the reference run).
#
# INJ_ZONAL was a bare literal until 2026-08-03, so `INJ_ZONAL=0 run_prod.sh` was
# accepted by the shell and then silently discarded -- you would have got a zonal
# ring anyway and no warning. Same failure mode that already bit N_HOURS and
# FAST_SORT. The other four were never listed here at all; they did pass through by
# inheritance, but nothing told you they existed or what they defaulted to.
#
#   INJ_SO2_TG_YR  Tg SO2/yr, continuous. DEFAULT 0 = OFF (no-injection control).
#                  Was 10 until 2026-08-03; a forgotten flag now gives an obviously
#                  unforced baseline instead of a silent standard SAI scenario.
#   INJ_HPA        target altitude [hPa]          (snapped to the nearest model level)
#   INJ_LAT        target latitude [deg]          (snapped to the nearest row)
#   INJ_LON        target longitude [deg east]    (ignored when INJ_ZONAL=1)
#   INJ_ZONAL      1 = spread around the whole latitude ring, 0 = single cell
#   INJ_MIRROR     1 = release at BOTH +INJ_LAT and -INJ_LAT, total split 50/50
#                  (INJ_SO2_TG_YR=10 INJ_LAT=45 INJ_MIRROR=1 -> 5 Tg/yr at each of
#                  45N and 45S; the TOTAL is 10, not doubled). No-op at INJ_LAT=0.
#
# REPRODUCING the prod90d / prod1yr runs now needs the amount stated explicitly:
#   INJ_SO2_TG_YR=10 OUT_TAG=prod90d GPU=0 $REPO/run_prod.sh
#
# ALWAYS pair a scenario with its own OUT_TAG -- outputs and checkpoints are keyed
# by it, so reusing a tag overwrites the other scenario's results. coupling.py
# refuses a RESUME onto a checkpoint whose injection config differs, which catches
# the dangerous half of that mistake but not a fresh-run overwrite.
#   OUT_TAG=inj20_30N INJ_SO2_TG_YR=20 INJ_LAT=30 GPU=0 $REPO/run_prod.sh
#   OUT_TAG=inj5_eq_pt INJ_SO2_TG_YR=5 INJ_ZONAL=0 INJ_LON=120 GPU=0 $REPO/run_prod.sh
# The resolved geometry is echoed in the run header ("SAI injection: ... at ...").
# ============================================================================
# DOMAIN -- the production band, 1-150 hPa (24 levels), set HERE as of 2026-08-13
# ============================================================================
# These two were previously set only by driver_fast.py's own os.environ.setdefault,
# i.e. the production domain was a property of ONE ENTRY POINT rather than of the
# production environment. That was invisible until this launcher gained DRIVER:
# `DRIVER=coupling.py` silently got coupling.py's module defaults instead (1-100
# hPa, 21 levels), so two runs launched by the same script with the same knobs
# integrated DIFFERENT ATMOSPHERES -- and since where the floor sits governs where
# aerosol drains out, that is the last thing an inter-model drainage comparison
# can afford to have drift between drivers.
# Values are byte-identical to driver_fast.py's, whose comment block holds the
# rationale for each edge (1 hPa top = effectively aerosol-free inflow;
# 150 hPa bottom = right LATITUDE structure at the base, unlike the old 100 hPa).
# Its setdefault calls are now no-ops under this launcher and remain the fallback
# for invoking driver_fast.py directly.
# ============================================================================
# DRIVER -- which entry point this launcher execs (default driver_fast.py)
# ============================================================================
# One launcher, not two. Everything below this line -- the GPU pin, the libcuda
# path, the memory policy, the repo-as-$PWD guard -- is the production
# ENVIRONMENT, and it is identical whichever engine runs inside it. Copying this
# file to get a second entry point would fork all of that, and the copy would
# drift; adding a second launcher script is also exactly the debt CLAUDE.md says
# to stop accruing.
#
#   DRIVER=driver_fast.py  (default)  the batched tomas_jax.fast micro engine.
#                                     Requires MICRO=full -- it exits otherwise.
#   DRIVER=coupling.py                the standalone path: same advection, same
#                                     settling, same radiation, per-cell micro.
#                                     This is the one to use for MICRO=off, since
#                                     driver_fast.py's whole job is a micro engine
#                                     that MICRO=off does not call.
#
# ADVECTION-ONLY INTER-MODEL COMPARISON (2026-08-13). Pure transport + settling of
# a prescribed uniform PSD -- no microphysics, no radiation, no injection -- so
# that re-running the identical code against another model's winds isolates
# inter-model spread in transport. The band drains and nothing refills it:
#
#   cd <runs dir>
#   DRIVER=coupling.py MICRO=off RAD=0 AER_SRC=fixed \
#     BC_TOP_AER=0 BC_BOT_AER=0 \
#     OUT_TAG=mip_cesm $REPO/run_prod.sh
#
# The band is the production 1-150 hPa either way (see the DOMAIN block above) and
# N_HOURS defaults to 2160 = 90 days, so neither needs restating. ~5 s/step without
# microphysics or radiation, so a 90-day run is ~30 min rather than ~33 h.
#
# The run header echoes "TRANSPORT-ONLY run" only when all of that is actually in
# force, and names whatever is still active otherwise. Add FIXED_PSD=flat for the
# drainage-vs-size curve (equal number in all 40 bins, which under MICRO=off are
# independent tracers), or FIXED_P_HI_HPA=30 to start with a vertical gradient the
# winds act on immediately rather than one settling has to create first.
# ============================================================================
# ENVIRONMENT -- inlined 2026-08-04
# ============================================================================
# This block used to be `PYTHONPATH=<absolute paths> ../<launcher outside the repo>`:
# a launcher OUTSIDE this repo plus two absolute paths, so a clone of this repo
# on any other machine could not start a run at all. Inlining it also removes the
# auto-pick step: the card is pinned here, so which GPU a 33-hour job lands on is
# never a function of what happened to be idle at launch.
#
# 1. libcuda.so.1. JAX needs it, and where it lives under a containerized
#    driver mount, which is not on the loader path. Without it JAX SILENTLY falls
#    back to CPU, which turns a 33-hour run into months. Set CUDA_DRIVER_LIB to
#    relocate it, or CUDA_DRIVER_LIB= (empty) to skip this for a normal install.
CUDA_DRIVER_LIB=${CUDA_DRIVER_LIB-/run/nvidia/driver/usr/lib/x86_64-linux-gnu}
if [[ -n "$CUDA_DRIVER_LIB" ]]; then
    if [[ -e "$CUDA_DRIVER_LIB/libcuda.so.1" ]]; then
        export LD_LIBRARY_PATH="$CUDA_DRIVER_LIB:${LD_LIBRARY_PATH:-}"
    else
        _found=$(find /run/nvidia /usr/lib -name 'libcuda.so.1' 2>/dev/null | head -1 || true)
        if [[ -n "$_found" ]]; then
            export LD_LIBRARY_PATH="$(dirname "$_found"):${LD_LIBRARY_PATH:-}"
            echo "run_prod.sh: libcuda.so.1 found at $(dirname "$_found")" >&2
        else
            echo "run_prod.sh: WARNING: libcuda.so.1 not found. If the run reports no" >&2
            echo "             GPU it has fallen back to CPU -- set CUDA_DRIVER_LIB." >&2
        fi
    fi
fi
# 2. ONE card, pinned. GPU 0 is this project's only allocation.
export CUDA_VISIBLE_DEVICES=${GPU:-0}
# 3. Grow GPU memory on demand rather than preallocating 75% of the card, so the
#    run is a good neighbour on a shared box.
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
# 4. The sibling dependency repos are NOT set here. coupling.py resolves
#    tomas-jax and jax-rrtmgp itself (TOMAS_JAX_PATH / RRTMGP_PATH, else
#    else ../<name>), so both a clone beside them and an installed copy work with
#    nothing exported. Export those two variables before calling this script if the
#    repos live somewhere else.
echo "run_prod.sh: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  preallocate=$XLA_PYTHON_CLIENT_PREALLOCATE" >&2

N_HOURS=${N_HOURS:-2160} \
P_LO_HPA=${P_LO_HPA:-1.0} \
P_HI_HPA=${P_HI_HPA:-150.0} \
INJ_SO2_TG_YR=${INJ_SO2_TG_YR:-0.0} \
INJ_HPA=${INJ_HPA:-55.0} \
INJ_LAT=${INJ_LAT:-0.0} \
INJ_LON=${INJ_LON:-180.0} \
INJ_ZONAL=${INJ_ZONAL:-1} \
INJ_MIRROR=${INJ_MIRROR:-0} \
INIT_BIN=so4 \
STATE_CKPT=1 \
FRAME_EVERY=${FRAME_EVERY:-24} \
FAST_SORT=${FAST_SORT:-1} \
FAST_CELL_CAP=50000 \
ADV_VPOS=1 \
DEBUG=1 PROFILE=1 \
OUT_TAG=${OUT_TAG:-prod90d} \
RESUME=${RESUME:-0} \
    exec python3 "$HERE/${DRIVER:-driver_fast.py}"
