#!/bin/bash
# The TAGGED-PULSE Brewer-Dobson experiment: seed only the tropical lower
# stratosphere and watch where the blob goes.
#
#   ./run_pulse_bdc.sh [YEARS]        # default 15, run from a directory OUTSIDE the repo
#
# NEEDS: tomas-jax on branch gpu-fast (imported for the bin grid even at MICRO=off),
# and CESM h1 hourly files for U V OMEGA T num_a1..3 so4_a1..3 SO2 H2SO4 OH RELHUM
# under $CESM_DIR -- all 14 are opened at startup even though a transport-only run
# reads few of them. jax-rrtmgp is NOT needed here: RAD=0 drops it.
# Try YEARS=1 first (~50 min) to prove the paths before committing to 15.
#
# Produces coupled_{frames,timeseries,state,final}_pulse_${YEARS}yr.npz, then:
#
#   python3 <repo>/pulse_progress_abs.py pulse_15yr     # one lat-p panel per year
#   python3 <repo>/pulse_deep_branch.py  pulse_15yr     # the ascent/descent Hovmoller
#   python3 <repo>/pulse_deep_branch.py  pulse_15yr --log
#   python3 <repo>/gif_run.py pulse_15yr --log --massdens --decades 5 --stride 4
#
# WHY A PULSE. A uniform initial condition can only show the circulation as the
# ABSENCE of tracer arriving (drainage age), which needs the clean-air front to
# traverse the whole circuit before anything is visible -- 4-6 years. A pulse
# shows the ascent directly, from the first month, as the thing that MOVES. It is
# the model analogue of the water-vapour tape recorder.
#
# EVERY SETTING BELOW IS A DECISION, NOT A DEFAULT:
#
#   FIXED_LAT_MAX_DEG=15   |lat| <= 15, with FIXED_P_LO/HI_HPA=40/90, is the seed
#   FIXED_P_LO_HPA=40      window: the tropical lower stratosphere, 18-24 km. It
#   FIXED_P_HI_HPA=90      lands on 5 model levels (43.2-87.8 hPa) and carries
#                          0.036 Tg of SO4 at 4.77e-10 kg/kg.
#   SETTLE=0               the point is the CIRCULATION. With settling on, the
#                          0.05 hPa model top sends a second clean front DOWNWARD
#                          which accounts for 74-86% of the depletion above
#                          3 hPa and would contaminate the polar leg.
#   N_BINS=1               with SETTLE=0 and MICRO=off the size bins are
#                          INDEPENDENT and IDENTICAL passive tracers -- same
#                          winds, no fall speed, no coagulation -- so 40 of them
#                          is 40 copies of one answer. Do NOT reuse this with
#                          SETTLE=1: it collapses the fall-speed spectrum the
#                          drainage runs depend on. Side effect: with one bin
#                          spanning the whole mass grid there is no bin-bound
#                          tying num to mas, so Dp(M/N) in the log is meaningless
#                          -- analyse frames_zm_mas and ignore the number moment.
#   BC_TOP_AER=0           both faces stay aerosol-free, so nothing is
#   BC_BOT_AER=0           re-injected and the pulse is a closed budget minus
#                          what leaves. NOTE the bottom face is at 143 hPa, which
#                          is 1-3 km ABOVE the real extratropical tropopause, so
#                          tracer is removed somewhat early at mid and high
#                          latitudes; treat the decay rate as a lower bound on
#                          the true stratospheric lifetime.
#   RAD=0, MICRO=off       transport only. dT_rad is then identically zero and
#                          the plotting tools skip those panels by themselves.
#   FRAME_EVERY=120        5-day frames. The whole frames history is rewritten at
#                          every frame, so cost grows as (frames)^2 -- at 24 h a
#                          15-year run would be unusable.
#
# COST. ~2950 s per simulated year on one H100, so ~12 h for 15 years, and 3.8 GB
# of frames (measured on the 15-year run). Checkpoints are continuous, so stopping
# early leaves a usable file.
set -euo pipefail

YEARS=${1:-15}
TAG=pulse_${YEARS}yr
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$PWD" = "$REPO" ]; then
    echo "run from a directory OUTSIDE the repo -- outputs land in \$PWD" >&2
    exit 1
fi

exec env \
    DRIVER=coupling.py MICRO=off RAD=0 AER_SRC=fixed \
    BC_TOP_AER=0 BC_BOT_AER=0 P_LO_HPA=0.03 \
    SETTLE=0 N_BINS=1 \
    FIXED_LAT_MAX_DEG=15 FIXED_P_LO_HPA=40 FIXED_P_HI_HPA=90 \
    N_HOURS=$((YEARS * 8760)) FRAME_EVERY=120 OUT_TAG="$TAG" \
    "$REPO/run_prod.sh"
