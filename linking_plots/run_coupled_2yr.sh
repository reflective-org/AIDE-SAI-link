#!/bin/bash
# Rebuild the COUPLED figures from two years of runs (after every y2 coupled run shows "Complete in").
# Run as ubuntu. Year-1 per-day caches are reused; only days 365..729 are computed.
# usage: GPUS="0 1 2 3 4 5 6 7" NW=24 bash run_coupled_2yr.sh       (each stage skips days already done, so it can be re-run)
set -eu
export LP_NDAYS=${LP_NDAYS:-730}
export LP_MET=${LP_MET:-/data/AIDE-SAI-link/runs/emu_winds_P2s200_1997_1998/met}
export LP_COUPLED=${LP_COUPLED:-/data/AIDE-SAI-link/linking_plots/coupled}
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mpl_$(whoami)
[ -d /run/nvidia/driver/usr/lib/x86_64-linux-gnu ] && export LD_LIBRARY_PATH=/run/nvidia/driver/usr/lib/x86_64-linux-gnu
PY=/data/aide_emu_venv/bin/python               # numpy/jax: processing + radiation
PYC=/data/strat-dycore-compare/.venv/bin/python # cartopy: map figures
GPUS=${GPUS:-0 1 2 3 4 5 6 7}; NW=${NW:-24}
C=$LP_COUPLED/cache; cd $LP_COUPLED/scripts
echo "[1/5] per-day reductions, days 365-$((LP_NDAYS-1)) (CPU)"; date
NW=$NW $PY process.py 365 $LP_NDAYS > $C/process_y2.log 2>&1
PAIRS_SEL=mam PFX=mam_ NW=$NW $PY process.py 365 $LP_NDAYS > $C/process_mam_y2.log 2>&1
echo "[2/5] diurnal SW re-solves on GPUs: $GPUS"; date
n=$(echo $GPUS | wc -w); i=0
for g in $GPUS; do CUDA_VISIBLE_DEVICES=$g $PY rad_diurnal.py $i $n > $C/diurnal_y2_$i.log 2>&1 & i=$((i+1)); done; wait
echo "[3/5] wet size distribution (CPU)"; date
NW=$NW $PY wet_dist.py > $C/wetdist_y2.log 2>&1
echo "[4/5] aggregate"; date
$PY agg2.py > $C/agg2_y2.log 2>&1
echo "[5/5] figures"; date
$PYC plots_v2.py; $PYC plots_extra.py; $PY plot_wet_dist.py; $PY ../extra/heating_figs.py   # plots_extra.py added 2026-11-09 (fig0 double axis, fig5_b, fig6)
echo "done: figures in $LP_COUPLED/figs_v2 and $LP_COUPLED/extra"; date
