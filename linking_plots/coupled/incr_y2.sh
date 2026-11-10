#!/bin/bash
# Incremental year-2 processing WHILE the y2 runs are still going (2026-11-09). Every stage skips days already cached,
# so this only front-loads work that run_coupled_2yr.sh would otherwise do at the end. Lowest priority, few workers.
# D = number of complete days available in ALL six runs (frames 4d..4d+3 present for every d < D).
# rad_diurnal runs on GPU 7 only once nothing else is on it (MAM4 runs and the age-of-air runs finished).
export LP_MET=/data/AIDE-SAI-link/runs/emu_winds_P2s200_1997_1998/met LP_COUPLED=/data/AIDE-SAI-link/linking_plots/coupled
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mpl_ubuntu
export LD_LIBRARY_PATH=/run/nvidia/driver/usr/lib/x86_64-linux-gnu
PY=/data/aide_emu_venv/bin/python; Y2=/data/AIDE-SAI-link/runs/emu_campaign_1997_y2; C=$LP_COUPLED/cache
cd $LP_COUPLED/scripts
while true; do
  D=730
  for r in r1_tomas_emu r1c_tomas_emu_ctl r2_tomas_fast r2c_tomas_fast_ctl r3_mam4 r3c_mam4_ctl; do
    m=$(ls $Y2/$r/frames3d/ 2>/dev/null | grep -oE '^step_[0-9]+\.npz$' | sort | tail -1 | grep -oE '[0-9]+'); m=$((10#${m:-0}))
    d=$(( (m + 1) / 4 )); [ $d -lt $D ] && D=$d
  done
  echo "$(date +%H:%M) complete days in all runs: $D"
  if [ $D -gt 365 ]; then
    LP_NDAYS=730 NW=4 nice -n 19 $PY process.py 365 $D >> $C/process_y2_incr.log 2>&1
    LP_NDAYS=730 PAIRS_SEL=mam PFX=mam_ NW=4 nice -n 19 $PY process.py 365 $D >> $C/process_mam_y2_incr.log 2>&1
    LP_NDAYS=$D NW=4 nice -n 19 $PY wet_dist.py >> $C/wetdist_y2_incr.log 2>&1
    if ! nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 7 | grep -q .; then
      LP_NDAYS=$D CUDA_VISIBLE_DEVICES=7 nice -n 19 $PY rad_diurnal.py 0 1 >> $C/diurnal_y2_incr.log 2>&1
    fi
    echo "$(date +%H:%M) processed through day $((D-1))"
  fi
  [ $D -ge 730 ] && break
  sleep 1200
done
echo done
