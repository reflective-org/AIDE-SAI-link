#!/bin/bash
# Wait for all six y2 coupled runs to finish, then do the 2-year coupled rebuild (2026-11-09 overnight).
Y2=/data/AIDE-SAI-link/runs/emu_campaign_1997_y2; LP=/data/AIDE-SAI-link/linking_plots
R="r1_tomas_emu r1c_tomas_emu_ctl r2_tomas_fast r2c_tomas_fast_ctl r3_mam4 r3c_mam4_ctl"
while true; do
  n=0; for r in $R; do grep -q "Complete in" $Y2/$r/run.log 2>/dev/null && n=$((n+1)); done
  [ $n -eq 6 ] && break
  sleep 120
done
echo "$(date) all six complete"; for r in $R; do echo "$r $(grep 'Complete in' $Y2/$r/run.log | tail -1) frames3d $(ls $Y2/$r/frames3d | grep -c '^step_')"; done
pkill -f incr_y2.sh; sleep 5
while pgrep -f "process.py|wet_dist.py|rad_diurnal.py" >/dev/null; do sleep 10; done    # let an in-flight incremental stage finish
cd $LP && GPUS="0 1 2 3 4 5 6 7" NW=24 bash run_coupled_2yr.sh > coupled_2yr.log 2>&1
echo "$(date) run_coupled_2yr.sh exit $?"; tail -3 coupled_2yr.log
echo FINISH_2YR_DONE
