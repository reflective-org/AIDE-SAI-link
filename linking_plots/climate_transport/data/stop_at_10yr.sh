#!/bin/bash
# aoa_transport_2yr.py's outer loop counts 2-year CYCLES (labelled "year"), so --years 10 = 20 model years.
# Stop each run right after its 5th cycle (= 10 model years) has been saved.
cd /data/AIDE-SAI-link/linking_plots/climate_transport/data
while true; do
  for t in age2yr_band_trop143:age2yr_band_trop143.log age2yr_fullcol_p150_sfc:age2yr_fullcol.log; do
    tag=${t%%:*}; log=${t##*:}
    if grep -q "== year 5 done" $log && pgrep -f "tag $tag" >/dev/null; then pkill -f "tag $tag"; echo "$(date +%H:%M) stopped $tag after 10 model years"; fi
  done
  ! pgrep -f aoa_transport_2yr.py >/dev/null && grep -q "== year 5 done" age2yr_fullcol.log && grep -q "== year 5 done" age2yr_band_trop143.log && break
  sleep 30
done
