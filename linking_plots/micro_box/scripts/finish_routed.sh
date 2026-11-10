#!/bin/bash
# merge the routed parts with the reused numerical results and make main (routed) + supp figures
set -e
D=/data/AIDE-SAI-link/linking_plots/micro_box; PY=/data/aide_emu_venv/bin/python
$PY - <<PYEOF
import numpy as np, json
D='$D'
a = dict(np.load(f'{D}/box_chem.npz'))
p = [np.load(f'{D}/routed_part{i}.npz') for i in (0, 1)]
for k in ('emulator_N', 'emulator_M', 'emulator_S', 'emulator_H'):
    x = np.where(np.isnan(p[0][k]), p[1][k], p[0][k]); assert np.isfinite(x).all(), k
    # t=0 must equal the numerical t=0 (same frame cells)
    assert np.allclose(x[0], a['fast_' + k[-1]][0]), k
    a[k] = x
r = np.where(p[0]['route'] >= 0, p[0]['route'], p[1]['route']); assert (r >= 0).all(); a['route'] = r
for k in ('emulator_time', 'emulator_clip', 'triage_quiet'):
    a.pop(k, None)
np.savez(f'{D}/box_routed_merged.npz', **a)
g = json.loads(str(p[0]['glob'])) + json.loads(str(p[1]['glob']))
json.dump(g, open(f'{D}/routed_grid_fractions.json', 'w'), indent=0)
print('merged; grid calls', len(g))
PYEOF
mkdir -p $D/supp/bare $D/supp/rawclip
$PY $D/scripts/plot_box.py $D/cells.npz $D/box_routed_merged.npz $D '' 'Deployed TOMAS emulator (r1_tomas_emu routing)' > $D/summary.txt
$PY $D/scripts/plot_box.py $D/cells.npz $D/box_chem.npz $D/supp/bare _bare 'BARE emulator network (+chem closure, no routing)' > $D/supp/bare/summary_bare.txt
$PY $D/scripts/plot_box.py $D/cells.npz $D/box_clipvariant_merged.npz $D/supp/rawclip _rawclip 'RAW emulator network (clip only, no routing)' > $D/supp/rawclip/summary_rawclip.txt
