#!/bin/bash
# Reproduce the micro box comparison (run as shreyas; reads /data only).
set -e
D=/data/AIDE-SAI-link/linking_plots/micro_box; PY=/data/aide_emu_venv/bin/python
export XLA_PYTHON_CLIENT_PREALLOCATE=false LD_LIBRARY_PATH=/run/nvidia/driver/usr/lib/x86_64-linux-gnu
BARE="EMU_PHYS_TOP=0 EMU_PHYS_NUM=0 EMU_PHYS_COARSE=0 EMU_PHYS_BURST=0 EMU_ONLINE_CORR=0 EMU_PHYS_HSS=0 EMU_PHYS_ROT=0 EMU_PHYS_EVERY=0 EMU_TRIAGE=0 EMU_CHEM_NCLIP=1"
$PY $D/scripts/sample_cells.py $D/cells.npz                                   # CPU, ~5 min (80 frames over NFS)
env CUDA_VISIBLE_DEVICES=1 $BARE EMU_PROJECT=chem DEBUG=1 $PY $D/scripts/run_box.py $D/cells.npz $D/box_chem.npz fast,emulator &
env CUDA_VISIBLE_DEVICES=2 $BARE EMU_PROJECT=clip DEBUG=1 $PY $D/scripts/run_box.py $D/cells.npz $D/box_clip.npz emulator &
wait
$PY -c "
import numpy as np
a=dict(np.load('$D/box_chem.npz')); b=np.load('$D/box_clip.npz')
a.update({k: b[k] for k in b.files if k.startswith('emulator')}); np.savez('$D/box_clipvariant_merged.npz', **a)"
$PY $D/scripts/plot_box.py $D/cells.npz $D/box_chem.npz $D > $D/summary.txt
mkdir -p $D/supp; $PY $D/scripts/plot_box.py $D/cells.npz $D/box_clipvariant_merged.npz $D/supp _rawclip > $D/supp/summary_rawclip.txt
