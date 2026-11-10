#!/usr/bin/env bash
# v6 ("toward 1 %") launch recipes — docs/RADIATION.md, "v6: toward 1 %".
# Run as ubuntu from anywhere:   sudo -u ubuntu scripts/radiation/launch_v6.sh <stage> <gpu>
# Stages (one GPU each; the two label stages first, the aer-cum-v3 stage after they finish):
#   labels-jan   resume train_v3jan at globe 41 (new seed, so no draw repeats globes 0-40)
#   labels-yr    resume train_v3yr  at globe 40 (--dt-cap 5)
#   planck-jac   Planck net, Jacobian head, v4 recipe otherwise (11 M columns, ~1 h)
#   planck-s1    Planck net, v4 recipe, --seed 1 (2nd seed for the ensemble test)
#   aer-s1       aerosol net, v5 recipe, --seed 1
#   aer-cum      aerosol net, cum head, v1-v2 labels (8 M columns, ~2.5 h)
#   aer-cum-v3   aerosol net, cum head, v1-v3 labels (16 M columns, ~5 h) — needs the label stages done
#   aer-mixed-v3 aerosol net, v5 recipe (mixed head) on the v1-v3 labels: the pure data-doubling test
#   aer-cum-v3-lr1e-4  cum head on v1-v3 at lr 1e-4 (lr 3e-4 spiked 17x in 53 epochs on 2026-10-15)
#   aer-cum-v3-s1 / planck-jac-s1   second seeds of the v6 production nets (ensemble test, 2026-10-17)
#   planck-jac-v3   Planck jac head on the v1-v3 aerosol labels + both noise sets (19.5 M cols): LW 4.3 -> 3.3 % (2026-10-17)
#   labels-planck-v3 / planck-jac-v4   600 more Planck-only globes (28k cols each), then the jac net on ~36 M cols (2nd data doubling)
#   smoke-jac / smoke-cum   2-shard, 2-epoch code checks of the two new heads on the GPU
# HOST MEMORY (2026-10-16): a v1-v3 aerosol trainer holds ~255 GB (cum head) / ~210 GB (mixed) of host RAM
# for the whole run; the node has 1 TB, no swap, and user jobs are capped at 880 GB (user.slice). Two
# trainers of the OLD loader (3-4 copies + pin_memory of the set) hung the node on 10-14 and 10-15.
# `free -g` first; the trainer refuses to start when the set would exceed 80 % of MemAvailable.
set -euo pipefail
cd "$(dirname "$0")/../.."
stage=${1:?stage}; gpu=${2:?gpu}
PY=/data/aide_emu_venv/bin/python
ENV=(LD_LIBRARY_PATH=/run/nvidia/driver/usr/lib/x86_64-linux-gnu XLA_PYTHON_CLIENT_PREALLOCATE=false
     HDF5_USE_FILE_LOCKING=FALSE P_LO_HPA=1 P_HI_HPA=150 INIT_BIN=so4 CUDA_VISIBLE_DEVICES=$gpu)
L=outputs/rad_emulator/labels; C=outputs/rad_emulator/ckpt; VAL=$L/val_phys30d
TR_AER_V2="$L/train_v1 $L/train_v1b $L/train_v1c $L/train_v2jan $L/train_v2yr"
TR_AER_V3="$TR_AER_V2 $L/train_v3jan $L/train_v3yr"
TR_PL_V2="$TR_AER_V2 $L/planck_noise_v1 $L/planck_noise_v2"
TR_PL_V3="$TR_AER_V3 $L/planck_noise_v1 $L/planck_noise_v2"   # planck_jac_v3 (19.5 M cols, 204 GB host, 2026-10-17)
TR_PL_V4="$TR_PL_V3 $L/planck_noise_v3"                      # + 600 Planck-only globes x 28k cols (~36 M cols, ~370 GB host)
PLANCK_V4="--net planck --width 768 --blocks 4 --dt-shape 1 --dt-eps 0.02 --x-clip 10 --epochs 300 --max-dt 3 --lr 3e-4"
AER_V5="--net aer --width 1024 --blocks 5 --tau-w 3e-7 --x-clip 10 --epochs 200 --max-dt 5 --max-aod 0.5 --lr 3e-4 --x-cpu"
launch() {  # launch <log> <script args...>
  local log=$1; shift
  setsid nohup env "${ENV[@]}" $PY "$@" > "$log" 2>&1 < /dev/null &
  echo "$stage: GPU $gpu, pid $!, log $log"
}
case $stage in
  labels-jan) launch $L/train_v3jan_resume.log scripts/radiation/label_columns.py \
      --states runs/raddata_30NS_jan/rad_sample_raddata_30NS_jan_s*.npz --n-globes 300 --out $L/train_v3jan --seed 31 --start 41 ;;
  labels-yr)  launch $L/train_v3yr_resume.log scripts/radiation/label_columns.py \
      --states runs/lr1yr_d177_sample/rad_sample_lr1yr_d177_sample_s*.npz --n-globes 300 --out $L/train_v3yr --seed 32 --start 40 --dt-cap 5 ;;
  planck-jac) launch $C/planck_jac.log scripts/radiation/train_rad_mlp.py $PLANCK_V4 --head jac --train $TR_PL_V2 --val $VAL --out $C/planck_jac.npz ;;
  planck-s1)  launch $C/planck_v5s1.log scripts/radiation/train_rad_mlp.py $PLANCK_V4 --seed 1 --x-cpu --train $TR_PL_V2 --val $VAL --out $C/planck_v5s1.npz ;;
  aer-s1)     launch $C/aer_v5s1.log scripts/radiation/train_rad_mlp.py $AER_V5 --head mixed --seed 1 --train $TR_AER_V2 --val $VAL --out $C/aer_v5s1.npz ;;
  aer-cum)    launch $C/aer_cum.log scripts/radiation/train_rad_mlp.py $AER_V5 --head cum --train $TR_AER_V2 --val $VAL --out $C/aer_cum.npz ;;
  aer-cum-v3) launch $C/aer_cum_v3.log scripts/radiation/train_rad_mlp.py $AER_V5 --head cum --train $TR_AER_V3 --val $VAL --out $C/aer_cum_v3.npz ;;
  aer-mixed-v3) launch $C/aer_v6mixed.log scripts/radiation/train_rad_mlp.py $AER_V5 --head mixed --train $TR_AER_V3 --val $VAL --out $C/aer_v6mixed.npz ;;
  aer-cum-v3-lr1e-4) launch $C/aer_cum_v3_lr1e-4.log scripts/radiation/train_rad_mlp.py $AER_V5 --head cum --lr 1e-4 --train $TR_AER_V3 --val $VAL --out $C/aer_cum_v3_lr1e-4.npz ;;
  aer-cum-v3-s1) launch $C/aer_cum_v3_s1.log scripts/radiation/train_rad_mlp.py $AER_V5 --head cum --lr 1e-4 --seed 1 --train $TR_AER_V3 --val $VAL --out $C/aer_cum_v3_s1.npz ;;
  planck-jac-s1) launch $C/planck_jac_s1.log scripts/radiation/train_rad_mlp.py $PLANCK_V4 --head jac --seed 1 --train $TR_PL_V2 --val $VAL --out $C/planck_jac_s1.npz ;;
  planck-jac-v3) launch $C/planck_jac_v3.log scripts/radiation/train_rad_mlp.py $PLANCK_V4 --head jac --x-cpu --train $TR_PL_V3 --val $VAL --out $C/planck_jac_v3.npz ;;
  planck-jac-v4) launch $C/planck_jac_v4.log scripts/radiation/train_rad_mlp.py $PLANCK_V4 --head jac --x-cpu --train $TR_PL_V4 --val $VAL --out $C/planck_jac_v4.npz ;;
  labels-planck-v3) launch $L/planck_noise_v3.log scripts/radiation/label_columns.py \
      --states runs/raddata_30NS_jan/rad_sample_*.npz runs/lr1yr_d177_sample/rad_sample_*.npz --planck-only --dt-noise 0.05 --keep 28000 --n-globes 600 --seed 3 --out $L/planck_noise_v3 ;;
  smoke-jac)  launch $C/smoke_jac.log scripts/radiation/train_rad_mlp.py $PLANCK_V4 --head jac --epochs 2 --max-shards 2 --train $L/train_v2jan $L/planck_noise_v2 --val $VAL --out $C/smoke_jac.npz ;;
  smoke-cum)  launch $C/smoke_cum.log scripts/radiation/train_rad_mlp.py $AER_V5 --head cum --epochs 2 --max-shards 2 --train $L/train_v2jan $L/train_v2yr --val $VAL --out $C/smoke_cum.npz ;;
  *) echo "unknown stage $stage"; exit 2 ;;
esac
