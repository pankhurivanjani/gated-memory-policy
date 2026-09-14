#!/bin/bash
#SBATCH -A m3
#SBATCH -p booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH -t 04:00:00
#SBATCH -o /e/project1/m3/vanjani1/ssmpolicy/logs/%x-%j.out
# Open-loop MAE for a GMP checkpoint, same split/ground-truth/metric as the SSM policy.
set -uo pipefail
SSM=/e/project1/m3/vanjani1/ssmpolicy
source $SSM/env/setup_env_gmp.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLBACKEND=Agg
cd "$GMP_ROOT"
TASK="${TASK:?}"; ARM="${ARM:-pi_mem}"
# Split is per task: plate_sponge_sep9 has no held-out split (its SSM numbers are train-fit).
case "$TASK" in plate_sponge_sep9) SPLIT="${SPLIT:-train}" ;; *) SPLIT="${SPLIT:-test}" ;; esac
CK=$(ls -1 data/franka_${TASK}/*/*_${TASK}_${ARM}/checkpoints/epoch_*.ckpt 2>/dev/null \
     | sed -E 's|.*/epoch_([0-9]+)_.*|\1 &|' | sort -rn | head -1 | cut -d' ' -f2-)
[ -z "$CK" ] && { echo "no $ARM checkpoint for $TASK"; exit 1; }
echo "task=$TASK arm=$ARM split=$SPLIT h=${H:-4}"
python -u jsc/openloop_gmp.py --checkpoint "$CK" \
  --memmap "$SSM/artifacts/memmap/${TASK}" --split "$SPLIT" \
  --exec_horizon "${H:-4}" ${EPISODES:+--episodes $EPISODES} \
  --out "$SSM/artifacts/openloop_gmp/${TASK}__${ARM}__h${H:-4}.json"
rc=$?; echo "exit=$rc at $(date)"; exit $rc
