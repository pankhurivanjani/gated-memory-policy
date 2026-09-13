#!/bin/bash
#SBATCH -A m3
#SBATCH -p booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -t 06:00:00
#SBATCH -o /e/project1/m3/vanjani1/ssmpolicy/logs/%x-%j.out
#
# GMP stage 3, retried as THREE processes. See jsc/gate_labels_split.py for why one process
# cannot work (a destructor clears the global vision-model registry mid-run).
# The memory arm already completed for all three tasks before the crash, so it is reused.
set -uo pipefail
source /e/project1/m3/vanjani1/ssmpolicy/env/setup_env_gmp.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline MPLBACKEND=Agg
cd "$GMP_ROOT"
TASK="${TASK:?}"
pick_best () { ls -1 data/franka_${TASK}/*/*_${TASK}_$1/checkpoints/epoch_*.ckpt 2>/dev/null \
  | sed -E 's|.*/epoch_([0-9]+)_.*|\1 &|' | sort -rn | head -1 | cut -d' ' -f2-; }
WITH_MEM=$(pick_best pi_mem); NO_MEM=$(pick_best pi)
[ -z "$WITH_MEM" ] || [ -z "$NO_MEM" ] && { echo "missing checkpoints for $TASK"; exit 1; }
echo "with_mem: $WITH_MEM"; echo "no_mem:   $NO_MEM"

# NO_MEM_POOL default 5000, not upstream's 50000. The pool is PER EPISODE and independent of
# episode length, so 50000 against 430-900 frame episodes resamples each frame ~56x with
# replacement -- that redundancy is the whole 3.5 h cost of this arm. 5000 still gives ~6-11
# samples per frame, enough for the error distribution the gate labels are built from, at a
# tenth of the time. A deliberate, documented deviation; raise NO_MEM_POOL to reproduce
# upstream exactly.
python -u jsc/gate_labels_split.py eval --ckpt "$WITH_MEM" --pool "${WITH_MEM_POOL:-200}" \
  --dataset_dir data/datasets/franka --dataset_type train || exit 1
python -u jsc/gate_labels_split.py eval --ckpt "$NO_MEM" --pool "${NO_MEM_POOL:-5000}" \
  --dataset_dir data/datasets/franka --dataset_type train || exit 1
OUT="data/franka_${TASK}/$(date +%Y-%m-%d)/$(date +%H-%M-%S)_train_comparison"
python -u jsc/gate_labels_split.py finalize --with_mem_ckpt "$WITH_MEM" --no_mem_ckpt "$NO_MEM" \
  --dataset_type train --window_size 5 --out_dir "$OUT"
rc=$?; echo "exit=$rc at $(date)"; exit $rc
