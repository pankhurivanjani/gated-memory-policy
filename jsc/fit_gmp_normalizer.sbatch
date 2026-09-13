#!/bin/bash
#SBATCH -A m3
#SBATCH -p booster
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH -t 02:00:00
#SBATCH -o /e/project1/m3/vanjani1/ssmpolicy/logs/%x-%j.out
# GMP needs a fitted normalizer in the dataset dir before training will start
# ("Normalizer is not found in the dataset. Please fit the normalizer first.").
# One fit per task, shared by pi and pi_mem so both see identical normalisation.
set -uo pipefail
source /e/project1/m3/vanjani1/ssmpolicy/env/setup_env_gmp.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline MPLBACKEND=Agg
cd "$GMP_ROOT"
TASK="${TASK:?}"
python -u scripts/fit_dataset_normalizer.py \
  "data/datasets/franka/${TASK}" "franka_${TASK}" \
  --policy_name diffusion_transformer --quantile 1.0
echo "EXIT=$?"
