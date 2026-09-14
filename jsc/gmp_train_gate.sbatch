#!/bin/bash
#SBATCH -A m3
#SBATCH -p booster
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -t 12:00:00
#SBATCH --signal=B:USR1@900
#SBATCH -o /e/project1/m3/vanjani1/ssmpolicy/logs/%x-%j.out
#
# GMP stage 4: train the memory gate on the labels stage 3 produced.
# The gate is a SigLIP tower over the front camera plus proprio feeding a 512-wide MLP down
# to one scalar, trained with BCE against
#   with_mem_errors * threshold < no_mem_errors   ("memory measurably helps here").
# NOT a cheap frozen-feature probe: train_memory_gate.yaml overrides image_encoder.frozen
# AND freeze_mlp to false, so the whole tower is finetuned. It is cheap only relative to the
# policy, because compose_memory_gate_hydra_config always composes <dataset_type>_single_traj
# -- one frame per sample, not the 17-chunk two-camera stack a policy step carries.
set -uo pipefail
source /e/project1/m3/vanjani1/ssmpolicy/env/setup_env_gmp.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline MPLBACKEND=Agg
cd "$GMP_ROOT"

TASK="${TASK:?}"
N_NODES="${SLURM_JOB_NUM_NODES:-1}"; N_GPU=4
WORLD=$((N_NODES * N_GPU))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((29900 + SLURM_JOB_ID % 90))

# The gate trains on the SLIDING-WINDOW file, not the raw merge: train_gate.sh derives
# "${statistics_path%.pt}_window_${window_size}.pt" and stage 3 wrote it with window_size 5.
# Pointing at the merge instead would train on un-smoothed labels with no error to say so.
# Newest match wins: sponge has a stale comparison dir from the run that died on the
# task-name check before it wrote anything.
STATS=$(ls -t data/franka_${TASK}/*/*_train_comparison/*_window_5.pt 2>/dev/null | head -1)
if [ -z "$STATS" ]; then
  echo "no stage-3 window statistics for $TASK yet; looked in data/franka_${TASK}/*/*_train_comparison/"
  ls -la data/franka_${TASK}/*/*_train_comparison/ 2>/dev/null | head
  exit 1
fi
echo "statistics: $STATS"

# In stage 4 the GATE is workspace.model, and memory_gate_config.yaml points its proprio at
# ${...output_data_meta.robot0_10d} -- this repo's 10-d convention (3 position + 6d rotation +
# gripper). Our Franka proprio is robot0_8d (7 joints + gripper), so composition dies with
# InterpolationKeyError before a single batch is loaded. The task config's own memory_gate
# block does NOT cover this: that one sets workspace.model.memory_gate.proprio_meta, which is
# the path used when the gate is nested inside a POLICY, not when it is the model itself.
# Single-quoted so the outer shell leaves the ${...} for hydra to resolve.
PROPRIO_REF='\${workspace.train_dataset.output_data_meta.robot0_8d}'

# Single quotes so $SLURM_NODEID expands PER TASK inside srun, not once on node 0 -- the
# rendezvous trap that cost MemoryVLA a 15-minute timeout.
# Single quotes so $SLURM_NODEID expands PER TASK inside srun, not once on the submitting
# shell -- the rendezvous trap that cost MemoryVLA a full 15-minute timeout.
srun bash -c '
  accelerate launch \
    --num_machines '"$N_NODES"' --machine_rank $SLURM_NODEID \
    --main_process_ip '"$MASTER_ADDR"' --main_process_port '"$MASTER_PORT"' \
    --num_processes '"$WORLD"' --gpu_ids all \
    scripts/train_memory_gate.py \
      +task_name='"franka_$TASK"' \
      +policy_name=gate \
      +project_name=gmp_baseline \
      +logger_project_name=gmp_baseline \
      +run_name='"${TASK}_gate"' \
      ++workspace.train_dataset.root_dir=data/datasets/franka \
      ++workspace.train_dataset.compressed_dir=data/datasets/franka \
      ++workspace.train_dataset.normalizer_dir=data/datasets/franka \
      ++workspace.train_dataset.statistics_data_path='"$STATS"' \
      ++workspace.model.proprio_meta='"$PROPRIO_REF"' \
      '"${ADDITIONAL_ARGS:-}"'
'
rc=$?
echo "exit=$rc at $(date)"
exit $rc
