#!/bin/bash
#SBATCH -A m3
#SBATCH -p booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -t 12:00:00
# B: sends the signal to THIS shell, not to the job step. Without it SLURM SIGTERMs the step
# at the cap and SIGKILLs the script seconds later, so the chain block at the bottom never
# runs and a multi-slice run simply stops. Every MemoryVLA slice that chained had ended early
# on its own -- the walltime path had never actually been exercised.
#SBATCH --signal=B:USR1@900
#SBATCH -o /e/project1/m3/vanjani1/ssmpolicy/logs/%x-%j.out
#
# Gated Memory Policy baseline. POLICY selects the arm:
#   pi      -> diffusion_transformer         (no history at all)
#   pi_mem  -> diffusion_memory_transformer + memory_gate=null  (memory ALWAYS on, ungated)
# The config ships memory_gate populated, so pi_mem MUST override it to null -- otherwise a
# gate trains and the run is not the ungated baseline, with nothing in the log to show it.
set -uo pipefail
source /e/project1/m3/vanjani1/ssmpolicy/env/setup_env_gmp.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline MPLBACKEND=Agg
cd "$GMP_ROOT"

TASK="${TASK:?}"; POLICY="${POLICY:?}"
# base_trainer.yaml default; the chain needs it to know when the run is finished.
NUM_EPOCHS="${NUM_EPOCHS:-21}"
N_NODES="${SLURM_JOB_NUM_NODES:-1}"; N_GPU=4
# Intended width, sticky down the whole chain. Using the width THIS slice happened to get
# would let a run that once landed narrow re-submit itself narrow forever.
WANT_NODES="${WANT_NODES:-$N_NODES}"
WORLD=$((N_NODES * N_GPU))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((29600 + SLURM_JOB_ID % 300))

case "$POLICY" in
  pi)      policy_name="diffusion_transformer";        extra="" ;;
  pi_mem)  policy_name="diffusion_memory_transformer"; extra="+workspace.model.memory_gate=null" ;;
  *) echo "POLICY must be pi or pi_mem"; exit 1 ;;
esac

run_name="${TASK}_${POLICY}"
echo "task=$TASK policy=$POLICY($policy_name) nodes=$N_NODES world=$WORLD run=$run_name"


# srun in the background, so the USR1 trap can run while we wait. A foreground srun would
# block the shell and the handler would only fire after it returned -- i.e. after SIGKILL.
SRUN_PID=""
on_walltime() {
  echo "[walltime] USR1 at $(date) -- stopping the trainer so the chain block can run"
  [ -n "$SRUN_PID" ] && kill -TERM "$SRUN_PID" 2>/dev/null
}
WALLTIME_STOP=0
trap 'WALLTIME_STOP=1; on_walltime' USR1
run_srun() {
  srun "$@" &
  SRUN_PID=$!
  # `wait` returns early when a trapped signal arrives, so wait again for the real exit.
  wait "$SRUN_PID"; local r=$?
  if [ "$r" -gt 128 ]; then wait "$SRUN_PID" 2>/dev/null; r=$?; fi
  return $r
}

# NOTE the single quotes: $SLURM_NODEID must expand PER TASK inside srun. Double quotes would
# let the submitting shell on node 0 expand it once, every node would launch as machine_rank 0,
# and rendezvous would hang until timeout -- exactly what happened to MemoryVLA on 2026-09-12.

# 21 epochs do not fit the 12 h QOS cap, so slices chain. train_policy.py has NO resume path
# -- it only ever calls workspace.train() -- but scripts/train_diffusion_policy_resume.py
# rebuilds the config from the checkpoint itself and calls workspace.resume_training(), which
# restores model, EMA, optimizer, scheduler and epoch. Because the config comes from the
# checkpoint, a resumed slice CANNOT drift from the settings the run started with, and it
# takes no hydra overrides -- so every override below applies to the first slice only.
#
# output_dir is data/<task>/<date>/<time>_<run_name>, a fresh directory per submission, but a
# resumed slice writes into the ORIGINAL one because that path is in the checkpoint's config.
# So look for the newest latest.ckpt across all of them, not just the most recent directory.
# workspace.ckpt, NOT latest.ckpt. Both are written on every save, but they differ:
#   latest.ckpt    (~755 MB) model_state_dict + normalizer only
#   workspace.ckpt (~2.9 GB) + optimizer_state_dict, lr_scheduler_state_dict, ema_model
# _load_workspace prints "No optimizer state dict found in ckpt" and carries on with a FRESH
# optimizer and a FRESH cosine schedule when they are absent. Resuming from latest.ckpt at
# epoch 18 of 21 would therefore re-run 2000 warmup steps and ramp the LR back up over the
# last three epochs -- a quietly damaged final model, with only that one log line to show it.
RUN_GLOB="data/franka_${TASK}/*/*_${run_name}/checkpoints/workspace.ckpt"
LATEST=$(ls -t $RUN_GLOB 2>/dev/null | head -1)
if [ -z "$LATEST" ]; then
  # workspace.ckpt is skipped when a previous epoch's save thread still holds it, so fall
  # back rather than start from scratch -- a reset optimizer still beats losing 18 epochs.
  LATEST=$(ls -t data/franka_${TASK}/*/*_${run_name}/checkpoints/latest.ckpt 2>/dev/null | head -1)
  [ -n "$LATEST" ] && echo "[resume] WARNING: no workspace.ckpt; falling back to $LATEST (optimizer state will reset)"
fi

if [ "${RESUME:-0}" = "1" ] && [ -n "$LATEST" ]; then
  echo "[resume] $LATEST"
  run_srun bash -c '
    accelerate launch \
      --num_machines '"$N_NODES"' --machine_rank $SLURM_NODEID \
      --main_process_ip '"$MASTER_ADDR"' --main_process_port '"$MASTER_PORT"' \
      --num_processes '"$WORLD"' --gpu_ids all \
      scripts/train_diffusion_policy_resume.py --ckpt_path '"$LATEST"'
  '
else
  [ "${RESUME:-0}" = "1" ] && echo "[resume] asked for, but no checkpoint under $RUN_GLOB -- starting fresh"
  run_srun bash -c '
    accelerate launch \
      --num_machines '"$N_NODES"' --machine_rank $SLURM_NODEID \
      --main_process_ip '"$MASTER_ADDR"' --main_process_port '"$MASTER_PORT"' \
      --num_processes '"$WORLD"' --gpu_ids all \
      scripts/train_policy.py \
        +task_name='"franka_$TASK"' \
        +policy_name='"$policy_name"' \
        +project_name=gmp_baseline \
        +train_server_name=localhost \
        +logger_project_name=gmp_baseline \
        +run_name='"$run_name"' \
        +workspace.train_dataset.root_dir=data/datasets/franka \
        +workspace.train_dataset.compressed_dir=data/datasets/franka \
        +workspace.train_dataset.normalizer_dir=data/datasets/franka \
        '"$extra"' '"${ADDITIONAL_ARGS:-}"'
  '
fi
rc=$?
# EXIT WITH THE TRAINER'S CODE. Ending on `echo` made the script exit 0, so sacct reported
# State=COMPLETED for a run that had died of CUDA OOM on all eight ranks. Job state is the
# first thing checked when a run is unattended; it must not lie.
echo "exit=$rc at $(date)"

# ---- chain ----------------------------------------------------------------
# A deliberate scancel sends SIGTERM, srun exits, and this block would otherwise resubmit the
# very run just killed. The sentinel is the kill switch (same design as MemoryVLA).
SENTINEL="$GMP_ROOT/.gmp_chain_disabled"
CKDIR=$(dirname "$(ls -t data/franka_${TASK}/*/*_${run_name}/checkpoints/epoch_*.ckpt 2>/dev/null | head -1)" 2>/dev/null)
# The epoch is recorded only in the per-epoch filenames (epoch_<N>_train_mean_loss_<v>.ckpt);
# latest.ckpt is a copy and carries no epoch in its name.
DONE_EP=$(ls -1 "$CKDIR"/epoch_*.ckpt 2>/dev/null | sed -E 's/.*epoch_([0-9]+)_.*/\1/' | sort -n | tail -1)
DONE_EP="${DONE_EP:-0}"
LAST_EP="${LAST_EP:-0}"
if [ -f "$SENTINEL" ]; then
  echo "[chain] $SENTINEL present -- not resubmitting"
elif [ "${CHAIN:-0}" != "1" ]; then
  :
elif [ "$rc" -ne 0 ] && [ "$WALLTIME_STOP" != "1" ] && [ "${RETRIES:-0}" -ge 2 ]; then
  echo "[chain] slice exited $rc after ${RETRIES:-0} retries -- giving up (a real failure, not a blip)"
elif [ "$rc" -ne 0 ] && [ "$WALLTIME_STOP" != "1" ]; then
  # A crash is not always fatal to the RUN: a lost node, an NCCL timeout or a transient
  # filesystem error kills the slice but nothing is wrong with the training. Refusing to
  # resubmit turns a blip into a permanent stop that nobody notices until morning. Retry a
  # bounded number of times, carrying the count forward so a genuine crash loop still halts.
  echo "[chain] slice exited $rc (retry $(( ${RETRIES:-0} + 1 )) of 2) -- resubmitting from the last checkpoint"
  jid=$(sbatch --parsable --nodes="$WANT_NODES" -J "$SLURM_JOB_NAME" \
        --export=ALL,TASK="$TASK",POLICY="$POLICY",RESUME=1,CHAIN=1,\
WANT_NODES="$WANT_NODES",NUM_EPOCHS="$NUM_EPOCHS",LAST_EP="$LAST_EP",RETRIES=$(( ${RETRIES:-0} + 1 )) \
        /e/project1/m3/vanjani1/ssmpolicy/env/train_gmp.sbatch)
  echo "[chain] -> $jid"
elif [ "$DONE_EP" -ge "$((NUM_EPOCHS - 1))" ]; then
  echo "[chain] $run_name reached epoch $DONE_EP of $NUM_EPOCHS -- done"
elif [ "$DONE_EP" -le "$LAST_EP" ]; then
  echo "[chain] NO PROGRESS (newest epoch $DONE_EP <= resumed-from $LAST_EP) -- refusing to resubmit"
else
  jid=$(sbatch --parsable --nodes="$WANT_NODES" -J "$SLURM_JOB_NAME" \
        --export=ALL,TASK="$TASK",POLICY="$POLICY",RESUME=1,CHAIN=1,\
WANT_NODES="$WANT_NODES",NUM_EPOCHS="$NUM_EPOCHS",LAST_EP="$DONE_EP",RETRIES=0 \
        /e/project1/m3/vanjani1/ssmpolicy/env/train_gmp.sbatch)
  echo "[chain] $run_name at epoch $DONE_EP -> resubmitted as $jid"
fi
exit $rc
