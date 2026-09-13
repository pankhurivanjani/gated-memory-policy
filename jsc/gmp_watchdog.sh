#!/bin/bash
# Resubmit any GMP run that is neither finished nor queued.
#
# The in-job chain covers a clean exit and a walltime stop, but it cannot cover the case
# where the batch script never gets to run at all: NODE_FAIL, a hard preemption, or the
# script being killed outright. Then the run simply vanishes from the queue mid-training and
# waits for a human. This is the backstop for that.
#
# Idempotent by construction: a run already queued or running is skipped, so running it
# often is safe. It takes the same flock discipline as autoresume.sh, where four concurrent
# passes once submitted the same 13 cells and left 52 jobs where 13 belonged.
set -uo pipefail
ROOT=/e/project1/m3/vanjani1/ssmpolicy
GMP=$ROOT/baselines/gated-memory-policy/imitation-learning-policies
LOG=$ROOT/logs/gmp_watchdog.log
NUM_EPOCHS=${NUM_EPOCHS:-21}

exec 9>"$ROOT/logs/.gmp_watchdog.lock"
flock -w 300 9 || { echo "[$(date '+%F %T')] lock timeout, skipping" >> "$LOG"; exit 0; }

[ -f "$GMP/.gmp_chain_disabled" ] && { echo "[$(date '+%F %T')] sentinel present, standing down" >> "$LOG"; exit 0; }

# Snapshot the queue once: what is already running or waiting must never be resubmitted.
QUEUED=$(squeue -u "$USER" -h -o "%j" 2>/dev/null)

for TASK in plate_sponge_sep9 plant_flower_2scoops pottimer; do
  for POLICY in pi pi_mem; do
    run_name="${TASK}_${POLICY}"
    jobname="gmp_${run_name}"
    echo "$QUEUED" | grep -qx "$jobname" && continue          # in flight

    # RACE GUARD. The in-job chain resubmits from inside the dying slice, so there is a
    # window where the old job has left the queue and the new one is not in it yet. A
    # watchdog pass landing in that window submits a SECOND job for the same run, and both
    # resume into the same directory -- the duplicate-training failure this project has
    # already had once, when four concurrent passes turned 13 cells into 52 jobs. Give the
    # chain 15 minutes of grace after the last job for this run ended before stepping in.
    last_end=$(sacct -u "$USER" --name="$jobname" --format=End -nP -X 2>/dev/null \
               | grep -vE "Unknown|^$" | sort | tail -1)
    if [ -n "$last_end" ]; then
      age=$(( ($(date +%s) - $(date -d "$last_end" +%s 2>/dev/null || echo 0)) / 60 ))
      if [ "$age" -ge 0 ] && [ "$age" -lt 15 ]; then
        echo "[$(date '+%F %T')] $jobname: ended ${age}m ago, leaving it to the chain" >> "$LOG"
        continue
      fi
    fi
    ckdir=$(dirname "$(ls -t $GMP/data/franka_${TASK}/*/*_${run_name}/checkpoints/epoch_*.ckpt 2>/dev/null | head -1)" 2>/dev/null)
    done_ep=$(ls -1 "$ckdir"/epoch_*.ckpt 2>/dev/null | sed -E 's/.*epoch_([0-9]+)_.*/\1/' | sort -n | tail -1)
    done_ep="${done_ep:-0}"
    if [ -z "$ckdir" ] || [ ! -d "$ckdir" ]; then
      echo "[$(date '+%F %T')] $run_name: no checkpoints and not queued -- NOT resubmitting (never started; needs a look)" >> "$LOG"
      continue
    fi
    if [ "$done_ep" -ge "$((NUM_EPOCHS - 1))" ]; then
      echo "[$(date '+%F %T')] $run_name: finished at epoch $done_ep" >> "$LOG"
      continue
    fi
    nodes=4; [ "$POLICY" = "pi" ] && nodes=2
    jid=$(sbatch --parsable --nodes=$nodes -J "$jobname" \
      --export=ALL,TASK="$TASK",POLICY="$POLICY",RESUME=1,CHAIN=1,WANT_NODES=$nodes,NUM_EPOCHS=$NUM_EPOCHS,LAST_EP="$done_ep",RETRIES=0 \
      "$ROOT/env/train_gmp.sbatch" 2>/dev/null)
    echo "[$(date '+%F %T')] $run_name: stalled at epoch $done_ep, not queued -> resubmitted as $jid" >> "$LOG"
  done
done
