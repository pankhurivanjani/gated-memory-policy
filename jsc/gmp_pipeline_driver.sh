#!/bin/bash
# Drive GMP stages 3 -> 4 -> 5 per task, unattended.
#
# The stages are strictly sequential and each reads a file the previous one writes, so this
# just watches for that file and submits the next stage. Idempotent: it only ever submits when
# the input exists AND nothing for that stage is already queued, so running it on a loop is
# safe. Matches jobs by NAME, because scontrol does not expose --export variables and a guard
# that greps for TASK= matches nothing on every job and so always fires.
set -uo pipefail
ROOT=/e/project1/m3/vanjani1/ssmpolicy
GMP=$ROOT/baselines/gated-memory-policy/imitation-learning-policies
LOG=$ROOT/logs/gmp_pipeline.log
exec 9>"$ROOT/logs/.gmp_pipeline.lock"
flock -w 60 9 || exit 0
[ -f "$GMP/.gmp_chain_disabled" ] && exit 0

say () { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
QUEUED=$(squeue -u "$USER" -h -o "%j" 2>/dev/null)
queued () { echo "$QUEUED" | grep -qx "$1"; }

for TASK in plate_sponge_sep9 plant_flower_2scoops pottimer; do
  short="${TASK%%_*}"
  stats=$(ls -t "$GMP"/data/franka_${TASK}/*/*_train_comparison/*_window_5.pt 2>/dev/null | head -1)
  # A FINISHED gate, not any gate. Checkpoints appear every epoch, so globbing epoch_*.ckpt
  # matched epoch_0 eleven minutes into a 72-minute run and launched stage 5 against a gate
  # that had seen one epoch -- four nodes training a gated policy around a near-random gate,
  # which would have looked like a plausible result rather than an error. Require the last
  # epoch AND no gate job still queued for this task.
  gate=$(ls -t "$GMP"/data/franka_${TASK}/*/*_${TASK}_gate/checkpoints/epoch_${GATE_LAST_EPOCH:-20}_*.ckpt 2>/dev/null | head -1)
  if queued "gmp_gate_${short}"; then gate=""; fi

  # ---- stage 4: needs stage 3's window statistics ----
  if [ -n "$stats" ] && [ -z "$gate" ]; then
    if queued "gmp_gate_${short}" || queued "gmp_gate2_${short}"; then :; else
      jid=$(sbatch --parsable --nodes=4 -J "gmp_gate_${short}" \
            --export=ALL,TASK="$TASK" "$ROOT/env/gmp_train_gate.sbatch" 2>/dev/null)
      say "$TASK: stage 3 done -> stage 4 (gate) submitted as $jid"
    fi
    continue
  fi

  # ---- stage 5: needs a trained gate ----
  if [ -n "$gate" ]; then
    done5=$(ls -t "$GMP"/data/franka_${TASK}/*/*_${TASK}_pi_gated/checkpoints/epoch_20_*.ckpt 2>/dev/null | head -1)
    if [ -n "$done5" ]; then say "$TASK: stage 5 finished"; continue; fi
    if queued "gmp_${TASK}_pi_gated"; then continue; fi
    # Warm-start from pi_mem: pi_gated inherits the memory transformer and the gate stays
    # frozen (memory_gate.yaml defaults), so the policy does not need to relearn from scratch.
    # load_base_ckpt uses strict=False, so the gate's absence from the base checkpoint is fine.
    base=$(ls -t "$GMP"/data/franka_${TASK}/*/*_${TASK}_pi_mem/checkpoints/epoch_20_*.ckpt 2>/dev/null | head -1)
    extra="+workspace.model.memory_gate.ckpt_path=$gate"
    [ -n "$base" ] && extra="$extra +base_ckpt_path=$base"
    jid=$(sbatch --parsable --nodes=4 -J "gmp_${TASK}_pi_gated" \
          --export=ALL,TASK="$TASK",POLICY=pi_gated,CHAIN=1,WANT_NODES=4,ADDITIONAL_ARGS="$extra" \
          "$ROOT/env/train_gmp.sbatch" 2>/dev/null)
    say "$TASK: gate ready -> stage 5 (pi_gated) submitted as $jid, warm-started from pi_mem"
  fi
done
