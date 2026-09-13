# Running GMP on the JSC real-robot Franka data

Tasks: `plate_sponge_sep9`, `plant_flower_2scoops`, `pottimer`.
Everything below is run from `/e/project1/m3/vanjani1/ssmpolicy`; the `.sbatch` files here
are copies of what lives in that repo's `env/`.

Run the stages in order. Each depends on the previous one's output.

## 0. Convert the data (once per task)

    python jsc/memmap_to_zarr.py --task plate_sponge_sep9

Writes `<out>/<task>/episode_data.zarr`. The store MUST be named `episode_data.zarr` and its
`episode_frame_nums` attribute MUST be a dict keyed by `str(index)`; the loader's docstring
says neither, and both fail unhelpfully.

## 1. Fit the normalizer (once per task, before any training)

    sbatch -J gmpfit_<task> --export=ALL,TASK=<task> jsc/fit_gmp_normalizer.sbatch

Without it the workspace refuses to instantiate: *"Normalizer is not found in the dataset."*
Uses `--quantile 1.0`, because anything below 1.0 asserts a `range_clip` normalizer and every
upstream task config uses plain `range`. It covers proprio and actions only (images are
`identity`), so changing image size does NOT require a refit.

## 2. Train the policies

    sbatch --nodes=2 -J gmp_<task>_pi     --export=ALL,TASK=<task>,POLICY=pi,CHAIN=1,WANT_NODES=2  jsc/train_gmp.sbatch
    sbatch --nodes=4 -J gmp_<task>_pi_mem --export=ALL,TASK=<task>,POLICY=pi_mem,CHAIN=1,WANT_NODES=4 jsc/train_gmp.sbatch

`pi` is the no-memory control, `pi_mem` the ungated memory policy. 21 epochs.
Measured: pi ~7-13 min/epoch on 2 nodes; pi_mem ~20 min/epoch on 4 nodes (sponge), ~39 min
(plant/pottimer).

`CHAIN=1` resubmits across the 12 h cap and needs no babysitting. `WANT_NODES` must be passed:
it is the INTENDED width, carried down the chain, so a slice that lands narrow cannot pin the
run there forever.

Kill switch for everything: `touch imitation-learning-policies/.gmp_chain_disabled`.

Optional safety net, resubmits a run that is neither queued nor finished (for a node failure
where the batch script never runs at all):

    while true; do bash jsc/gmp_watchdog.sh; sleep 900; done

## 3. Gate labels

    sbatch -J gmp_gate2_<task> --export=ALL,TASK=<task> jsc/gmp_gate_labels2.sbatch

Use THIS, not `scripts/generate_gate_labels.py`. That script runs both evaluation arms in one
process and cannot survive it: `BaseDenoisingPolicy.__del__` calls `SharedModelManager.reset()`,
clearing the global vision-model registry, so when the first arm's policy is garbage-collected
it evicts the ViT the second arm is using and the next forward raises
`KeyError: 'google/siglip2-base-patch16-256'`. It fails at a different batch every time
because the trigger is GC timing. This wrapper runs the arms as separate processes via
`imitation-learning-policies/jsc/gate_labels_split.py`.

It reuses a completed arm's `*_eval/train_results.pt`, so a re-run after a crash is cheap.

Knobs: `WITH_MEM_POOL` (default 200) and `NO_MEM_POOL` (default 5000, against upstream's
50000 — the pool is per episode and independent of episode length, so 50000 resamples each
frame of a 430-900 frame episode ~56x and that redundancy is the whole cost of the arm).
Set `NO_MEM_POOL=50000` to reproduce upstream exactly.

Output: `data/franka_<task>/<date>/<time>_train_comparison/val_results_statistics_window_5.pt`
(named `val_*` whatever the split — upstream hardcodes it).

## 4. Train the gate

    sbatch -J gmp_gate_<task> --export=ALL,TASK=<task> jsc/gmp_train_gate.sbatch

Picks up the newest `*_window_5.pt` from stage 3 automatically. It reads the SLIDING-WINDOW
file, not the raw merge, matching what `shell_scripts/train_gate.sh` derives.

Note the gate is NOT a frozen-feature probe: `train_memory_gate.yaml` overrides both
`image_encoder.frozen` and `freeze_mlp` to false, so the whole SigLIP tower is finetuned. It
is cheap only because `compose_memory_gate_hydra_config` always composes
`<dataset_type>_single_traj` — one frame per sample, not a 17-chunk two-camera stack.

## 5. Train the gated policy

Not yet run here. `diffusion_gated_transformer` inherits the memory transformer and adds the
gate, which stays frozen during policy training (`memory_gate.yaml` defaults). Two mechanisms
exist to avoid 21 fresh epochs and are worth trying before a from-scratch run:
`MemoryGate(ckpt_path=...)` loads the stage-4 gate directly, and `load_base_ckpt` loads
weights with `strict=False`, so the policy can warm-start from the `pi_mem` checkpoint.

## Splits

Not uniform, and the baselines must match whatever the comparison uses:
`plate_sponge_sep9` has NO held-out split and is evaluated train-fit; `plant_flower_2scoops`
and `pottimer` report on `test`.

## 6. Evaluating on the real robot

GMP is served, not embedded: the policy runs as a server on a GPU machine and the robot sends
observations to it over the network. Start it with

    shell_scripts/serve_policy_ckpt.sh <ckpt_path> <gpu_id>

or directly, which is what that script wraps:

    python scripts/run_policy_server.py \
      --ckpt_path data/franka_<task>/<date>/<time>_<task>_pi_mem/checkpoints/epoch_20_*.ckpt \
      --server_endpoint tcp://0.0.0.0:18765 --device cuda

The port defaults to `gpu_id + 18765`. Transport is `robotmq` (`RMQServer`), request/reply,
with these topics:

    policy_inference        send an observation dict, get an action dict back
    policy_reset            clear the memory/history between episodes -- REQUIRED, see below
    policy_config           query what the loaded checkpoint expects
    new_checkpoint_loaded   hot-swap a checkpoint without restarting
    done_rollout            end of episode
    export_recorded_data    dump what the server recorded

The request payload is a `robotmq.serialize`d dict carrying `episode_idx` plus the
observation entries named in the task config (`front_rgb`, `wrist_rgb`, `robot0_8d`). A scalar
`episode_idx` is treated as a single un-batched episode and the batch dimension is added and
removed for you. The reply is a dict of action arrays.

**The client is not in this repo.** It lives in the `real-env` submodule
(`git@github.com:real-stanford/real-env.git`), which is not checked out here and is
lab-specific anyway, so the robot side has to be written against your own Franka stack. The
protocol above is all it needs.

**Call `policy_reset` between episodes.** The memory policy accumulates trajectory history,
so without a reset episode N sees episode N-1's context and the numbers quietly drift. Note
that on this side `reset()` also clears the shared vision-model registry, which is the same
global that breaks two policies sharing one process -- one policy per server process.

### Hardware

Fits comfortably on the robot PC. Measured parameter counts:

| policy | params | fp16 weights |
|---|---|---|
| GMP `pi_mem` | 189 M (107 M SigLIP2 + 82 M denoising net) | 0.38 GB |
| our SSM policy | 642 M (423 M frozen DINOv2+T5) | 1.28 GB |
| MemoryVLA | 8377 M (6.7 B Llama-2 backbone) | 16.8 GB |

GMP runs on an 8 GB card directly, so the server can sit on the robot PC itself and the
network hop is optional. MemoryVLA cannot: 16.8 GB of weights does not fit an RTX 4060, or a
16 GB 4060 Ti, before any activations -- it has to be served from a bigger GPU or quantised.

### Offline comparison

For open-loop MAE against the SSM policy, evaluation does NOT need the robot and does not use
this server: it replays recorded observations from the memmap. See the harness at
`wt-main/scripts/openloop_memmap.py` and its MemoryVLA counterpart
`baselines/MemoryVLA/jsc/openloop_memvla.py`. Match the split per task (see above) and the
action execution horizon of whatever numbers you are comparing against.
