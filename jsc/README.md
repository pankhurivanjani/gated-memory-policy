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
