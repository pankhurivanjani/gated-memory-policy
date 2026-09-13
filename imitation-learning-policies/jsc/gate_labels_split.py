"""Stage 3 as three separate processes, because it cannot survive as one.

generate_gate_labels.py calls eval_model twice in a single process. That does not work:
BaseDenoisingPolicy.__del__ calls SharedModelManager.reset(), which clears the GLOBAL
vision-model registry. When the first evaluation's policy is garbage-collected -- at a moment
Python chooses, not the script -- it evicts the ViT that the second evaluation is actively
using, and the next forward dies with

    KeyError: 'google/siglip2-base-patch16-256'

On 2026-09-13 that killed all three tasks after 1287, 4876 and 5842 batches of the second
pass: different points, same cause, because the trigger is GC timing. Roughly three hours of
evaluation was lost per task.

Splitting the work into separate processes sidesteps it without touching upstream's
destructor: each process holds exactly one policy, and a destructor that clears global state
on the way out is then harmless.

Modes:
  eval      one arm, one process   (--ckpt, --pool)
  finalize  statistics, merge, sliding window, once both arms have results
"""
import os, sys, click
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.eval_model import eval_model
from scripts.calc_err_statistics import (calc_statistics, merge_statistics,
                                         plot_statistics, statistics_sliding_window)


def _paths(ckpt_path: str, dataset_type: str):
    d = "/".join(ckpt_path.split("/")[:-1])
    epoch = int(ckpt_path.split("/")[-1].split("_")[1])
    return d, epoch, f"{d}/epoch_{epoch}_eval/{dataset_type}_results.pt"


@click.group()
def cli(): ...


@cli.command("eval")
@click.option("--ckpt", required=True)
@click.option("--pool", type=int, required=True)
@click.option("--dataset_type", default="train")
@click.option("--dataset_dir", default="")
@click.option("--eval_episode_num", type=int, default=-1)
@click.option("--skip_if_done/--no-skip_if_done", default=True)
def eval_one(ckpt, pool, dataset_type, dataset_dir, eval_episode_num, skip_if_done):
    _, _, out = _paths(ckpt, dataset_type)
    if skip_if_done and os.path.exists(out):
        print(f"[eval] already done, reusing {out}", flush=True); return
    eval_model(ckpt_path=ckpt, rounds=1, dataset_dir=dataset_dir,
               eval_episode_num=eval_episode_num,
               index_pool_size_per_episode=pool, dataset_type=dataset_type)
    print(f"[eval] wrote {out}", flush=True)


@cli.command("finalize")
@click.option("--with_mem_ckpt", required=True)
@click.option("--no_mem_ckpt", required=True)
@click.option("--dataset_type", default="train")
@click.option("--eval_episode_num", type=int, default=-1)
@click.option("--window_size", type=int, default=5)
@click.option("--out_dir", required=True)
def finalize(with_mem_ckpt, no_mem_ckpt, dataset_type, eval_episode_num, window_size, out_dir):
    wd, we, wres = _paths(with_mem_ckpt, dataset_type)
    nd, ne, nres = _paths(no_mem_ckpt, dataset_type)
    for p in (wres, nres):
        if not os.path.exists(p):
            raise SystemExit(f"missing evaluation results: {p}")
    os.makedirs(out_dir, exist_ok=True)
    calc_statistics(nres, eval_episode_num=eval_episode_num)
    calc_statistics(wres, eval_episode_num=eval_episode_num)
    suf = f"_{eval_episode_num}" if eval_episode_num > 0 else ""
    wst = f"{wd}/epoch_{we}_eval/{dataset_type}_results_statistics{suf}.pt"
    nst = f"{nd}/epoch_{ne}_eval/{dataset_type}_results_statistics{suf}.pt"
    # Upstream names the merged file val_* whatever the split is; keep that so the gate's
    # ${statistics_path%.pt}_window_N.pt convention still finds it.
    merged = f"{out_dir}/val_results_statistics{suf}.pt"
    plot_statistics({"no_mem": nst, "with_mem": wst}, out_dir)
    merge_statistics(wst, nst, merged)
    if window_size > 1:
        statistics_sliding_window(merged, window_size)
        print(f"[finalize] wrote {merged.replace('.pt', f'_window_{window_size}.pt')}", flush=True)


if __name__ == "__main__":
    cli()
