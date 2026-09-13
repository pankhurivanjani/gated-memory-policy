"""LeRobot-memmap -> episode-wise zarr, the format GMP's EpisodicDataset reads.

Layout produced (matching imitation_learning/datasets/episodic_dataset.py):

    <out>/<task>/data.zarr
        .zattrs                      {"episode_frame_nums": {"0": n0, "1": n1, ...}}
        episode_0/
            front_rgb      (n, 224, 224, 3) uint8
            wrist_rgb      (n, 224, 224, 3) uint8
            joint_pos      (n, 7)  float32
            gripper_pos    (n, 1)  float32
            abs_actions    (n, 8)  float32
        episode_1/ ...

Our source is the SAME memmap the SSM policy trains on, so the two baselines see byte-identical
pixels and actions -- which is the point. Third writer over this source after
build_lerobot_franka_memmap.py and memmap_to_rlds.py.

  python memmap_to_zarr.py --memmap artifacts/memmap/plate_sponge_sep9 --split train \
      --out baselines/gated-memory-policy/imitation-learning-policies/data/datasets/franka
"""
import argparse, json, os
import numpy as np
import zarr
from numcodecs import Blosc


def convert(memmap_root: str, split: str, out_dir: str, task: str):
    root = os.path.join(memmap_root, split)
    meta = json.load(open(os.path.join(root, "meta.json")))
    T = meta["total_frames"]
    trajs = meta["trajectories"]

    act = np.memmap(os.path.join(root, "actions.dat"), dtype=np.float32, mode="r").reshape(T, -1)
    obs = np.memmap(os.path.join(root, "observations.dat"), dtype=np.float32, mode="r").reshape(T, -1)
    imgs = {}
    for cam in ("front_rgb", "wrist_rgb"):
        f = os.path.join(root, f"images_{cam}.dat")
        px = os.path.getsize(f) // T
        side = int(round((px // 3) ** 0.5))
        assert side * side * 3 == px, f"{cam}: {px} bytes/frame is not a square RGB image"
        imgs[cam] = np.memmap(f, dtype=np.uint8, mode="r").reshape(T, side, side, 3)
    print(f"  {split}: {T} frames, {len(trajs)} episodes, "
          f"act {act.shape[1]}d obs {obs.shape[1]}d, images {side}x{side}")

    os.makedirs(out_dir, exist_ok=True)
    # NOT "data.zarr": base_dataset._load_data_store looks for episode_data.zarr (or
    # <name>.zarr). The docstring in episodic_dataset.py says "data.zarr" and is wrong.
    store_path = os.path.join(out_dir, task, "episode_data.zarr")
    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    z = zarr.open(store_path, mode="w")
    # Blosc/zstd: the images dominate and compress well; without it this is ~6.4 GB per task.
    comp = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

    lens = []
    for i, tr in enumerate(trajs):
        s, n = tr["start_idx"], tr["length"]
        g = z.create_group(f"episode_{i}")
        for cam in ("front_rgb", "wrist_rgb"):
            g.create_dataset(cam, data=np.asarray(imgs[cam][s:s + n]),
                             chunks=(1, side, side, 3), compressor=comp)
        a = np.asarray(act[s:s + n], dtype=np.float32)
        o = np.asarray(obs[s:s + n], dtype=np.float32)
        # 8 dims = 7 joints + gripper. Split so the config can compose them the way
        # robomimic composes eef_pos/quat/gripper into robot0_10d.
        g.create_dataset("joint_pos",   data=o[:, :7],  chunks=(n, 7))
        g.create_dataset("gripper_pos", data=o[:, 7:8], chunks=(n, 1))
        g.create_dataset("abs_actions", data=a,         chunks=(n, a.shape[1]))
        lens.append(int(n))
        if i % 10 == 0:
            print(f"    episode_{i}: {n} frames", flush=True)

    # dict keyed by STRINGIFIED episode index, not a list: _check_data_validity does
    #   store_episode_frame_nums[str(episode_idx)]
    # and calls .update() on it. A list gets past the writer and fails at dataset build.
    z.attrs["episode_frame_nums"] = {str(i): n for i, n in enumerate(lens)}
    z.attrs["source_memmap"] = os.path.abspath(root)
    z.attrs["action_dim"] = int(act.shape[1])
    z.attrs["proprio_dim"] = int(obs.shape[1])
    print(f"  wrote {store_path}  ({len(lens)} episodes, {sum(lens)} frames)")
    return store_path, lens


def verify(store_path, memmap_root, split, lens):
    """Round-trip one episode against the memmap. A converter that silently drops or
    misaligns frames is the failure mode that costs a whole training run."""
    root = os.path.join(memmap_root, split)
    meta = json.load(open(os.path.join(root, "meta.json")))
    T = meta["total_frames"]
    act = np.memmap(os.path.join(root, "actions.dat"), dtype=np.float32, mode="r").reshape(T, -1)
    z = zarr.open(store_path, mode="r")
    efn = z.attrs["episode_frame_nums"]
    assert isinstance(efn, dict), f"episode_frame_nums must be a dict, got {type(efn).__name__}"
    assert [efn[str(i)] for i in range(len(lens))] == lens, "episode_frame_nums mismatch"
    for i in (0, len(lens) // 2, len(lens) - 1):
        tr = meta["trajectories"][i]
        s, n = tr["start_idx"], tr["length"]
        za = z[f"episode_{i}"]["abs_actions"][:]
        assert za.shape[0] == n, f"episode_{i}: {za.shape[0]} rows, memmap says {n}"
        assert np.array_equal(za, np.asarray(act[s:s + n])), f"episode_{i}: actions differ"
    print(f"  VERIFIED: episode_frame_nums and actions match the memmap byte-for-byte "
          f"(checked episodes 0, {len(lens)//2}, {len(lens)-1})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--memmap", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default=None)
    a = ap.parse_args()
    task = a.task or os.path.basename(a.memmap.rstrip("/"))
    p, lens = convert(a.memmap, a.split, a.out, task)
    verify(p, a.memmap, a.split, lens)
