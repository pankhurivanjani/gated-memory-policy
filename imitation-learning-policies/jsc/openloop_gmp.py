"""Open-loop rollout of a trained GMP policy on an ssmpolicy memmap split.

The GMP counterpart of wt-main/scripts/openloop_memmap.py and
baselines/MemoryVLA/jsc/openloop_memvla.py: same split, same ground truth, same metric
(mean |predicted - teleoperated| over every frame and every action dimension), so the three
numbers are comparable.

"Open loop" means the policy sees the RECORDED observation at every step and its own
predictions never affect what it sees next.

Three things this has to get right, each of which fails silently otherwise:

  * The policy is rebuilt from cfg_str_unresolved INSIDE the checkpoint, the same way
    scripts/train_diffusion_policy_resume.py does. Constructing it from the on-disk config
    instead would let evaluation drift from training whenever a config is edited.

  * predict_action takes a NORMALIZED batch and returns a NORMALIZED action. The normalizer
    travels in the checkpoint (normalizer_state_dict); skipping either direction yields
    plausible numbers in the wrong units.

  * policy.reset() between episodes. A memory policy accumulates trajectory history, so
    without it episode N is conditioned on episode N-1 and the MAE drifts quietly downward
    on later episodes.

Images are resized 224 -> 256 because siglip2-base-patch16-256 has a fixed 16x16 position
embedding; this mirrors the Resize in the task config that training used.
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omegaconf import DictConfig, OmegaConf
import hydra, dill
from robot_utils.torch_utils import torch_load
from robot_utils.config_utils import register_resolvers

IMG = 224
SSM_OPENLOOP = "/e/project1/m3/vanjani1/ssmpolicy/wt-main/scripts/openloop_memmap.py"


def _load_phases():
    """Lift _phases from the SSM harness so phase MAE is computed identically."""
    try:
        src = open(SSM_OPENLOOP).read()
        ns: dict = {}
        start = src.index("def _phases(")
        end = src.index("\ndef ", start + 1)
        exec("import numpy as np\n" + src[start:end], ns)
        return ns["_phases"], "lifted from openloop_memmap.py"
    except Exception as e:
        print(f"WARNING: could not lift _phases ({type(e).__name__}); phase MAE omitted", flush=True)
        return (lambda gt, **kw: (None, None)), "unavailable"


def _load_split(root: Path, split: str):
    d = root / split
    meta = json.load(open(d / "meta.json"))
    n = meta["total_frames"]
    if n == 0:
        raise SystemExit(f"{d} has 0 frames -- this task was built without a held-out split")
    obs = np.memmap(d / "observations.dat", dtype="float32", mode="r", shape=(n, meta["observation_dim"]))
    act = np.memmap(d / "actions.dat", dtype="float32", mode="r", shape=(n, meta["action_dim"]))
    front = np.memmap(d / "images_front_rgb.dat", dtype="uint8", mode="r", shape=(n, IMG, IMG, 3))
    wrist_p = d / "images_wrist_rgb.dat"
    wrist = np.memmap(wrist_p, dtype="uint8", mode="r", shape=(n, IMG, IMG, 3)) if wrist_p.exists() else None
    return meta, obs, act, front, wrist


def build_policy(ckpt_path: str, device: str):
    register_resolvers()
    ckpt = torch_load(ckpt_path, pickle_module=dill)
    cfg = OmegaConf.create(ckpt["cfg_str_unresolved"])
    OmegaConf.set_struct(cfg, True)
    assert isinstance(cfg, DictConfig)
    cfg["workspace"]["cfg_str_unresolved"] = ckpt["cfg_str_unresolved"]
    workspace = hydra.utils.instantiate(cfg["workspace"])
    workspace.model.load_state_dict(ckpt["model_state_dict"], strict=False)
    workspace.model.to(device).eval()
    # The normalizer lives in the checkpoint, not the dataset dir, so evaluation cannot drift
    # from the statistics training actually used.
    if "normalizer_state_dict" in ckpt and workspace.train_dataset.normalizer is not None:
        workspace.train_dataset.normalizer.load_state_dict(ckpt["normalizer_state_dict"])
    norm = workspace.train_dataset.normalizer
    # fit_normalizer() ends with `self.normalizer.to(torch.device("cpu"))`, so a freshly loaded
    # normalizer holds CPU tensors while the batch is on cuda -> "Expected all tensors to be on
    # the same device". Move it once here rather than shuttling every batch back and forth.
    if norm is not None and hasattr(norm, "to"):
        norm.to(torch.device(device))
    return workspace, norm, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--memmap", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--exec_horizon", type=int, default=1,
                    help="execute N steps per prediction; 1 = replan every frame")
    ap.add_argument("--episodes", default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    phases, phases_src = _load_phases()
    meta, obs, act, front, wrist = _load_split(Path(a.memmap).resolve(), a.split)
    want = None if a.episodes is None else {int(x) for x in a.episodes.split(",") if x.strip()}

    print(f"loading {a.checkpoint}", flush=True)
    workspace, normalizer, cfg = build_policy(a.checkpoint, a.device)
    policy = workspace.model
    img_keys = list(cfg["workspace"]["train_dataset"]["image_keys"])
    print(f"  image keys: {img_keys}  exec_horizon={a.exec_horizon}  split={a.split}", flush=True)
    if wrist is None and any("wrist" in k for k in img_keys):
        raise SystemExit("policy expects a wrist camera but the memmap split has none")

    def prep(arr, r):
        x = torch.from_numpy(np.asarray(arr[r])).to(a.device).float().permute(2, 0, 1) / 255.0
        return F.interpolate(x[None], size=(256, 256), mode="bilinear", align_corners=False)

    episodes = []
    for ep_i, tr in enumerate(meta["trajectories"]):
        if want is not None and ep_i not in want:
            continue
        s, L = tr["start_idx"], tr["length"]
        policy.reset()                      # REQUIRED: clears trajectory history between episodes
        preds, gts, committed = [], [], None
        with torch.no_grad():
            for t in range(L):
                r = s + t
                if t % a.exec_horizon == 0:
                    batch = {"episode_idx": torch.tensor([ep_i], device=a.device)}
                    for k in img_keys:
                        src = front if "front" in k else wrist
                        batch[k] = prep(src, r)[None]            # (1,1,3,256,256)
                    batch["robot0_8d"] = torch.from_numpy(np.asarray(obs[r]))[None, None].to(a.device).float()
                    nb = normalizer.normalize(batch)
                    out = policy.predict_action(nb)
                    key = next(k for k in out if k.startswith("action0_"))
                    committed = normalizer.unnormalize({key: out[key]})[key][0].cpu().numpy()
                preds.append(committed[min(t % a.exec_horizon, len(committed) - 1)])
                gts.append(np.asarray(act[r]))
        P, G = np.asarray(preds), np.asarray(gts)
        err = np.abs(P - G)
        episodes.append({"episode": ep_i, "mae": float(err.mean()),
                         "mae_per_dim": err.mean(0).round(6).tolist(), "frames": int(L)})
        print(f"  episode {ep_i:3d}  frames {L:4d}  MAE {err.mean():.6f}", flush=True)

    if not episodes:
        raise SystemExit("no episodes evaluated")
    payload = {"checkpoint": a.checkpoint, "split": a.split, "exec_horizon": a.exec_horizon,
               "phases_source": phases_src, "episodes": episodes,
               "overall_mae": float(np.mean([e["mae"] for e in episodes]))}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(payload, open(a.out, "w"), indent=1)
    print(f"\noverall MAE {payload['overall_mae']:.6f} over {len(episodes)} episodes -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
