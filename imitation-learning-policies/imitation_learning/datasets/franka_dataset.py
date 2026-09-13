"""Dataset for our single-arm Franka data (JSC), converted from the LeRobot memmap.

Mirrors robomimic_dataset.py, but the composition is much simpler: we are in JOINT space, so
there is no rotation representation to convert. robomimic turns eef_pos + quat + gripper_qpos
into a 10-d vector via rot_6d; we concatenate 7 joint positions with 1 gripper width into 8 d,
and the action is already an absolute 8-d joint target.

Zarr entries produced by jsc/memmap_to_zarr.py:
    joint_pos   (n, 7)  gripper_pos (n, 1)  abs_actions (n, 8)
    front_rgb / wrist_rgb  (n, 224, 224, 3) uint8
"""
from typing import Any

import numpy as np
import numpy.typing as npt

from imitation_learning.datasets.base_dataset import BaseDataset
from imitation_learning.datasets.episodic_dataset import EpisodicDataset
from imitation_learning.datasets.multi_traj_dataset import MultiTrajDataset


def _process_source_data(
    self: BaseDataset, data_dict: dict[str, npt.NDArray[Any]]
) -> dict[str, npt.NDArray[Any]]:
    # proprio: 7 joints + gripper -> 8 d. Literal entry names rather than robomimic's
    # prefix-splitting, which would look for "robot0_joint_pos" given an output named
    # "robot0_8d" and fail on our zarr.
    for name, entry_meta in self.output_data_meta.items():
        if entry_meta.name.endswith("8d") and not entry_meta.name.startswith("action"):
            length = entry_meta.length
            joints = data_dict["joint_pos"][-length:]
            gripper = data_dict["gripper_pos"][-length:]
            data_dict[name] = np.concatenate([joints, gripper], axis=-1)

    # action is already absolute 8-d joint targets; no conversion, just rename
    if "abs_actions" in data_dict:
        data_dict["action0_8d"] = data_dict.pop("abs_actions")

    return data_dict


class FrankaMultiTrajDataset(MultiTrajDataset, EpisodicDataset):
    pass


FrankaMultiTrajDataset._process_source_data = _process_source_data


class FrankaSingleTrajDataset(EpisodicDataset):
    pass


FrankaSingleTrajDataset._process_source_data = _process_source_data
