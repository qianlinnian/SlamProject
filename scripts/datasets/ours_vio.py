import os

import cv2
import numpy as np
import torch
from tqdm import tqdm


class OursVIODataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_dir = cfg["dataset"]["root"]
        self.rgb_dir = os.path.join(self.dataset_dir, "image_00", "data")
        self.preload_rgbinfo()
        self.c2i = np.loadtxt(os.path.join(self.dataset_dir, "metadata", "c2i.txt"))
        self.tqdm = tqdm(total=self.__len__())

    def __len__(self):
        return len(self.rgbinfo_dict["timestamp"])

    def preload_camtimestamp(self):
        camstamp = np.loadtxt(
            os.path.join(self.dataset_dir, "metadata", "camstamp.txt"),
            dtype=str,
        )
        if camstamp.ndim == 1:
            camstamp = camstamp.reshape(1, -1)
        return camstamp[:, 0].astype(np.float64).reshape(-1, 1)

    def preload_imu(self):
        imu = np.loadtxt(os.path.join(self.dataset_dir, "metadata", "imu.txt"))
        if imu.ndim == 1:
            imu = imu.reshape(1, -1)
        imu[:, 0] -= self.cfg["dataset"].get("imu_delay", 0.0)
        return imu

    def preload_rgbinfo(self):
        camstamp = np.loadtxt(
            os.path.join(self.dataset_dir, "metadata", "camstamp.txt"),
            dtype=str,
        )
        if camstamp.ndim == 1:
            camstamp = camstamp.reshape(1, -1)
        self.rgbinfo_dict = {
            "timestamp": camstamp[:, 0].astype(np.float64).tolist(),
            "filepath": [
                os.path.join(self.rgb_dir, name.strip())
                for name in camstamp[:, 1].tolist()
            ],
        }

    def __getitem__(self, idx):
        resized_h = int(self.cfg["frontend"]["image_size"][0])
        resized_w = int(self.cfg["frontend"]["image_size"][1])

        rgb_raw = cv2.imread(self.rgbinfo_dict["filepath"][idx])
        if rgb_raw is None:
            raise FileNotFoundError(self.rgbinfo_dict["filepath"][idx])

        rgb = cv2.resize(rgb_raw, (resized_w, resized_h))
        rgb = torch.tensor(rgb[..., [2, 1, 0]]).permute(2, 0, 1).unsqueeze(0)
        rgb = rgb.to(self.cfg["device"]["tracker"])

        u_scale = resized_h / self.cfg["intrinsic"]["H"]
        v_scale = resized_w / self.cfg["intrinsic"]["W"]
        intrinsic = torch.tensor(
            [
                self.cfg["intrinsic"]["fv"] * v_scale,
                self.cfg["intrinsic"]["fu"] * u_scale,
                self.cfg["intrinsic"]["cv"] * v_scale,
                self.cfg["intrinsic"]["cu"] * u_scale,
            ],
            dtype=torch.float32,
            device=self.cfg["device"]["tracker"],
        )

        self.tqdm.update(1)
        return {
            "timestamp": self.rgbinfo_dict["timestamp"][idx],
            "rgb": rgb,
            "intrinsic": intrinsic,
        }

    def load_gt_dict(self):
        pose_dir = os.path.join(self.dataset_dir, "pose")
        if not os.path.isdir(pose_dir):
            return None
        pose_files = os.listdir(pose_dir)
        timestamps = np.array([float(name.replace(".txt", "")) for name in pose_files])
        c2ws = np.array([np.loadtxt(os.path.join(pose_dir, name)) for name in pose_files])
        order = np.argsort(timestamps)
        return {"timestamps": timestamps[order], "c2ws": c2ws[order]}


def get_dataset(config):
    return OursVIODataset(config)
