import os
import torch
import numpy as np
from gaussian.gaussian_model import GaussianModel
from gaussian.vis_utils import load_ply
import cv2
from gaussian.general_utils import load_config, get_name


"""
save-dir|
        |ply|*.ply
            |intrinsic.yaml
"""


class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.mapper = GaussianModel(cfg)

        H, W = 344, 616
        raw_intrinsic_dict = cfg["intrinsic"]
        raw_H, raw_W = raw_intrinsic_dict["H"], raw_intrinsic_dict["W"]
        self.mapper.tfer.H, self.mapper.tfer.W = H, W
        self.mapper.tfer.fu, self.mapper.tfer.fv = (
            raw_intrinsic_dict["fu"] / raw_H * H,
            raw_intrinsic_dict["fv"] / raw_W * W,
        )
        self.mapper.tfer.cu, self.mapper.tfer.cv = (
            raw_intrinsic_dict["cu"] / raw_H * H,
            raw_intrinsic_dict["cv"] / raw_W * W,
        )
        self.mapper.initialized_state = True
        self.intrinsic_dict = {
            "fu": self.mapper.tfer.fu,
            "fv": self.mapper.tfer.fv,
            "cu": self.mapper.tfer.cu,
            "cv": self.mapper.tfer.cv,
            "H": self.mapper.tfer.H,
            "W": self.mapper.tfer.W,
        }

    def evaluate_one(self, c2w):
        with torch.no_grad():
            w2c = torch.inverse(c2w)
            rets = self.mapper.render(w2c, self.intrinsic_dict)
        return rets

    def load_model(self, model_path):
        property_dict_npy = load_ply(model_path)
        self.mapper._xyz = torch.tensor(
            property_dict_npy["_xyz"], device=self.mapper.device, dtype=torch.float32
        )
        self.mapper._rgb = torch.tensor(
            property_dict_npy["_rgb"], device=self.mapper.device, dtype=torch.float32
        )
        self.mapper._scaling = torch.tensor(
            property_dict_npy["_scaling"],
            device=self.mapper.device,
            dtype=torch.float32,
        )
        self.mapper._rotation = torch.tensor(
            property_dict_npy["_rotation"],
            device=self.mapper.device,
            dtype=torch.float32,
        )
        self.mapper._opacity = torch.tensor(
            property_dict_npy["_opacity"],
            device=self.mapper.device,
            dtype=torch.float32,
        )

    # TODO: Unfinisihed.
    def run(self, c2w, save_dir):
        rgb = (
            torch.clip(
                self.evaluate_one(c2w)["rgb"].permute(1, 2, 0).cpu(), 0.0, 1.0
            ).numpy()
            * 255.0
        ).astype(np.uint8)[..., [2, 1, 0]]
        cv2.imwrite(os.path.join(save_dir, "render.png"), rgb)
        return rgb

    def YawPitchRaw2RotationMatrix(yaw, pitch, raw):
        Yaw = np.array(
            [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]],
            dtype=np.float32,
        )
        Pitch = np.array(
            [
                [1, 0, 0],
                [0, np.cos(pitch), -np.sin(pitch)],
                [0, np.sin(pitch), np.cos(pitch)],
            ],
            dtype=np.float32,
        )
        Raw = np.array(
            [[np.cos(raw), -np.sin(raw), 0], [np.sin(raw), np.cos(raw), 0], [0, 0, 1]],
            dtype=np.float32,
        )
        return np.dot(np.dot(Yaw, Pitch), Raw)

    def RM_offset2C2W(RotationMatrix, offset):
        return torch.tensor(
            [
                [
                    RotationMatrix[0, 0],
                    RotationMatrix[0, 1],
                    RotationMatrix[0, 2],
                    offset[0],
                ],
                [
                    RotationMatrix[1, 0],
                    RotationMatrix[1, 1],
                    RotationMatrix[1, 2],
                    offset[1],
                ],
                [
                    RotationMatrix[2, 0],
                    RotationMatrix[2, 1],
                    RotationMatrix[2, 2],
                    offset[2],
                ],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
            device="cuda:0",
        )


def getEvaluator():
    CFG_PATH = "configs/kitti_2011_09_30_drive_0018.yaml"
    CKPT_PATH = "ckpts/idx=299_3dgs.ply"
    cfg = load_config(CFG_PATH)
    evaluator = Evaluator(cfg)
    evaluator.load_model(CKPT_PATH)
    return evaluator


if __name__ == "__main__":
    # KITTI
    CFG_PATH = "configs/kitti_2011_09_30_drive_0018.yaml"
    OUTPUT_DIR = "output/"
    CKPT_PATH = "ckpts/idx=299_3dgs.ply"

    # This is transformation matrix from camera to world coordinate.
    DEMO_C2W = torch.tensor(
        [
            [-0.0394, -0.3006, 0.9529, 0.4508],
            [-0.0102, -0.9535, -0.3012, 0.3173],
            [0.9992, -0.0216, 0.0345, 0.3940],
            [0.0000, 0.0000, 0.0000, 1.0000],
        ],
        device="cuda:0",
    )

    cfg = load_config(CFG_PATH)
    evaluator = Evaluator(cfg)
    evaluator.load_model(CKPT_PATH)
    print(Evaluator.YawPitchRaw2RotationMatrix(37 / 180 * 3.14, 37 / 180 * 3.14, 0))
    for i in range(0, 1000):
        C2W = Evaluator.RM_offset2C2W(
            Evaluator.YawPitchRaw2RotationMatrix(37 / 180 * 3.14, 37 / 180 * 3.14, 0),
            [
                0.4508 + i / 1000 * 0.4805405,
                0.3173 + i / 1000 * -0.60155356,
                0.3940 + i / 1000 * 0.63813335,
            ],
        )
        evaluator.run(C2W, OUTPUT_DIR)
