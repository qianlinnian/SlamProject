# scipts/datasets/euroc.py
import os
import numpy as np
import torch
import cv2
from tqdm import tqdm

class EuRoCDataset:
    """
    EuRoC MAV Dataset loader for VINGS-Mono.
    Expects ASL format:
      <root>/
        cam0/
          data.csv        # timestamp [ns], filename
          data/           # PNG images
          sensor.yaml     # camera intrinsics
        imu0/
          data.csv        # timestamp [ns], gx, gy, gz, ax, ay, az
        state_groundtruth_estimate0/
          data.csv        # ground truth poses
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_dir = cfg['dataset']['root']
        self.rgb_strip   = cfg['dataset'].get('rgb_strip', 1)

        # Preload image list
        self.preload_rgbinfo()

        # EuRoC camera-to-IMU extrinsic (T_cam_imu)
        # cam0 to imu0 for V1_01_easy (from sensor.yaml)
        self.c2i = np.array([
            [ 0.0148655429818, -0.999880929698,  0.00414029679422, -0.0216401454975],
            [ 0.999557249008,   0.0149672133247,  0.025715529948,  -0.064676986768 ],
            [-0.0257744366974,  0.00375618835797, 0.999660727178,   0.00981073058949],
            [ 0.0,              0.0,              0.0,              1.0            ]
        ], dtype=np.float64)

        self.tqdm = tqdm(total=self.__len__())

    def __len__(self):
        return len(self.rgbinfo_dict['timestamp'])

    def preload_rgbinfo(self):
        """Load camera timestamps and image paths from cam0/data.csv"""
        cam_csv = os.path.join(self.dataset_dir, 'cam0', 'data.csv')
        data    = np.loadtxt(cam_csv, delimiter=',', dtype=str, skiprows=1)

        # EuRoC timestamps are in nanoseconds — convert to seconds
        timestamps = data[:, 0].astype(np.float64) / 1e9
        filenames  = data[:, 1]

        # Apply rgb_strip (subsample frames)
        timestamps = timestamps[::self.rgb_strip]
        filenames  = filenames[::self.rgb_strip]

        filepaths = [
            os.path.join(self.dataset_dir, 'cam0', 'data', fname.strip())
            for fname in filenames
        ]

        self.rgbinfo_dict = {
            'timestamp': timestamps.tolist(),
            'filepath':  filepaths
        }

    def preload_camtimestamp(self):
        """Return camera timestamps as (N, 1) float64 array — matches KITTI loader format"""
        return np.array(self.rgbinfo_dict['timestamp']).reshape(-1, 1)

    def preload_imu(self):
        """
        Load IMU data from imu0/data.csv.
        EuRoC IMU format: timestamp[ns], gx, gy, gz, ax, ay, az
        VINGS-Mono expects: timestamp[s], gx, gy, gz, ax, ay, az  (N, 7)
        """
        imu_csv = os.path.join(self.dataset_dir, 'imu0', 'data.csv')
        imu_raw = np.loadtxt(imu_csv, delimiter=',', skiprows=1)

        all_imu = np.zeros((imu_raw.shape[0], 7), dtype=np.float64)
        all_imu[:, 0]   = imu_raw[:, 0] / 1e9   # ns -> seconds
        all_imu[:, 1:4] = imu_raw[:, 1:4]        # gyro  gx, gy, gz  [rad/s]
        all_imu[:, 4:7] = imu_raw[:, 4:7]        # accel ax, ay, az  [m/s^2]

        return all_imu

    def __getitem__(self, idx):
        resized_h = int(self.cfg['frontend']['image_size'][0])
        resized_w = int(self.cfg['frontend']['image_size'][1])

        # Load and resize image
        rgb_raw = cv2.imread(self.rgbinfo_dict['filepath'][idx])
        if rgb_raw is None:
            raise FileNotFoundError(f"Image not found: {self.rgbinfo_dict['filepath'][idx]}")

        rgb = (
            torch.tensor(cv2.resize(rgb_raw, (resized_w, resized_h)))[..., [2, 1, 0]]
        ).permute(2, 0, 1).unsqueeze(0).to(self.cfg['device']['tracker'])

        # Scale intrinsics to resized resolution
        u_scale = resized_h / self.cfg['intrinsic']['H']
        v_scale = resized_w / self.cfg['intrinsic']['W']
        intrinsic = torch.tensor([
            self.cfg['intrinsic']['fv'] * v_scale,
            self.cfg['intrinsic']['fu'] * u_scale,
            self.cfg['intrinsic']['cv'] * v_scale,
            self.cfg['intrinsic']['cu'] * u_scale
        ], dtype=torch.float32, device=self.cfg['device']['tracker'])

        data_packet = {
            'timestamp': self.rgbinfo_dict['timestamp'][idx],  # float (seconds)
            'rgb':       rgb,                                   # (1, 3, H, W)
            'intrinsic': intrinsic                              # (4,)
        }

        self.tqdm.update(1)
        return data_packet

    def load_gt_dict(self):
        """
        Load ground truth from state_groundtruth_estimate0/data.csv
        Format: timestamp[ns], px, py, pz, qw, qx, qy, qz, vx, vy, vz, ...
        """
        gt_csv = os.path.join(
            self.dataset_dir, 'state_groundtruth_estimate0', 'data.csv'
        )
        gt_raw    = np.loadtxt(gt_csv, delimiter=',', skiprows=1)
        timestamps = gt_raw[:, 0] / 1e9   # ns -> seconds
        positions  = gt_raw[:, 1:4]        # px, py, pz
        quats      = gt_raw[:, 4:8]        # qw, qx, qy, qz

        return {
            'timestamps': timestamps,
            'positions':  positions,
            'quats':      quats
        }


def get_dataset(config):
    return EuRoCDataset(config)
