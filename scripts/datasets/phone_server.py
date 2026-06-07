import numpy as np
import time
import os
import glob
import bisect
import torch
import cv2
from datetime import datetime
from tqdm import tqdm

'''
PhoneServerDataset — 实时 WebSocket 模式专用数据集读取器。
不从磁盘读文件，而是从 server2tracker_queue 中取实时帧和 IMU 数据。

数据格式:
  server2tracker_queue item = {
      'rgb':       np.ndarray (H, W, 3) BGR,
      'timestamp': str,  EXIF 时间戳,
      'imu':       list of [timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z],
  }
'''

class PhoneDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.h_resize, self.w_resize = int(cfg['frontend']['image_size'][0]), int(cfg['frontend']['image_size'][1])
        # Load extrinsic/intrinsic parameters.
        self.c2i       = np.eye(4)
        self.intrinsic = None
        self.rgbinfo_dict = {"timestamp": []}
        # 存储从 server2tracker_queue 收到的 IMU 数据
        self.all_imu_buffer = []  # list of [timestamp, ax, ay, az, gx, gy, gz]

    def __len__(self):
        return 1000000

    def convert_to_unix_timestamp(self, line_str):
        # Input: 2011-09-26 13:02:25.446243840
        date_str, raw_time_str = line_str.split(' ')
        time_str, mm_second = raw_time_str.split('.')
        mm_second = '.' + mm_second
        datetime_str = f"{date_str} {time_str}"
        dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
        timestamp = time.mktime(dt.timetuple()) + dt.microsecond / 1e6
        timestamp += float(mm_second)
        return timestamp

    def preload_camtimestamp(self):
        return np.array(self.rgbinfo_dict['timestamp']).reshape(-1, 1)

    def preload_imu(self):
        if len(self.all_imu_buffer) == 0:
            # 还没有 IMU 数据，返回和图像帧数匹配的全零数组
            return np.zeros((len(self.rgbinfo_dict['timestamp']), 7))
        # 返回累积的 IMU 数据: (N, 7) = [timestamp, ax, ay, az, gx, gy, gz]
        imu_arr = np.array(self.all_imu_buffer)[:, :7]  # 确保7列
        return imu_arr

    def load_rgb(self, rgb, idx):
        resized_h, resized_w = int(self.cfg['frontend']['image_size'][0]), int(self.cfg['frontend']['image_size'][1])

        rgb_raw = rgb["rgb"]
        timestamp = rgb["timestamp"]

        # 提取 IMU 数据并存储
        if 'imu' in rgb and len(rgb['imu']) > 0:
            self.all_imu_buffer.extend(rgb['imu'])

        # The mobile image stream orientation differs from the previous takePicture path.
        # Rotate into portrait and avoid extra flips so the rendered image matches
        # the user-facing orientation on the phone.
        rgb_raw = cv2.rotate(rgb_raw, cv2.ROTATE_90_CLOCKWISE)
        rgb = (torch.tensor(cv2.resize(rgb_raw, (resized_w, resized_h)))[...,[2,1,0]]).permute(2,0,1).unsqueeze(0).to(self.cfg['device']['tracker'])
        u_scale, v_scale = resized_h/self.cfg['intrinsic']['H'], resized_w/self.cfg['intrinsic']['W']
        intrinsic = torch.tensor([self.cfg['intrinsic']['fv']*v_scale, self.cfg['intrinsic']['fu']*u_scale, \
                                self.cfg['intrinsic']['cv']*v_scale, self.cfg['intrinsic']['cu']*u_scale], dtype=torch.float32, device=self.cfg['device']['tracker'])
        data_packet = {}
        data_packet['timestamp'] = idx # float
        data_packet['rgb']       = rgb                                 # (1, 3, H, W)
        data_packet['intrinsic'] = intrinsic                           # (4, )
        self.rgbinfo_dict['timestamp'].append(timestamp)
        return data_packet


def get_dataset(config):
    return PhoneDataset(config)
