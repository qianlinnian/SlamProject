import numpy as np
import shutil
import torch
import os
from frontend import gtsam_compat
from frontend.dbaf import DBAFusion
from gaussian.gaussian_model import GaussianModel
from gaussian.vis_utils import save_ply, vis_map
import argparse
parser = argparse.ArgumentParser(description="Add config path.")
parser.add_argument("config")
args = parser.parse_args()
config_path = args.config
from gaussian.general_utils import load_config, get_name
config = load_config(config_path)
import importlib
get_dataset = importlib.import_module(config["dataset"]["module"]).get_dataset
from vings_utils.middleware_utils import judge_and_package, retrieve_to_tracker
from storage.storage_manage import StorageManager
from loop.loop_model import LoopModel

class Runner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset  = get_dataset(cfg)
        cfg['frontend']['c2i'] = self.dataset.c2i # (4, 4), ndarray
        self.tracker = DBAFusion(cfg)
        self.mapper = GaussianModel(cfg)
        
        # self.looper = LoopModel(cfg)

        if 'use_storage_manager' in cfg.keys() and cfg['use_storage_manager']:
            self.use_storage_manager = True
            self.storage_manager = StorageManager(cfg)
        else:
            self.use_storage_manager = False

    def run(self):
        # Load imu data.
        self.tracker.frontend.all_imu   = self.dataset.preload_imu()
        self.tracker.frontend.all_stamp = self.dataset.preload_camtimestamp()
        
        # Run Tracking.
        for idx in range(len(self.dataset)):
            data_packet = self.dataset[idx]
            # TTD 2024/10/18
            self.tracker.frontend.all_imu   = self.dataset.preload_imu()
            self.tracker.frontend.all_stamp = self.dataset.preload_camtimestamp()
            
            self.tracker.track(data_packet)
            torch.cuda.empty_cache()
            # Judge whether new keyframe is added and package keyframe dict.
            viz_out = judge_and_package(self.tracker, data_packet['intrinsic'])
            if viz_out is not None and (self.cfg['mode']=='vo' or self.tracker.video.imu_enabled):
                # Save and check?
                self.mapper.run(viz_out)
                
                if self.use_storage_manager and (idx+1) % 10 == 0:
                    self.storage_manager.run(self.tracker, self.mapper, viz_out)
                    torch.cuda.empty_cache()
                
                if self.cfg['use_vis'] and (idx+1) % 20 == 0:
                    vis_map(self.tracker, self.mapper)
            
            # if (idx == len(self.dataset) - 1) and self.mapper._xyz.shape[0] > 0:
            if ((idx+1) % 300 == 0 or (idx == len(self.dataset) - 1)) and self.mapper._xyz.shape[0] > 0:
                save_ply(self.mapper, idx, save_mode='3dgs')
                # save_ply(self.mapper, idx, save_mode='pth')
            

if __name__ == '__main__':
    config['output']['save_dir'] = os.path.join(config['output']['save_dir'], get_name(config))
    os.makedirs(config['output']['save_dir']+'/droid_c2w', exist_ok=True)
    os.makedirs(config['output']['save_dir']+'/rgbdnua', exist_ok=True)
    os.makedirs(config['output']['save_dir']+'/ply', exist_ok=True)
    if 'debug_mode' in list(config.keys()) and config['debug_mode']:
        os.makedirs(config['output']['save_dir']+'/debug_dict', exist_ok=True)
    shutil.copy(config_path, config['output']['save_dir']+'/config.yaml')
    
    # os.chdir('/data/wuke/workspace/MobileApp/3DGS_SLAM_mobile_app/server/pose2img/')
    # os.system('/home/wuke/anaconda3/envs/Mobile3DGS6/bin/python /data/wuke/workspace/MobileApp/3DGS_SLAM_mobile_app/server/pose2img/server.py')
    # os.chdir('/data/wuke/workspace/MobileApp/3DGS_SLAM_mobile_app/server/pose2img/')
    
    runner = Runner(config)
    torch.backends.cudnn.benchmark = True
    runner.run()

    