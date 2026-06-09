# run.py
import numpy as np
import shutil
import torch
from frontend import gtsam_compat
from lietorch import SE3
import os
from frontend.dbaf import DBAFusion
from gaussian.gaussian_model import GaussianModel
from gaussian.vis_utils import save_ply, vis_map, vis_bev
import argparse
parser = argparse.ArgumentParser(description="Add config path.")
parser.add_argument("config")
parser.add_argument("--prefix", default='')
args = parser.parse_args()
config_path = args.config
from gaussian.general_utils import load_config, get_name
config = load_config(config_path)
import importlib
get_dataset = importlib.import_module(config["dataset"]["module"]).get_dataset
from vings_utils.middleware_utils import judge_and_package, retrieve_to_tracker, datapacket_to_nerfslam
from storage.storage_manage import StorageManager
from loop.loop_model import LoopModel
if config.get('use_metric', False):
    from metric.metric_model import Metric_Model
import time
from tqdm import tqdm
if config['mode'] == 'vo_nerfslam': from frontend_vo.vio_slam import VioSLAM


class Runner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset  = get_dataset(cfg)
        cfg['frontend']['c2i'] = self.dataset.c2i # (4, 4), ndarray
        
        if self.cfg['mode'] == 'vio' or self.cfg['mode'] == 'vo':
            self.tracker = DBAFusion(cfg)
        elif self.cfg['mode'] == 'vo_nerfslam':     
            self.tracker = VioSLAM(cfg)
        else: assert False, "Error \"mode\" in config file."
        
        if 'phone' not in cfg['dataset']['module']: self.tracker.dataset_length = len(self.dataset)
        
        self.mapper = GaussianModel(cfg)
        
        self.looper = LoopModel(cfg) if cfg.get('use_loop', False) else None
        
        if 'use_metric' in cfg.keys() and cfg['use_metric']:
            self.metric_predictor = Metric_Model(cfg) 
        
        if 'use_storage_manager' in cfg.keys() and cfg['use_storage_manager']:
            self.use_storage_manager = True
            self.storage_manager = StorageManager(cfg)
            if cfg['dataset']['module'] != 'phone':
                self.storage_manager.dataset_length = self.dataset.rgbinfo_dict['timestamp'][-1] - self.dataset.rgbinfo_dict['timestamp'][0] 
        else:
            self.use_storage_manager = False

    def save_tracking_pose_only(self):
        video = self.tracker.frontend.video
        save_dir = self.cfg['output']['save_dir']

        def to_cpu(value):
            if torch.is_tensor(value):
                return value.detach().cpu()
            if hasattr(value, 'value'):
                return value.value
            return value

        tstamp_save = to_cpu(video.tstamp_save)
        poses_save = to_cpu(video.poses_save)
        payload = {
            'mode': self.cfg.get('mode'),
            'frontend': {
                'video': {
                    'tstamp_save': tstamp_save,
                    'poses_save': poses_save,
                    'poses': to_cpu(video.poses) if hasattr(video, 'poses') else None,
                    'counter': to_cpu(video.counter) if hasattr(video, 'counter') else None,
                }
            }
        }

        if hasattr(video, 'count_save'):
            payload['frontend']['video']['count_save'] = video.count_save
            payload['frontend']['video']['count_save_bias'] = getattr(video, 'count_save_bias', 0)

        finite_pose = torch.isfinite(poses_save).all(dim=1)
        finite_time = torch.isfinite(tstamp_save)
        valid = finite_pose & finite_time & (tstamp_save != 0)
        if hasattr(video, 'count_save'):
            save_limit = int(video.count_save) + int(getattr(video, 'count_save_bias', 0))
            if save_limit > 0:
                in_saved_range = torch.zeros_like(valid)
                in_saved_range[:min(save_limit, len(in_saved_range))] = True
                valid = valid & in_saved_range

        if valid.any():
            pose_device = self.cfg['device']['tracker']
            c2w = SE3(poses_save[valid].to(pose_device)).inv().matrix().detach().cpu()
            timestamps = tstamp_save[valid].cpu()
            translations = c2w[:, :3, 3]
            steps = torch.linalg.norm(translations[1:] - translations[:-1], dim=1)
            payload['frontend']['video']['c2w_save'] = c2w
            payload['trajectory_stats'] = {
                'valid_poses': int(valid.sum().item()),
                'first_timestamp': float(timestamps[0].item()),
                'last_timestamp': float(timestamps[-1].item()),
                'span_xyz': (translations.max(dim=0).values - translations.min(dim=0).values).tolist(),
                'final_translation': translations[-1].tolist(),
                'median_step': float(torch.median(steps).item()) if len(steps) else 0.0,
                'max_step': float(torch.max(steps).item()) if len(steps) else 0.0,
            }

            pose_dir = os.path.join(save_dir, 'droid_c2w')
            os.makedirs(pose_dir, exist_ok=True)
            for stamp, mat in zip(timestamps.tolist(), c2w.numpy()):
                np.savetxt(os.path.join(pose_dir, f'{stamp:.6f}.txt'), mat)

            with open(os.path.join(save_dir, 'trajectory_summary.txt'), 'w') as f:
                for key, value in payload['trajectory_stats'].items():
                    f.write(f'{key}: {value}\n')

        torch.save(payload, os.path.join(save_dir, 'tracking_pose_only.pt'))

    def run(self):
        # Load imu data.
        self.tracker.frontend.all_stamp = self.dataset.preload_camtimestamp()
        if self.cfg.get('mode') == 'vo':
            import numpy as np
            stamps = self.tracker.frontend.all_stamp.reshape(-1)
            dummy_imu = np.zeros((len(stamps) + 16, 7), dtype=np.float64)
            dt = float(np.median(np.diff(stamps))) if len(stamps) > 1 else 0.033333
            dummy_imu[:, 0] = stamps[0] + np.arange(len(dummy_imu)) * dt
            dummy_imu[:, 6] = 9.81
            self.tracker.frontend.all_imu = dummy_imu
        else:
            self.tracker.frontend.all_imu = self.dataset.preload_imu()
        
        mapper_run_times = 0
        
        # Run Tracking.
        for idx in tqdm(range(len(self.dataset))):
            
            data_packet = self.dataset[idx]
            
            if 'use_mobile' in self.cfg.keys() and self.cfg['use_mobile']:
                self.tracker.frontend.all_imu   = self.dataset.preload_imu()
                self.tracker.frontend.all_stamp = self.dataset.preload_camtimestamp()
            
            if 'use_metric' in self.cfg.keys() and self.cfg['use_metric']:
                if 'depth' not in data_packet.keys() or data_packet['depth'] is None:
                    data_packet['depth'] = self.metric_predictor.predict(data_packet['rgb'][0])
            
            if self.cfg.get('mode') != 'vo':
                self.tracker.frontend.all_imu   = self.dataset.preload_imu()
                self.tracker.frontend.all_stamp = self.dataset.preload_camtimestamp()
            
            # torch.set_grad_enabled(False)
            self.tracker.track(data_packet if not self.cfg['mode']=='vo_nerfslam' else datapacket_to_nerfslam(data_packet, idx))
            # torch.set_grad_enabled(True)
            
            if self.cfg.get('tracking_only', False):
                if idx == len(self.dataset) - 1 and hasattr(self.tracker, 'save_pt_ckpt'):
                    if self.cfg.get('save_tracking_pose_only', False):
                        self.save_tracking_pose_only()
                    else:
                        self.tracker.save_pt_ckpt(os.path.join(self.cfg['output']['save_dir'], 'tracker.pt'))
                torch.cuda.empty_cache()
                continue
            
            torch.cuda.empty_cache()
            # Judge whether new keyframe is added and package keyframe dict.
            viz_out = judge_and_package(self.tracker, data_packet['intrinsic'])
            
            tracking_ready = (
                self.cfg['mode'] in ['vo', 'vo_nerfslam']
                or self.tracker.video.imu_enabled
                or getattr(self.tracker.video, 'visual_only_init', False)
            )
            if viz_out is not None and tracking_ready:
                # Save and check.
                new_viz_out = self.mapper.run(viz_out, True)
                
                if 'use_loop' in list(self.cfg.keys()) and self.cfg['use_loop']:
                    if viz_out["global_kf_id"][-1] > 10 and viz_out["global_kf_id"][-1] % 3 == 0:
                        self.looper.run(self.mapper, self.tracker, viz_out, idx)

                if self.use_storage_manager and (idx+1) % 10 == 0:
                    self.storage_manager.run(self.tracker, self.mapper, viz_out)
                    torch.cuda.empty_cache()
                
                vis_interval = int(self.cfg.get('vis', {}).get('interval', 1))
                if self.cfg['use_vis'] and vis_interval > 0 and (idx+1) % vis_interval == 0:
                    if not self.cfg['use_storage_manager'] or self.storage_manager._xyz.shape[0]==0:
                        vis_map(self.tracker, self.mapper)
                        vis_bev(self.tracker, self.mapper) 
                    else:
                        self.storage_manager.vis_map_storage(self.tracker, self.mapper)    
                        self.storage_manager.vis_bev_storage(self.tracker, self.mapper)    
            
            if (idx == len(self.dataset) - 1) and self.mapper._xyz.shape[0] > 0:
            # if ((idx+1) % 100 == 0 or (idx == len(self.dataset) - 1)) and self.mapper._xyz.shape[0] > 0:
                save_ply(self.mapper, idx, save_mode='2dgs')
                if self.cfg.get('use_vis', False):
                    if not self.cfg['use_storage_manager'] or self.storage_manager._xyz.shape[0]==0:
                        vis_map(self.tracker, self.mapper)
                        vis_bev(self.tracker, self.mapper)
                    else:
                        self.storage_manager.vis_map_storage(self.tracker, self.mapper)
                        self.storage_manager.vis_bev_storage(self.tracker, self.mapper)
                # save_ply(self.mapper, idx, save_mode='pth')
            

if __name__ == '__main__':
    
    config['output']['save_dir'] = os.path.join(config['output']['save_dir'], get_name(config)+'-{}-'.format(config_path.split('/')[-1].strip('.yaml'))+args.prefix)
    os.makedirs(config['output']['save_dir']+'/droid_c2w', exist_ok=True)
    os.makedirs(config['output']['save_dir']+'/rgbdnua', exist_ok=True)
    os.makedirs(config['output']['save_dir']+'/ply', exist_ok=True)
    if 'debug_mode' in list(config.keys()) and config['debug_mode']:
        os.makedirs(config['output']['save_dir']+'/debug_dict', exist_ok=True)
    shutil.copy(config_path, config['output']['save_dir']+'/config.yaml')
    
    runner = Runner(config)
    torch.backends.cudnn.benchmark = True
    
    runner.run()
    
    