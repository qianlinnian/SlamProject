'''
Customize ADC(Adaptive Densify Control) for Gaussian model.
Version: 0.0
Date: 2021-07-04
Description: Vanilla 2DGS's policy.(with new clone opacity)
'''
from gaussian.gaussian_base import GaussianBase
import torch
import torch.nn as nn
from gaussian.gaussian_utils import distCUDA2, get_pointcloud, get_split_properties
from gaussian.general_utils import inverse_sigmoid
from gaussian.wandb_utils import Wandber
from torch.autograd import Variable
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gaussian.cameras import get_camera
from gaussian.sky_utils import SkyModel
# TTD 2024/07/12
import matplotlib.pyplot as plt
import numpy as np
import copy

class GaussianModel(GaussianBase):
    def __init__(self, cfg):
        super(GaussianModel, self).__init__(cfg)
        self.wandber = Wandber(cfg, self.cfg['output']['save_dir'].split('/')[-1])
        self.global_c2w = {}
        self.time_idx   = 0

    def init_first_frame(self, batch):
        '''
        (1) Reset self.tfer.
        (2) Get init pointcloud and relative attributes.
        '''
        depths = batch["depths"] # (N, 344, 616, 1)
        images          = batch["images"] # (N, 344, 616, 4)
        poses           = batch["poses"] # (N, 4, 4)
        depths_cov      = batch["depths_cov"] # (N, 344, 616, 1) 
        
        # (1) Reset self.tfer.
        self.tfer.H = batch['intrinsic']['H']
        self.tfer.W = batch['intrinsic']['W']
        self.tfer.fu, self.tfer.cu = batch['intrinsic']['fu'], batch['intrinsic']['cu']
        self.tfer.fv, self.tfer.cv = batch['intrinsic']['fv'], batch['intrinsic']['cv']
        
        # (2) Accumulate Point Cloud.
        pc_world_list = []
        pc_color_list = []
        pc_rots_list = []
        for idx in range(depths.shape[0]):
            pose = poses[idx] # (4, 4)
            depth = depths[idx] # (H, W)
            rgb = images[idx] # (H, W, 3)
            xyz, rgb, q = get_pointcloud(self.tfer, pose, rgb.permute(2, 0, 1), depth.permute(2, 0, 1), None, 10000)
            pc_world_list.append(xyz)
            pc_color_list.append(rgb)
            pc_rots_list.append(q)

        # (3) Set self.history_list.
        self.history_list = batch['viz_out_idx_to_f_idx'].tolist()

        pc_world = torch.cat(pc_world_list, dim=0)# (N, 3)
        pc_world_color = torch.cat(pc_color_list, dim=0) # (N, 3)
        pc_world_rots  = torch.cat(pc_rots_list, dim=0) # (N, 4)
                
        dist2 = torch.clamp_min(distCUDA2(pc_world), 0.0000001)
        scales = torch.log(1.0 * torch.sqrt(dist2))[..., None].repeat(1, 2) # (N, 2)
        opacities = inverse_sigmoid(0.1 * torch.ones((pc_world.shape[0], 1), dtype=torch.float, device=self.device))

        self._xyz      = nn.Parameter(pc_world.contiguous().requires_grad_(True))
        self._rgb      = nn.Parameter(pc_world_color.contiguous().requires_grad_(True))
        self._scaling  = nn.Parameter(scales.contiguous().requires_grad_(True))
        self._rotation = nn.Parameter(pc_world_rots.contiguous().requires_grad_(True))
        self._opacity  = nn.Parameter(opacities.contiguous().requires_grad_(True))
        self._local_scores  = torch.zeros((pc_world.shape[0], 2), dtype=torch.float32, device=self.device)
        self._global_scores = torch.zeros((pc_world.shape[0], 2), dtype=torch.float32, device=self.device)
        self._stable_mask   = torch.zeros(pc_world.shape[0], dtype=torch.bool, device=self.device)

        if self.cfg['use_sky']:
            self.sky_model = SkyModel(self)
            self.sky_model.init_first_frame(batch)
    
    def add_new_frame(self, new_added_frame):
        new_added_pose = new_added_frame['pose'] # (4, 4)
        new_added_depth = new_added_frame['depth'] # (H, W, 1)
        new_added_color = new_added_frame['image'] # (H, W, 3)
        intrinsic_dict  = new_added_frame['intrinsic']
        new_added_c2w = new_added_pose # (4, 4)
        new_added_w2c = torch.inverse(new_added_c2w)
        with torch.no_grad():
            # Render Accumulation.
            rets = self.render(new_added_w2c, intrinsic_dict)
            pred_rgb   = rets['rgb']   # (3, H, W)
            pred_depth = rets['depth'] # (1, H, W)
            radii      = rets['radii'] # (1, H, W)

            # Delete pixels with large rgb error and in 1.5*gt_depth range.
            res_rgb = torch.abs(pred_rgb - new_added_color.permute(2, 0, 1)).sum(axis=0) # (H, W)
            loss_threshold   = 0.25
            # delete_pixelmask = torch.bitwise_and((pred_depth.squeeze(0) < 1.5 * new_added_depth.squeeze(-1)), (res_rgb > loss_threshold)) # 
            # Dangerous Option.
            delete_pixelmask = torch.bitwise_and((pred_depth.squeeze(0) < 2.5 * new_added_depth.squeeze(-1)), (res_rgb > 0.4))

            proj_uv = self.tfer.transform(self.get_property('_xyz'), 'world', 'pixel', pose=new_added_c2w) # (P, 3), P = validdepth_mask.sum()
            visible_gaussianmask = (proj_uv[:, 0] > 0) & (proj_uv[:, 0] < self.tfer.H-1) & (proj_uv[:, 1] > 0) & (proj_uv[:, 1] < self.tfer.W-1) & (proj_uv[:, 2] > 0.01) # (P)
            
            delete_gaussianmask  = torch.zeros_like(visible_gaussianmask)
            delete_gaussianmask[visible_gaussianmask][delete_pixelmask[proj_uv[visible_gaussianmask,0].to(torch.int32), proj_uv[visible_gaussianmask,1].to(torch.int32)]] = True
            # delete_gaussianmask[visible_gaussianmask][proj_uv[visible_gaussianmask,2] > 1.5 * new_added_depth.squeeze(-1)[proj_uv[visible_gaussianmask,0].to(torch.int32), proj_uv[visible_gaussianmask,1].to(torch.int32)]] = False

            # Prune Gaussians have big radii.
            delete_gaussianmask[radii>15] = True
            
        new_dict = self.prune_tensors_from_optimizer(self.optimizer, delete_gaussianmask)
        self.update_properties(new_dict)
        self.update_records(mode="prune", prune_gaussianmask=delete_gaussianmask)

        with torch.no_grad():
            rets = self.render(new_added_w2c, intrinsic_dict)
            # Add Gaussians on area with "large rgb/depth error or have low accum".
            pred_accum  = rets['accum'] # (1, H, W)
            pred_depth  = rets['depth'] # (1, H, W)
            depth_error = torch.abs(pred_depth-new_added_depth.permute(2, 0, 1))
            rgb_error   = torch.abs(pred_rgb-new_added_color.permute(2, 0, 1)).sum(axis=0, keepdim=True)
            pred_accum[depth_error > 50*depth_error.median()] = 0.0
            pred_accum[rgb_error > 0.1] = 0.0

        # Get point cloud and concat it to GaussianModel.
        new_added_pc, new_added_pc_color, unnorm_rots = get_pointcloud(self.tfer, new_added_c2w, new_added_color.permute(2, 0, 1), new_added_depth.permute(2, 0, 1), pred_accum, 30000)
        num_pts = new_added_pc.shape[0]

        dist2 = torch.clamp_min(distCUDA2(new_added_pc), 0.0000001)
        log_scales = torch.log(1.0 * torch.sqrt(dist2))[..., None].repeat(1, 2)
        logit_opacities = inverse_sigmoid((0.8*torch.ones((num_pts, 1), device=new_added_pc.device))).to(torch.float)

        new_params = {
            '_xyz': new_added_pc,
            '_rgb': new_added_pc_color,
            '_scaling': log_scales,
            '_rotation': unnorm_rots,
            '_opacity': logit_opacities
        }
        
        self._xyz = torch.nn.Parameter(torch.cat((self._xyz, new_params['_xyz']), dim=0).requires_grad_(True))
        self._rgb = torch.nn.Parameter(torch.cat((self._rgb, new_params['_rgb']), dim=0).requires_grad_(True))
        self._scaling = torch.nn.Parameter(torch.cat((self._scaling, new_params['_scaling']), dim=0).requires_grad_(True))
        self._rotation = torch.nn.Parameter(torch.cat((self._rotation, new_params['_rotation']), dim=0).requires_grad_(True))
        self._opacity = torch.nn.Parameter(torch.cat((self._opacity, new_params['_opacity']), dim=0).requires_grad_(True))
        self.update_records(mode="densify", densify_gaussiannum=num_pts)
        
        if self.cfg['use_sky']:
            self.sky_model.add_new_frame(new_added_frame)
        
        self.setup_optimizer()
    
    def add_records(self, _current_scores):
        self._local_scores[:, 0]  += _current_scores[:, 0]
        self._global_scores[:, 0] += _current_scores[:, 0]
        largeerror_mask = _current_scores[:, 1] > self._local_scores[:, 1]
        self._local_scores[largeerror_mask, 1]  = _current_scores[largeerror_mask, 1]
        self._global_scores  = torch.clamp(self._global_scores, 0, 1e4)
    
    def update_records(self, mode=None, densify_gaussiannum=None, prune_gaussianmask=None):
        if mode == "densify":
            self._local_scores  = torch.cat((self._local_scores, torch.zeros((densify_gaussiannum, 2), dtype=torch.float32, device=self.device)), dim=0)
            self._global_scores = torch.cat((self._global_scores, torch.zeros((densify_gaussiannum, 2), dtype=torch.float32, device=self.device)), dim=0)
            self._stable_mask   = torch.cat((self._stable_mask, torch.zeros((densify_gaussiannum, ), dtype=torch.bool, device=self.device)), dim=0)        
        elif mode == "prune":
            self._local_scores  = self._local_scores[~prune_gaussianmask]
            self._global_scores = self._global_scores[~prune_gaussianmask]
            self._stable_mask   = self._stable_mask[~prune_gaussianmask.reshape(-1)]
        else:
            assert False, "Invalid mode."
    
    def stablemask_control(self, current_iter):
        if (current_iter == self.cfg['training_args']['iters'] - 1) and \
           (self.time_idx+1) % self.cfg['training_args']['num_keyframe'] == 0:
            # Unstable → Stable ClassA: Gaussians whose "_local_scores[:, 0]" have no change during last num_iters.
            unstable2stable_mask = (~self._stable_mask) & (self._local_scores[:,0] < 1e-4)
            self._stable_mask[unstable2stable_mask] = True
            self._local_scores *= 0.0
            # Unstable → Stable ClassB: Gaussians whose "_local_scores[:,  ]" > th and "_local_score[:, 1]" < th.
            # Stable → Unstable: Gaussians whose "_local_scores[:, 1]" becomes too large.

    def adaptive_densify_control(self, current_iter, batch):
        pass
        '''
        ERROR_THRESHOLD = 1e4 # 0.004
        # Split Gaussians: gaussian-level, where _local_scores[:, 1]/_local_scores[:, 0] > th.
        if (current_iter == self.cfg['training_args']['iters'] - 1) and \
           (self.time_idx+1) % 2 == 0:
            # avg_error_scores   = self._local_scores[:, 1] / (self._local_scores[:, 0]+1e-4)
            # split_gaussianmask = (avg_error_scores > 0.3) & (~self._stable_mask) & (self._local_scores[:, 0] > 0.1)
            split_gaussianmask = (self._local_scores[:, 1] > ERROR_THRESHOLD) & (~self._stable_mask)
            subgaussian_dict   = get_split_properties(self, split_gaussianmask)

            new_dict = self.prune_tensors_from_optimizer(self.optimizer, split_gaussianmask)
            self.update_properties(new_dict)
            self.update_records(mode="prune", prune_gaussianmask=split_gaussianmask)

            new_dict = self.cat_tensors_to_optimizer(self.optimizer, subgaussian_dict)
            self.update_properties(new_dict)
            self.update_records(mode="densify", densify_gaussiannum=subgaussian_dict['_xyz'].shape[0])
            # print("Split Gaussians: ", split_gaussianmask.sum().item())

        # Densify Gaussians: pixel-level, where RGB error or Depth error is large.
        '''
        
    # TODO: Iter over all history frames.
    def storage_control(self, current_iter, batch):
        if (current_iter == self.cfg['training_args']['iters'] - 1) and \
           (self.time_idx+1) % self.cfg['training_args']['num_keyframe'] == 0:
            # Rerender on whole keyframe list and prune unstable gaussians whose _local_scores[:, 0] < 1.0.
            temp_importance_scores = torch.zeros_like(self._local_scores[:, 0]) # (P, )
            intrinsic_dict = batch["intrinsic"]
            for kf_idx in range(batch["poses"].shape[0]):
                c2w, gt_rgb = batch["poses"][kf_idx], batch["images"][kf_idx].permute(2, 0, 1) # (4, 4), (3, H, W)
                pred_rgb = self.render(torch.linalg.inv(c2w), intrinsic_dict)['rgb']
                (torch.abs(pred_rgb-gt_rgb)[:, gt_rgb.sum(axis=0)>0]).mean().backward()
                temp_importance_scores += self._zeros.grad.detach()[:, 0]
                self.optimizer.zero_grad()
                self._zeros.grad.zero_()
            prune_gaussianmask = (temp_importance_scores > 0.1) & (~self._stable_mask) & (temp_importance_scores < 0.8)
            new_dict = self.prune_tensors_from_optimizer(self.optimizer, prune_gaussianmask)
            self.update_properties(new_dict)
            self.update_records(mode="prune", prune_gaussianmask=prune_gaussianmask)





