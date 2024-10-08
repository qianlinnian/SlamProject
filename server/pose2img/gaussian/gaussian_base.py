import torch
import torch.nn as nn
import random
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_surfel_rasterization import SparseGaussianAdam

from gaussian.cameras import get_camera
from gaussian.tf import TFer
from gaussian.gaussian_utils import distCUDA2, weighting_grad, get_gaussian_mask
from gaussian.general_utils import inverse_sigmoid
from abc import ABCMeta, abstractmethod
from gaussian.loss_utils import get_loss, get_pixel_mask
from gaussian.normal_utils import depth_propagate_normal
from gaussian.vis_utils import vis_rgbdnua, load_ply, calc_psnr


class GaussianBase:
    def __init__(self, cfg):
        self.cfg = cfg

        self.tfer = TFer(cfg)
        self.dtype = torch.float32
        self.device = self.tfer.device

        self._xyz = torch.empty(0)
        self._rgb = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._global_scores  = torch.empty(0) # Importance Score & Error Score.
        self._local_scores = torch.empty(0)   # Importance Score & Error Score during training iters.
        self._stable_mask  = torch.empty(0)
        
        self.activate_dict = {'_scaling': torch.exp,
                              '_opacity': torch.sigmoid,
                              '_rotation': torch.nn.functional.normalize,
                              'inv_scaling': torch.log,
                              'inv_opacity': inverse_sigmoid}

        self.initialized_state = False
    
    def setup_optimizer(self):
        cfg = self.cfg
        lr_args = cfg['training_args']['lr']
        l = [
            {'params': [self._xyz], 'lr': lr_args['_xyz_lr'], "name": "_xyz"},
            {'params': [self._rgb], 'lr': lr_args['_rgb_lr'], "name": "_rgb"},
            {'params': [self._opacity], 'lr': lr_args['_opacity_lr'], "name": "_opacity"},
            {'params': [self._scaling], 'lr': lr_args['_scaling_lr'], "name": "_scaling"},
            {'params': [self._rotation], 'lr': lr_args['_rotation_lr'], "name": "_rotation"}
        ]
        self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
    
    def get_property(self, name):
        if name == '_xyz': y = self._xyz
        elif name == '_opacity': y = self.activate_dict['_opacity'](self._opacity)
        elif name == '_rotation': y = self.activate_dict['_rotation'](self._rotation)
        elif name == '_scaling': y = self.activate_dict['_scaling'](self._scaling)
        elif name == '_rgb': y = self._rgb
        elif name == '_zeros': y = torch.zeros_like(self._xyz[:, :2]).contiguous().requires_grad_(True)
        else: raise ValueError("Invalid property name: {}".format(name))
        return y

    def cat_tensors_to_optimizer(self, optimizer, tensors_dict):
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                del optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_tensors_from_optimizer(self, optimizer, prune_mask):
        valid_mask = ~prune_mask
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][valid_mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][valid_mask]
                del optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][valid_mask].requires_grad_(True)))
                optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][valid_mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    @abstractmethod
    def init_first_frame(self, batch):
        pass

    @abstractmethod
    def add_new_frame(self, processed_dict):
        pass
    
    def judge_new_frame(self, processed_dict):
        new_id_list = processed_dict['viz_out_idx_to_f_idx'].tolist()
        history_list = self.history_list
        exist_list = [(item in history_list) for item in new_id_list]
        if all(exist_list):
            return False, None
        else:
            new_id = None
            for e_id in range(len(exist_list)):
                if not exist_list[e_id]:
                    new_id = e_id
                    break
            new_added_dict = {}
            self.history_list.append(new_id_list[new_id])
            new_added_dict['pose'] = processed_dict['poses'][new_id]
            new_added_dict['idx'] = new_id_list[new_id]
            new_added_dict['depth'] = processed_dict['depths'][new_id]
            new_added_dict['image'] = processed_dict['images'][new_id]
            new_added_dict['cov'] = processed_dict['depths_cov'][new_id]
            new_added_dict['intrinsic'] = processed_dict['intrinsic']
            return True, new_added_dict

    def render(self, w2c, intrinsic_dict, unopt_gaussian_mask = None):
        # Copy 2DGS.
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(self._xyz, dtype=self.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        
        # (1) Setup raster_settings.
        camera = get_camera(w2c, intrinsic_dict)
        
        pixel_mask = torch.ones(int(camera.height) * int(camera.width), dtype=torch.bool).cuda()
            
        raster_settings = GaussianRasterizationSettings(
            image_height=int(camera.height),
            image_width=int(camera.width),
            tanfovx=camera.tanfovx,
            tanfovy=camera.tanfovy,
            # bg=torch.zeros(3, device=self.device) if (self._xyz.grad is not None or random.random()>0.5) else torch.ones(3, device=self.device),
            bg = torch.zeros(3, device=self.device),
            scale_modifier=1.0,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=0, # Set None here will lead to TypeError.
            campos=camera.camera_center,
            prefiltered=False,
            debug=False,
            # stable_mask = self._stable_mask if unopt_gaussian_mask is None else unopt_gaussian_mask,
            pixel_mask = pixel_mask,
            # pipe.debug
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        # (2) Render.
        means3D        = self.get_property('_xyz') # (N, 3)
        means2D        = screenspace_points
        opacity        = self.get_property('_opacity') # (N, 1)
        scales         = self.get_property('_scaling') # (N, 3)
        rotations      = self.get_property('_rotation') # (N, 4)
        colors_precomp = self.get_property('_rgb') # (N, 3)
        self._zeros    = self.get_property('_zeros') # (N, 2)
        render_zeros   = self._zeros
        
        rendered_image, radii, allmap = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            scores    = render_zeros,
            cov3D_precomp = None
        )

        render_alpha = allmap[1:2]
        # get expected depth map
        render_depth_expected = allmap[0:1]
        render_depth_expected = (render_depth_expected / render_alpha)
        render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
        
        rets = {}
        # (depth, accum, normal, (median_depth), dist)
        rets['radii']   = radii # (1, H, W)
        rets['accum']   = render_alpha # (1, H, W)
        rets['rgb']     = rendered_image # (3, H, W)
        rets['depth']   = render_depth_expected # (1, H, W)
        # transform normal from view space to world space
        rets['normal']  = (allmap[2:5].permute(1,2,0) @ (w2c[:3,:3])).permute(2,0,1) # (3, H, W)
        rets['dist']    = allmap[6:7] # (1, H, W)
        rets['surf_normal'] = depth_propagate_normal(rets['depth'].squeeze(0), self.tfer).permute(2,0,1) # (3, H, W)
        rets['surf_normal'] = ( rets['surf_normal'] .permute(1,2,0) @ (w2c[:3,:3]) ).permute(2,0,1)
        # rets['n_contrib']   = allmap[7:8]
        
        return rets
    
    def update_properties(self, new_dict):
        '''
        Run this when add/prune new gaussians.
        You should update optimizer and get new_dict from optimizer.
        '''
        self._xyz, self._rgb, self._opacity, self._scaling, self._rotation = new_dict['_xyz'], new_dict['_rgb'], new_dict['_opacity'], new_dict['_scaling'], new_dict['_rotation']
    
    def train_once(self, batch, train_iters):
        # Get data from batch.
        abs_frame_idx_list = batch["viz_out_idx_to_f_idx"]    # (N, 1)
        poses              = batch["poses"]                   # (N, 4, 4)
        images             = batch["images"]                  # (N, 344, 616, 3)
        depths             = batch["depths"]                  # (N, 344, 616, 1)
        depths_cov         = batch["depths_cov"]              # (N, 344, 616, 1) 
        intrinsic_dict     = batch["intrinsic"]               # {'fu', 'fv', 'cu', 'cv', 'H', 'W'}
        pixel_masks        = batch["pixel_mask"]             # (N, 344, 616)
        
        for curr_iter in range(train_iters):
            
            self.wandber.log_time('forward_time')

            curr_id = random.randint(0, poses.shape[0]-1)
            c2w = poses[curr_id]
            w2c = torch.linalg.inv(c2w)
            
            pred_dict = self.render(w2c, intrinsic_dict, None)
            gt_dict = {'rgb': images[curr_id].permute(2,0,1), 'depth': depths[curr_id].permute(2,0,1), 'uncert': depths_cov[curr_id].permute(2,0,1), 'c2w': c2w}
            
            self.wandber.log_time('forward_time')
            
            if self.cfg['use_sky']:
                gt_dict['sky_rgb'] = batch["sky_images"][curr_id].permute(2,0,1) # (3, H, W)
                pred_dict_sky = self.sky_model.render(w2c, intrinsic_dict)
                pred_dict['rgb'] = self.sky_model.fuse_rgb(pred_dict, pred_dict_sky)
            
            self.wandber.log_time('backward_time')
            pred_dict['time_idx'] = self.time_idx
            total_loss = get_loss(self.cfg, pred_dict, gt_dict)
            
            total_loss.backward()
            self.wandber.log_time('backward_time')

            # (1) Record Importance Score & Error Score. (2) Multiply weights by accumulate scores to avoid forgetting problem.
            self.wandber.log_time('record_time')
            _current_scores = self._zeros.grad.detach()
            self.add_records(_current_scores)
            weighting_grad(self, _current_scores, self._global_scores)
            self.wandber.log_time('record_time')
            
            self.wandber.log_time('step_time')
            radii = pred_dict['radii']
            radii[self._stable_mask] = 0
            self.optimizer.step(radii > 0, radii.shape[0])
            self.optimizer.zero_grad()
            self.wandber.log_time('step_time')

            if self.cfg['use_sky']:
                radii_sky = pred_dict_sky['radii']
                self.sky_model.optimizer.step(radii_sky > 0, radii_sky.shape[0])
                self.sky_model.optimizer.zero_grad()

            # self.wandber.log_time('Time_PerIter')

            if curr_iter == train_iters - 1:
                gt_dict['pose'] = c2w
                gt_dict['abs_frame_idx_list'] = batch["viz_out_idx_to_f_idx"]
                frame_id = batch["viz_out_idx_to_f_idx"][curr_id]
                vis_rgbdnua(self.cfg, frame_id, pred_dict, gt_dict)
                self.wandber.log_once("num_of_gaussians", self._xyz.shape[0])
                self.wandber.log_once("psnr", calc_psnr(pred_dict['rgb'], gt_dict['rgb'], gt_dict['depth'].squeeze(0)>0).item())


            self.stablemask_control(curr_iter)
            # self.adaptive_densify_control(curr_iter, batch)
            self.storage_control(curr_iter, batch)
        self.time_idx += 1
    
    def run_only_mapping(self, processed_dict):
        if self.initialized_state:
            new_frame_added, new_added_dict = self.judge_new_frame(processed_dict)
            if new_frame_added:
                self.add_new_frame(new_added_dict)
                # Excellent Option.
            self.train_once(processed_dict, self.cfg['training_args']['iters'])
        else:
            self.init_first_frame(processed_dict)
            self.initialized_state = True
            self.setup_optimizer()
            self.train_once(processed_dict, self.cfg['training_args']['iters'])
    
    def load_ckpt(self, ckpt_path):
        property_dict_npy = load_ply(ckpt_path)

        self._xyz = nn.Parameter(torch.tensor(property_dict_npy['_xyz'], dtype=torch.float32, device=self.device).contiguous().requires_grad_(True))
        self._rgb = nn.Parameter(torch.tensor(property_dict_npy['_rgb'], dtype=torch.float32, device=self.device).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(property_dict_npy['_opacity'], dtype=torch.float32, device=self.device).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(property_dict_npy['_scaling'], dtype=torch.float32, device=self.device).contiguous().requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(property_dict_npy['_rotation'], dtype=torch.float32, device=self.device).contiguous().requires_grad_(True))

        self.setup_optimizer()
        # Please remember to set tfer before render.
        self.tfer.H, self.tfer.W = None, None
        self.tfer.cu, self.tfer.cv = None, None
        self.tfer.fu, self.tfer.fv = None, None
        
        self.initialized_state = True

    # TODO: Implement GaussianModel.run().
    def run(self, processed_dict):
        self.run_only_mapping(processed_dict)
        

