import torch
from gaussian.general_utils import inverse_sigmoid
from gaussian.gaussian_utils import distCUDA2, get_pointcloud
from gaussian.normal_utils import normal_to_q
import torch.nn as nn
from diff_surfel_rasterization import SparseGaussianAdam
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gaussian.cameras import get_camera
from gaussian.normal_utils import qR_toq

class SkyModel:
    def __init__(self, gaussian_model):
        self.cfg = gaussian_model.cfg
        self.is_initialized = False
        self._xyz = torch.empty(0)
        self._rgb = torch.empty(0)
        self._scaling = torch.empty(0)
        self._opacity = torch.empty(0)
        self._rotation = torch.empty(0)
        self.sphere_radius = 10.0
        self.tfer = gaussian_model.tfer
        self.device = gaussian_model.device
        self.activate_dict = {'_scaling': torch.exp,
                              '_opacity': torch.sigmoid,
                              '_rotation': torch.nn.functional.normalize,
                              'inv_scaling': torch.log,
                              'inv_opacity': inverse_sigmoid}
    
    def get_property(self, name):
        if name == '_xyz': y = torch.nn.functional.normalize(self._xyz) * self.sphere_radius
        elif name == '_opacity': y = self.activate_dict['_opacity'](self._opacity)
        elif name == '_rotation': y = self.activate_dict['_rotation'](self._rotation)
        elif name == '_scaling': y = self.activate_dict['_scaling'](self._scaling) * self.sphere_radius
        elif name == '_rgb': y = self._rgb
        else: raise ValueError("Invalid property name: {}".format(name))
        return y
    
    def init_first_frame(self, batch, N_points=1000):
        '''
        只在第一帧的时候初始化
        '''

        images          = batch["images"] # (N, 344, 616, 4)
        poses           = batch["poses"] # (N, 4, 4)
        pc_world_list = []
        pc_color_list = []
        for idx in range(images.shape[0]):
            pose = poses[idx] # (4, 4)
            rgb = images[idx] # (H, W, 3)
            if 'sky_mask' not in batch.keys():
                sky_mask = rgb.sum(dim=-1) == 0 # (H, W)
            sky_depth = torch.ones_like(sky_mask, dtype=torch.float32).unsqueeze(0) # (1, H, W)
            sky_depth[:, ~sky_mask] = 0.0
            xyz, rgb, _ = get_pointcloud(self.tfer, pose, rgb.permute(2, 0, 1), sky_depth, None, N_points)
            xyz         = torch.nn.functional.normalize(xyz - pose[:3, -1].unsqueeze(0)) # (0.0~1.0)哈
            pc_world_list.append(xyz)
            pc_color_list.append(rgb)

        pc_world = torch.cat(pc_world_list, dim=0)# (N, 3)
        pc_world_color = torch.cat(pc_color_list, dim=0) # (N, 3)

        dist2 = torch.clamp_min(distCUDA2(pc_world), 0.0000001)
        scales = torch.log(1.0 * torch.sqrt(dist2))[..., None].repeat(1, 2) # (N, 2)
        opacities = inverse_sigmoid(0.1 * torch.ones((pc_world.shape[0], 1), dtype=torch.float, device=self.device))

        self._xyz      = nn.Parameter(pc_world.contiguous().requires_grad_(True))
        self._rgb      = nn.Parameter(pc_world_color.contiguous().requires_grad_(True))
        self._scaling  = nn.Parameter(scales.contiguous().requires_grad_(True))
        self._opacity  = nn.Parameter(opacities.contiguous().requires_grad_(True))
        self._rotation  = nn.Parameter(torch.randn((pc_world.shape[0], 4), device=self._xyz.device, dtype=torch.float32).contiguous().requires_grad_(True))

        self.setup_optimizer()

    def add_new_frame(self, new_added_frame, N_points=1000):
        
        new_added_pose = new_added_frame['pose'] # (4, 4)
        new_added_color = new_added_frame['image'] # (H, W, 3)
        new_added_c2w = new_added_pose # (4, 4)
        new_added_w2c = torch.inverse(new_added_c2w)
        intrinsic_dict  = new_added_frame['intrinsic']
        
        with torch.no_grad():
            # Render Accumulation.
            rets = self.render(new_added_w2c, intrinsic_dict)
            pred_accum = rets['accum'] # (1, H, W)
            if 'sky_mask' not in new_added_frame.keys():
                sky_mask = new_added_color.sum(dim=-1) == 0 # (H, W)
            sky_depth = torch.ones_like(sky_mask, dtype=torch.float32).unsqueeze(0) # (1, H, W)
            sky_depth[:, ~sky_mask] = 0.0
        
        new_added_pc, new_added_pc_color, _ = get_pointcloud(self.tfer, new_added_c2w, new_added_color.permute(2, 0, 1), sky_depth, pred_accum, N_points)
        new_added_pc = torch.nn.functional.normalize(new_added_pc - new_added_c2w[:3, -1].unsqueeze(0)) # (0.0~1.0)哈
        num_pts = new_added_pc.shape[0]
        if num_pts > 10:
            dist2 = torch.clamp_min(distCUDA2(new_added_pc), 0.0000001)
            log_scales = torch.log(1.0 * torch.sqrt(dist2))[..., None].repeat(1, 2)
            logit_opacities = inverse_sigmoid((0.5*torch.ones((num_pts, 1), device=new_added_pc.device))).to(torch.float)

            new_params = {
                '_xyz': new_added_pc,
                '_rgb': new_added_pc_color,
                '_scaling': log_scales,
                '_opacity': logit_opacities,
                '_rotation': torch.randn((num_pts, 4), device=self._xyz.device, dtype=torch.float32)
            }

            self._xyz = torch.nn.Parameter(torch.cat((self._xyz, new_params['_xyz']), dim=0).requires_grad_(True))
            self._rgb = torch.nn.Parameter(torch.cat((self._rgb, new_params['_rgb']), dim=0).requires_grad_(True))
            self._scaling = torch.nn.Parameter(torch.cat((self._scaling, new_params['_scaling']), dim=0).requires_grad_(True))
            self._opacity = torch.nn.Parameter(torch.cat((self._opacity, new_params['_opacity']), dim=0).requires_grad_(True))
            self._rotation = torch.nn.Parameter(torch.cat((self._rotation, new_params['_rotation']), dim=0).requires_grad_(True))

            self.setup_optimizer()
        
    def render(self, w2c, intrinsic_dict):
        # STEP 1 ：将 SphereGaussian变换到世界系;
        c2w = torch.linalg.inv(w2c)
        means3D_raw    = self.get_property('_xyz')
        scales         = self.get_property('_scaling')
        opacity        = self.get_property('_opacity')
        colors_precomp = self.get_property('_rgb') # (N, 3)
        rotations_raw  = self.get_property('_rotation') # (N, 4)
        
        # Be careful, don't change raw value in place.
        means3D        = means3D_raw + c2w[:3, -1].unsqueeze(0) # (N, 3)
        rotations      = qR_toq(rotations_raw, c2w[:3, :3])

        # STEP 2 : 渲染 (需要一个带mask的渲染器加快速度哈)
        screenspace_points = torch.zeros_like(self._xyz, dtype=torch.float32, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        camera = get_camera(w2c, intrinsic_dict)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(camera.height),
            image_width=int(camera.width),
            tanfovx=camera.tanfovx,
            tanfovy=camera.tanfovy,
            bg=torch.zeros(3, device=self.device),
            scale_modifier=1.0,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=0, # Set None here will lead to TypeError.
            campos=camera.camera_center,
            prefiltered=False,
            debug=False,
            # pipe.debug
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means2D        = screenspace_points
        rendered_image, radii, allmap = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = None
        )

        # STEP 3 : 拼RGB, 其他的不动;
        rets = {}
        rets['radii']     = radii
        rets['rgb']     = rendered_image # (3, H, W)
        rets['accum']   = allmap[1:2] # (1, H, W)
        return rets

    def setup_optimizer(self):
        cfg = self.cfg
        lr_args = cfg['training_args']['lr']
        l = [
            {'params': [self._xyz], 'lr': lr_args['_xyz_lr']/1.0, "name": "_xyz"},
            {'params': [self._rgb], 'lr': lr_args['_rgb_lr'], "name": "_rgb"},
            {'params': [self._opacity], 'lr': lr_args['_opacity_lr'], "name": "_opacity"},
            {'params': [self._scaling], 'lr': lr_args['_scaling_lr']/1.0, "name": "_scaling"},
            {'params': [self._rotation], 'lr': lr_args['_rotation_lr'], "name": "_rotation"}
        ]
        self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)

    def fuse_rgb(self, pred_dict, pred_dict_sky):
        pred_rgb          = pred_dict['rgb'] # (3, H, W)
        pred_accum_nograd = pred_dict['accum'].detach() # (1, H, W)
        pred_rgb_sky   = pred_dict_sky['rgb'] # (3, H, W)
        pred_accum_sky = pred_dict_sky['accum'] # (1, H, W)
        fused_rgb = pred_rgb * pred_accum_nograd + (1 - pred_accum_nograd) * pred_rgb_sky
        return fused_rgb
    
