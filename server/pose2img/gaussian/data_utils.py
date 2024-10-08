import torch
from lietorch import SE3
import numpy as np

import cv2
import os

from torch.autograd import Variable

# Weilai.
# COV_TIMES = 200
# MAX_COV   = None

# nuScenes&KITTI
# COV_TIMES = 200
MAX_COV = None


# KITTI360
# COV_TIMES = 1000

# Waymo
# COV_TIMES = 5000


# Replica
# COV_TIMES = 10000
# MAX_COV = None

# Wangu
COV_TIMES = 7000
MAX_COV = None


# 处理数据
def process_viz_dict_vanilla(viz_dict, mapper_device):
    
    viz_idx        = viz_dict["viz_idx"]
    cam0_T_world   = viz_dict["cam0_poses"]
    idepths_up     = viz_dict["cam0_idepths_up"]
    depths_cov_up  = viz_dict["cam0_depths_cov_up"]
    calibs         = viz_dict["calibs"]
    images         = viz_dict["cam0_images"]
    
    cam0_T_world = SE3(cam0_T_world).matrix().contiguous()
    depths = (1.0 / idepths_up[..., None])
    depths_cov = depths_cov_up[..., None] # (N, 344, 616, 1)
    
    # TTD 2024/04/27
    N_frames = depths.shape[0]
    cov_median = torch.tensor(np.median(depths_cov.reshape(N_frames, -1), axis=1)[:, None, None, None], device=depths.device) # (N, 1, 1, 1)
    # zero_mask = torch.bitwise_or(depths_cov>DEPTH_COV_MIN, depths_cov>DEPTH_COV_MEDIAN_MIN*cov_median)
    zero_mask = depths_cov>(cov_median*COV_TIMES)
    # TTD 2024/05/23
    depths[zero_mask] = 0
    if MAX_COV is not None:
        depths[depths_cov>MAX_COV] = 0
    # depths[depths>25] = 0
    
    # TTD 2024/04/26
    processed_dict = {}
    processed_dict['poses'] = cam0_T_world.to(mapper_device) # (N, 4, 4)
    processed_dict['images'] = (torch.tensor(images) / 255.0).to(mapper_device).permute(0, 2, 3, 1) # (N, 344, 616, 3)
    processed_dict['depths'] = depths.to(mapper_device) # (N, 344, 616, 1)
    
    # TTD 2024/08/20
    sky_mask = (processed_dict['images'].sum(dim=-1, keepdim=True) == 0)
    processed_dict['depths'][sky_mask] = 0
    
    processed_dict['depths_cov'] = depths_cov.to(mapper_device) # (N, 344, 616, 1) 
    # processed_dict['abs_frame_idx_list'] = [viz_dict['kf_idx_to_f_idx'][kf_idx] for kf_idx in viz_idx.tolist()]
    camera_model = calibs[0].camera_model
    processed_dict['intrinsic'] = {'fv': camera_model[0], 'fu': camera_model[1], 'cv': camera_model[2], 'cu': camera_model[3], 
                                   'H': depths.shape[1], 'W': depths.shape[2]}
    processed_dict['pixel_mask'] = torch.ones_like(processed_dict['depths'].squeeze(-1), dtype=torch.bool)
    processed_dict['viz_out_idx_to_f_idx'] = viz_dict['viz_out_idx_to_f_idx']
    
    # TTD 2024/04/25
    for idx in range(processed_dict['poses'].shape[0]):
        processed_dict['poses'][idx] = processed_dict['poses'][idx].inverse()
    
    return processed_dict # Attention, 这里应该是c2w的pose;


def process_viz_dict(viz_dict, mapper_device):
    processed_dict = process_viz_dict_vanilla(viz_dict, mapper_device)
    return processed_dict

def process_viz_dict_sky(viz_dict, mapper_device, dataset_class=None):
    processed_dict = process_viz_dict_vanilla(viz_dict, mapper_device)

    f_idx_list = processed_dict['viz_out_idx_to_f_idx'].tolist()

    sky_image_all = []
    for f_idx in f_idx_list:
        sky_image = dataset_class.get_sky_image(f_idx) # (H, W, 3)
        sky_image_all.append(sky_image)
    sky_image_all = torch.tensor(np.stack(sky_image_all), device=processed_dict['images'].device, dtype=torch.float32)
    
    processed_dict['sky_images'] = sky_image_all # (N, H, W, 3)
    
    return processed_dict

def normalize_imgs(images):
    img_normalized = images[:,:,:3, ...] / 255.0 # Drop alpha channel
    mean = torch.as_tensor([0.485, 0.456, 0.406], device=images.device)[:, None, None]
    stdv = torch.as_tensor([0.229, 0.224, 0.225], device=images.device)[:, None, None]
    mean = img_normalized.mean(dim=(3,4), keepdim=True)
    stdv = img_normalized.std(dim=(3,4), keepdim=True)
    img_normalized = img_normalized.sub_(mean).div_(stdv)
    return img_normalized