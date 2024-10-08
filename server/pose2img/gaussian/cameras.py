# Copy from 3DGS.
import math
import torch
from typing import NamedTuple


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def getProjectionMatrix(fovX, fovY, znear=0.01, zfar=100.0):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)
    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

class Camera(NamedTuple):
    world_view_transform: torch.tensor
    projection_matrix: torch.tensor
    full_proj_transform: torch.tensor
    camera_center: torch.tensor
    tanfovx: float
    tanfovy: float
    width: int
    height: int

def get_camera(w2c, intrinsic_dict):
    '''
    EXPLANATION
        tanfovx: W/(2*fv)
            proj: Seems like intrinsic matrix J.
                [[2fv/W, 0, 0, 0], 
                    [0, 2fu/H, 0, 0], 
                    [0, 0, 1, -0.01],
                    [0, 0, 1.0, 0]].T
        viewpoint_camera.world_view_transform: "w2c,4x4"
            viewpoint_camera.full_proj_transform: "w2c@proj, 4x4"
    '''
    width, height = intrinsic_dict['W'], intrinsic_dict['H']
    focal_length_y, focal_length_x = intrinsic_dict['fu'], intrinsic_dict['fv']
    FoVy = focal2fov(focal_length_y, height) # (atan(H/2fu))/2
    FoVx = focal2fov(focal_length_x, width) # (atan(H/2fv))/2

    # EXCELLENT OPTION.
    world_view_transform = w2c.T
    projection_matrix = getProjectionMatrix(fovX=FoVx, fovY=FoVy).transpose(0,1).cuda()
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]
    
    tanfovx = math.tan(FoVx * 0.5) # H/(2fu)
    tanfovy = math.tan(FoVy * 0.5) # W/(2fv)

    camera = Camera(world_view_transform, projection_matrix, full_proj_transform, camera_center, tanfovx, tanfovy, width, height)
    return camera




