import torch
import torch.nn.functional as F
import pytorch3d.transforms as pt3dtrans


def depth_propagate_normal(pred_depth, tfer):
    # STEP 1 Transform depth into cam coord.
    pc_cam_uv = tfer.transform(pred_depth, "depth", "uv_xyzdepth")
    
    # STEP 2 Cross product to get normal.
    H, W = pred_depth.shape[0], pred_depth.shape[1]
    bottom_point = pc_cam_uv[2:H,   1:W-1, :] # (H-2, W-2, 3)
    top_point    = pc_cam_uv[0:H-2, 1:W-1, :] # (H-2, W-2, 3)
    right_point  = pc_cam_uv[1:H-1, 2:W,   :] # (H-2, W-2, 3)
    left_point   = pc_cam_uv[1:H-1, 0:W-2, :] # (H-2, W-2, 3)
    left_to_right = right_point - left_point
    bottom_to_top = top_point - bottom_point 
    xyz_normal = torch.cross(left_to_right, bottom_to_top, dim=-1)
    xyz_normal = torch.nn.functional.normalize(xyz_normal, p=2, dim=-1) # (H-2, W-2, 3)
    xyz_normal = F.pad(xyz_normal, pad = (0, 0, 1, 1, 1, 1), mode='constant') # (H, W, 3)
    return xyz_normal # (H, W, 3), (0~1), 相机系下的法向量

def quat_to_rotmat(rot):
    '''
    Input: (N, 4)
    Output: (N, 3, 3)
    这里的 wxyz 和 xyzw还真有点乱;
    '''
    rot = torch.nn.normalization(rot, dim=-1)
    w, x, y, z = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]
    r1 = 1.0 - 2.0 * (y * y + z * z)
    r2 = 2.0 * (x * y + w * z)
    r3 = 2.0 * (x * z - w * y)
    r4 = 2.0 * (x * y - w * z)
    r5 = 1.0 - 2.0 * (x * x + z * z)
    r6 = 2.0 * (y * z + w * x)
    r7 = 2.0 * (x * z + w * y)
    r8 = 2.0 * (y * z - w * x)
    r9 = 1.0 - 2.0 * (x * x + y * y)

    R = torch.stack((r1, r2, r3, r4, r5, r6, r7, r8, r9), dim=1).view(-1, 3, 3)
    return R


def q_to_normal(rotations, c2w):
    '''
    only used as visualize.
    '''
    rot = rotations / torch.norm(rotations, dim=-1, keepdim=True)
    w, x, y, z = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]
    
    r0 = 2.0 * (x * z - w * y)
    r1 = 2.0 * (y * z + w * x)
    r2 = 1.0 - 2.0 * (x * x + y * y)
    
    normals3D = torch.stack((r0, r1, r2)).T # (N, 3)
    # 变到相机系判断正负
    normals3D_cam = (c2w[:3, :3].T @ normals3D.T).T # (N, 3)
    normals3D[normals3D_cam[:, -1]>0] *= -1

    return normals3D

def normal_to_q(normals):
    # 这里的normal是depth propagate得到的，应该方向是对的, 注意这里的normals应该是世界系下的;
    a = normals[:, 0]
    b = normals[:, 1]
    c = normals[:, 2]
    # 很好这里直接假定x=y算了;
    # q的顺序是[w, x, y, z]哈;
    # x = -b * torch.sqrt(a**2+b**2)/2.0
    # y =  a * torch.sqrt(a**2+b**2)/2.0
    # z = torch.zeros_like(x)
    # w = c / 2.0
    x = c / 2.0
    y = -b * torch.sqrt(a**2+b**2)/2.0
    z =  a * torch.sqrt(a**2+b**2)/2.0
    w = torch.zeros_like(x)
    q = torch.stack((w, x, y, z)).T
    return q


def qR_toq(qr1, R):
    '''
    Implement R on q, then get q_new.
    '''
    qrR = pt3dtrans.matrix_to_quaternion(R)
    qr2 = pt3dtrans.quaternion_multiply(qrR, qr1)
    return qr2