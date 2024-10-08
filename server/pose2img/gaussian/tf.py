import torch

# CX = runner.dataset.intrinsic[0][2]
# CY = runner.dataset.intrinsic[1][2]
# FX = runner.dataset.intrinsic[0][0]
# FY = runner.dataset.intrinsic[1][1]
# xx = (nonzero_v - CX)/FX
# yy = (nonzero_u - CY)/FY
# pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
# pix_ones = torch.ones(pts_cam.shape[0], 1).cuda().float()
# pts4 = torch.cat((pts_cam, pix_ones), dim=1)
# c2w = torch.inverse(w2c)
# pc_world = (c2w @ pts4.T).T[:, :3]
    
class TFer:
    
    def __init__(self, cfg):
        self.cfg = cfg
        # self.fu, self.fv, self.cu, self.cv = cfg['intrinsic']['fu']/cfg['intrinsic']['resolution_scale'], cfg['intrinsic']['fv']/cfg['intrinsic']['resolution_scale'], \
        #                                     cfg['intrinsic']['cu']/cfg['intrinsic']['resolution_scale'], cfg['intrinsic']['cv']/cfg['intrinsic']['resolution_scale']
        self.fu, self.fv, self.cu, self.cv = cfg['intrinsic']['fu'], cfg['intrinsic']['fv'], \
                                            cfg['intrinsic']['cu'], cfg['intrinsic']['cv']
        self.H, self.W = cfg['intrinsic']['H'], cfg['intrinsic']['W']
        self.device = cfg['device']['mapper']
        
            
    def transform(self, x_raw, source=None, target=None, pose=None):
        '''
        'depth'和'pixel'的区别是这样的, depth意味着x_raw.shape=(N,W), pixel意味着x_raw.shape=(N,3)
        
        pixel_coord:
                    _ _ _ _ v
                   |
                   |
                   |u
        cam_coord:
                   /z
                  /
                 /_ _ _ _x  
                 |
                 |
                 |y
        world_coord:
                
        '''
        if source == "cam" and target == "world":
            '''
            x.shape = (..., 3)
            pose.shape = (4, 4)
            '''
            # x = copy.deepcopy(x_raw).to(pose.dtype)
            x = x_raw + 0.0
            pix_ones = torch.ones_like(x[..., 0]).unsqueeze(-1).to(self.device).float() # (..., 1)
            pts4 = torch.cat((x, pix_ones), dim=1)
            c2w = pose
            pts_world = (c2w @ pts4.T).T[:, :3] # (..., 3) 
            y = pts_world
        
        elif source == "pixel" and target == "cam":
            '''
            x.shape=(N,3) | (u, v, depth)
            '''
            x = x_raw + 0.0
            u, v, cam_z = x[...,0], x[...,1], x[...,2]
            cam_x = (v - self.cv)/self.fv
            cam_y = (u - self.cu)/self.fu
            pts_cam = torch.stack((cam_x*cam_z, cam_y*cam_z, cam_z), dim=-1) # 相机坐标系
            y = pts_cam
        
        elif source == "pixel" and target == "world":
            '''
            x.shape=(N, 3)
            '''
            # x = copy.deepcopy(x_raw)
            x = x_raw + 0.0
            x = self.transform(x, 'pixel', 'cam')
            y = self.transform(x, 'cam', 'world', pose=pose)
        
        elif source == "depth" and target == "world":
            '''
            x.shape=(H, W)
            '''
            # x = copy.deepcopy(x_raw)
            x = x_raw + 0.0
            x = self.transform(x, 'depth', 'cam')
            y = self.transform(x, 'cam', 'world', pose=pose)
            
        elif source == "depth" and target == "cam":
            '''
            x.shape=(H, W)
            '''
            x = x_raw + 0.0
            x = self.transform(x, 'depth', 'pixel')
            x = self.transform(x, 'pixel', 'cam')
            y = x
                                
        # -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -

        elif source == "cam" and target == "pixel":
            '''
            x.shape=(N, 3)
            '''
            x = x_raw + 0.0
            cam_x, cam_y, cam_z = x[..., 0], x[..., 1], x[..., 2]
            v = cam_x/cam_z*self.fv + self.cv
            u = cam_y/cam_z*self.fu + self.cu
            pts_pixel = torch.stack((u, v, cam_z), dim=-1) # 相机坐标系
            y = pts_pixel # 返回的也是float，还是需要自己去转
        
        elif source == "world" and target == "cam":
            x = x_raw + 0.0
            pix_ones = torch.ones_like(x[..., 0]).unsqueeze(-1).to(x.device).float() # (..., 1)
            pts_world = torch.cat((x, pix_ones), dim=1)
            c2w = pose
            w2c = torch.linalg.inv(c2w)
            pts_cam = (w2c @ pts_world.T).T[:, :3] # (..., 3) 
            y = pts_cam
            
        elif source == "world" and target == "pixel":
            x = x_raw + 0.0
            x = self.transform(x, 'world', 'cam', pose=pose)
            y = self.transform(x, 'cam', 'pixel')            
        
        elif source == "depth" and target == "pixel":
            x = x_raw + 0.0
            uv = torch.nonzero(x)
            u, v = uv[..., 0], uv[..., 1]
            cam_z = x[u, v]
                        
            pts_pixel = torch.stack((u, v, cam_z), dim=-1) # 相机坐标系
            y = pts_pixel
        
        elif source == "depth" and target == "uv_xyzdepth":
            '''
            Input: (H, W)
            Output: (H, W, 3)
            '''
            x = x_raw + 0.0
            uv = torch.stack(torch.meshgrid(torch.arange(self.H), torch.arange(self.W))).permute(1, 2, 0).to(self.device).to(x.dtype) # (H, W, 2)
            pc_cam_yx =  (uv - torch.tensor([self.cu, self.cv], device=self.device, dtype=x.dtype).view(1, 1, 2)) / torch.tensor([self.fu, self.fv], device=self.device, dtype=x.dtype).view(1, 1, 2) * x.unsqueeze(-1) # (H, W, 2)
            pc_cam_xy = pc_cam_yx.flip(-1) # (H, W, 2)
            y = torch.cat((pc_cam_xy, x.unsqueeze(-1)), dim=-1) # (H, W, 3)
        
        
            
         
        return y