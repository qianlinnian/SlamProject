#dynamic_utils.py
import torch 

import os
import sys
sys.path.append('/data/wuke/workspace/FastSAM/')
from fastsam import FastSAM, FastSAMPrompt
import cv2
import yaml
sys.path.append('/data/wuke/workspace/Droid2DAcc/scripts/')

from gaussian.loss_utils import ssim_img

class DynamicModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.sam_model = FastSAM("/data/wuke/workspace/FastSAM/ckpts/FastSAM-x.pt")
        self.retrieve_dict = {'idx': [], 'dynamic_mask': []}
    
    def get_anns_raw(self, gt_rgb):
        '''
        If the enviroment not work (torch>2.0), you can try to infer before and loadthem here.
        gt_rgb.shape = (H, W, 3)
        '''
        everything_results = self.sam_model(gt_rgb.cpu().numpy()*255, device='cuda:0', retina_masks=True, imgsz=512, conf=0.4, iou=0.9)
        prompt_process = FastSAMPrompt(gt_rgb.numpy()*255, everything_results, device='cuda:0')
        raw_ann = prompt_process.everything_prompt().cpu()
        return raw_ann
    
    
    def generate_anns(self, data_type='custom'):
        '''
        Run this under nice-slam etc. envs.
        '''
        if data_type == 'custom':
            idx_list = list(map(lambda x: int(x.split('.')[0]), os.listdir(os.path.join(self.cfg['dataset']['folder'], 'rgb'))))
        elif data_type == 'kitti':
            idx_list = list(map(lambda x: int(x.split('.')[0]), os.listdir(os.path.join(self.cfg['dataset']['folder'], 'image_02', 'data'))))
        
        for idx in idx_list:
            if data_type == 'custom':
                rgb_filename = os.path.join(self.cfg['dataset']['folder'], 'rgb', str(idx).zfill(6)+'.png')
            elif data_type == 'kitti':
                rgb_filename = os.path.join(self.cfg['dataset']['folder'], 'image_02', 'data', str(idx).zfill(10)+'.png')
                
            rgb_uint8 = cv2.resize(cv2.imread(rgb_filename), (616, 344))
            ann_filepath = os.path.join(self.cfg['dataset']['folder'], 'sam_anns', str(idx).zfill(6)+'.pt')
            
            everything_results = self.sam_model(rgb_uint8, device='cuda:0', retina_masks=True, imgsz=512, conf=0.4, iou=0.9)
            prompt_process = FastSAMPrompt(rgb_uint8, everything_results, device='cuda:0')
            raw_ann = prompt_process.everything_prompt()
            torch.save(raw_ann, ann_filepath)
    
    
    def get_anns_load(self, idx):
        ann_filepath = os.path.join(self.cfg['dataset']['folder'], 'sam_anns', str(idx).zfill(6)+'.pt')
        raw_ann = torch.load(ann_filepath)
        return raw_ann
    
    def get_anns(self, gt_rgb, idx=None):
        raw_ann = self.get_anns_load(idx)
        return raw_ann
    

    def get_dynamic_mask(self, raw_ann, gt_rgb, pred_rgb):
        '''
        gt_rgb, pred_rgb.shape = (H, W, 3)
        raw_ann.shape = (K, H, W)
        '''
        
        rgb_l1_loss = torch.abs(pred_rgb-gt_rgb).mean(dim=-1)
        rgb_ssim_loss = 1-ssim_img(pred_rgb, gt_rgb).mean(dim=-1)
        multi_loss = rgb_l1_loss*rgb_ssim_loss
        multi_loss_90percent = torch.quantile(multi_loss, 0.9)
        multi_loss[multi_loss<multi_loss_90percent] = 0 # (H, W)
        multi_loss_mask = torch.zeros_like(multi_loss)
        multi_loss_mask[multi_loss>0] = 1
        
        high_loss_rate_list = []
        for idx in range(raw_ann.shape[0]):
            mask_pixels = multi_loss_mask[raw_ann[idx].to(torch.bool)]  # (X, )
            high_loss_rate = mask_pixels.sum()/mask_pixels.shape[0]
            
            masked_loss = multi_loss[raw_ann[idx].to(torch.bool)]
            
            # if high_loss_rate > 0.2 and masked_loss.mean()>0.01:
            if high_loss_rate > 0.2 and masked_loss.mean()>0.002:
                high_loss_rate_list.append(idx)

        ann = raw_ann[torch.tensor(high_loss_rate_list).to(torch.long)]
        
        dynamic_mask = torch.zeros_like(ann[0]).to(torch.bool)
        dynamic_mask[ann.sum(dim=0) > 0] = True
        
        return dynamic_mask


if __name__ == "__main__":
    
    def load_config(cfg_path):
        # Return a Dict.
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml.full_load(f)
        return cfg 
    
    # cfg_path = '/data/wuke/workspace/Droid2DAcc/configs/custom/wangu_dynamic.yaml'
    # cfg_path = '/data/wuke/workspace/Droid2DAcc/configs/custom/wangu_outdoor.yaml'
    cfg_path = '/data/wuke/workspace/Droid2DAcc/configs/kitti/kitti_sync_2011_09_26_drive0051.yaml'
    
    config = load_config(cfg_path)
    os.makedirs(os.path.join(config['dataset']['folder'], 'sam_anns'), exist_ok=True)
    dynamic_model = DynamicModel(config)
    dynamic_model.generate_anns('kitti')