import torch
import pytz
from datetime import datetime
import yaml

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def get_name(cfg=None):
    beijing_tz = pytz.timezone('Asia/Shanghai')
    now = datetime.now(beijing_tz)
    current_month = now.month
    current_day = now.day
    current_hour = now.hour
    current_minute = now.minute
    if cfg is not None:
        formatted_string = f"{current_month:02d}-{current_day:02d}-{current_hour:02d}-{current_minute:02d}-{cfg['dataset']['module'].split('.')[-1]}"
    else:
        formatted_string = f"{current_month:02d}-{current_day:02d}-{current_hour:02d}-{current_minute:02d}"
    return formatted_string

def load_config(cfg_path):
    # Return a Dict.
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.full_load(f)
    normalize_config_paths(cfg)
    return cfg 


def normalize_config_paths(cfg):
    dataset_prefix_map = {
        '/data/wuke/DATA/2023/': '/root/autodl-tmp/VINGS-Mono/dataset/',
        '/data/wuke/DATA/2024/': '/root/autodl-tmp/VINGS-Mono/dataset/',
        '/Users/krushna/Downloads/': '/root/autodl-tmp/VINGS-Mono/dataset/',
    }
    repo_prefix_map = {
        '/data/wuke/workspace/VINGS-Mono/': '/root/autodl-tmp/VINGS-Mono/',
        '/data/wuke/workspace/September/VINGS-Mono/': '/root/autodl-tmp/VINGS-Mono/',
        '/Users/krushna/VINGS-Mono/': '/root/autodl-tmp/VINGS-Mono/',
        '/Users/krushna/Downloads/VINGS-Mono/': '/root/autodl-tmp/VINGS-Mono/',
    }
    exact_path_map = {
        '/data/wuke/workspace/LightGlue-ONNX/weights/': '/root/autodl-tmp/VINGS-Mono/ckpts/lightglue/',
    }

    def rewrite_path(value, prefix_map):
        if not isinstance(value, str):
            return value
        if value in exact_path_map:
            return exact_path_map[value]
        for old_prefix, new_prefix in prefix_map.items():
            if value.startswith(old_prefix):
                return value.replace(old_prefix, new_prefix, 1)
        return value

    dataset_cfg = cfg.get('dataset', {})
    if 'root' in dataset_cfg:
        dataset_cfg['root'] = rewrite_path(dataset_cfg['root'], dataset_prefix_map)

    output_cfg = cfg.get('output', {})
    if 'save_dir' in output_cfg:
        output_cfg['save_dir'] = rewrite_path(output_cfg['save_dir'], repo_prefix_map)

    frontend_cfg = cfg.get('frontend', {})
    if 'weight' in frontend_cfg:
        frontend_cfg['weight'] = rewrite_path(frontend_cfg['weight'], repo_prefix_map)

    looper_cfg = cfg.get('looper', {})
    if 'lightglue_weight_dir' in looper_cfg:
        looper_cfg['lightglue_weight_dir'] = rewrite_path(looper_cfg['lightglue_weight_dir'], repo_prefix_map)
