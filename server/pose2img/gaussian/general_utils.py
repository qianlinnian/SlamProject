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
    return cfg 