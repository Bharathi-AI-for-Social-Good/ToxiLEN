import torch
import pandas as pd

common_config = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'clip_model_name': 'openai/clip-vit-base-patch32',
    'clip_img_dim':768,
    'clip_text_dim': 512,
    'caption_dim': 768,
    'detr_dim': 1024,
    'span_dim': 768,
    'proj_dim': 1024,
    'hidden_dim': 1024,
    'num_latents': 1,
    'num_classes': 2,
    'batch_size':8,
    'epochs': 20,
}

config = {
    'train': {
        **common_config,
        'image_dir': 'data/cindy/images/train',
        'data': 'data/cindy/data/train.csv',
    },
    # 'dev': {
    #     **common_config,
    #     'image_dir': 'data/cindy/images/dev',
    #     'text_csv': 'data/cindy/all/dev.csv',
    # },
    'test': {
        **common_config,
        'image_dir': 'data/cindy/images/test',
        'data': 'data/cindy/data/test.csv',
    },
    'toximm_train':{
        **common_config,
        'image_dir': 'data/toximm/images',
        'data': 'data/toximm/train.json',
    },
    'toximm_test':{
        **common_config,
        'image_dir': 'data/toximm/images',
        'data': 'data/toximm/test.json',
    },
    'MET_train':{
        **common_config,
        'image_dir': 'data/MET_data/cn_images',
        'text_csv': 'data/MET_data/train.csv',
    },
    "MET_test":{
        **common_config,
        'image_dir': 'data/MET_data/cn_images',
        'text_csv': 'data/MET_data/test.csv',
    },
    }


