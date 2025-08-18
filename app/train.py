import torch
import numpy as np
from dataset.cmmd_dataset import MemeDataset
from config import config
from models.main_model import MemeMultimodalDetector

from app.trainer.trainer import quick_smoke_train
from app.trainer.model_factory import build_model

from app.utils.logger import get_logger

logger = get_logger()

variants = ["shared_encoder"]
seeds    = [42, 43]
cfg      = {"batch_size": 8, "device": "cuda"}


train_dataset = MemeDataset(config['train'])
val_dataset = MemeDataset(config['test'])

def smoke_test():
    mini_results = {}
    for name in variants:
        f1_list = []
        for sd in seeds:
            model = build_model(name)
            best_f1,_,_,_ = quick_smoke_train(
                model, cfg,
                train_dataset, val_dataset,
                train_frac=0.02,  # 2 % Training
                val_frac=0.05,    # 5 % Validation
                epochs=3,
                seed=sd,
                lr=5e-5
            )
            f1_list.append(best_f1)
        mini_results[name] = (np.mean(f1_list), np.std(f1_list))
        logger.info(f"[Mini] {name}: {mini_results[name][0]*100:.2f} ± {mini_results[name][1]*100:.2f}")
        

def mini_ablation():
    mini_results = {}
    for name in variants:
        f1_list = []
        for sd in seeds:
            model = build_model(name)
            best_f1, _ , model , _= quick_smoke_train(
                model, cfg,
                train_dataset, val_dataset,
                train_frac=0.10,  # 10 % Training
                val_frac=0.30,    # 30 % Validation
                epochs=10,
                seed=sd,
                lr=1e-5
            )
            f1_list.append(best_f1)
        mini_results[name] = (np.mean(f1_list), np.std(f1_list))
        print(f"[Mini] {name}: {mini_results[name][0]*100:.2f} ± {mini_results[name][1]*100:.2f}")

def run():
    best_variant = "shared_encoder"
    final_model  = build_model(best_variant)

    best_f1, report, model, val_loader = quick_smoke_train(
        final_model, cfg,
        train_dataset, val_dataset,
        train_frac=1.0,
        val_frac=1.0,
        epochs=15,
        seed=42,
        lr=1e-5,
        early_stop_patience=10
    )
    print("★ Final Macro F1:", best_f1)