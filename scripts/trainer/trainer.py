import torch
import random
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit

from scripts.trainer.predict import predict


def collate_fn(batch):
    return {
        'image': [b['image'] for b in batch],
        'prompt': [b['prompt'] for b in batch],
        'caption':[b['caption']for b in batch],
        'explanation': [b['explanation'] for b in batch],
        'label': torch.stack([b['label'] for b in batch])
    }


def get_stratified_subset(dataset, labels, frac=0.1, seed=42):
    sss = StratifiedShuffleSplit(n_splits=1, train_size=frac, random_state=seed)
    idx_small, _ = next(sss.split(X=np.zeros(len(labels)), y=labels))
    return Subset(dataset, idx_small)

def set_seeds(seed=2025):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def quick_smoke_train(
        model,
        config: dict,
        train_ds,              # full train set
        val_ds,                # full val set
        train_frac=0.1,        # train subset ratio
        val_frac=0.3,          # val subset ratio (1.0 for full)
        epochs=3,
        seed=2025,
        lr=5e-5,               # learning rate
        early_stop_patience=3,
        use_amp=True,
        num_workers=4
    ):
    """
    return: (best_macro_f1, best_cls_report_dict)
    """
    # 0. set seed
    set_seeds(seed)

    # 1. stratified subset
    if not hasattr(train_ds, "labels"):
        raise AttributeError("train_ds must have labels attribute")

    if train_frac < 1.0:
      if not hasattr(train_ds, "labels"):
        raise AttributeError("train_ds must have labels attribute")
      sub_train = get_stratified_subset(train_ds, train_ds.labels, train_frac, seed)
    else:
      sub_train = train_ds

    if val_frac < 1.0:
        if not hasattr(val_ds, "labels"):
            raise AttributeError("val_ds must have labels attribute")
        sub_val = get_stratified_subset(val_ds, val_ds.labels, val_frac, seed)
    else:
        sub_val = val_ds

    # 2. DataLoader
    train_loader = DataLoader(
        sub_train, batch_size=config["batch_size"], shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        sub_val, batch_size=config["batch_size"], shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
    )

    # 3. optimizer
    model = model.to(config["device"])
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 4. train loop
    best_f1, best_report, wait = 0.0, None, 0
    for ep in range(1, epochs + 1):
        model.train(); total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"[EP {ep}/{epochs}]"):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                _, loss = model(
                    image=batch["image"],
                    prompt=batch["prompt"],
                    explanation=batch["explanation"],
                    caption=batch["caption"],
                    labels=batch["label"].to(config["device"])
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            total_loss += loss.item()

        # ——validation——
        model.eval()
        all_preds, all_labels = predict(model, val_loader, desciption="VAL")
        report   = classification_report(all_labels, all_preds,
                                         digits=4, output_dict=True)
        macro_f1 = report["macro avg"]["f1-score"]
        print(f"EP{ep}  loss={total_loss:.3f}  macroF1={macro_f1:.4f}")

        if macro_f1 > best_f1 + 1e-4:
            best_f1, best_report = macro_f1, report
            torch.save(model.state_dict(), f'checkpoint-model-temp.pth')
            wait = 0
        else:
            wait += 1
            if wait >= early_stop_patience:
                print("Early stopping triggered"); break

    return best_f1, best_report, model, val_loader