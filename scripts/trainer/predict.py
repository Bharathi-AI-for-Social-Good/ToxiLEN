import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from utils.logger import get_logger

logger = get_logger()


def collate_fn(batch):
    return {
        'image': [b['image'] for b in batch],
        'prompt': [b['prompt'] for b in batch],
        'caption':[b['caption']for b in batch],
        'explanation': [b['explanation'] for b in batch],
        'label': torch.stack([b['label'] for b in batch])
    }


def predict(model, dataloader, desciption):
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc= desciption):
            log_probs,_ = model(
                    image=batch["image"],
                    prompt=batch["prompt"],
                    explanation=batch["explanation"],
                    caption=batch["caption"],
                    labels=batch["label"].to('cuda')
            )
            preds = log_probs.argmax(dim=1).cpu().tolist()
            labels = batch['label'].cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)

    return all_preds, all_labels

def test(model, config, dataset):
    dataset = dataset
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)

    all_preds, all_labels = predict(model, dataloader,"Test")
    logger.info(classification_report(all_labels, all_preds, digits=4))
    return all_preds,all_labels