import pandas as pd
import torch
import os
from torch.utils.data import DataLoader
from .utils import use_cuda
from .config import Config as conf
from .dataset import TestDataset


def inference(model, transform, file_name):
    submission = pd.read_csv(os.path.join(conf.test_dir, 'info.csv'))
    ensemble = pd.read_csv(os.path.join(conf.test_dir, 'info.csv')) # save logits to use when ensemble with other models
    image_dir = os.path.join(conf.test_dir, 'images')

    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]

    test_dataset = TestDataset(image_paths, transform)

    test_loader = DataLoader(
        test_dataset,
        shuffle=False
    )
    
    all_predictions = []
    logit_predictions = []
    
    device = use_cuda()
    for images in test_loader:
        with torch.no_grad():
            images = images['image'].to(device)
            pred = model(images)
            logit_predictions.extend(pred.cpu().numpy())
            pred = pred.argmax(dim=-1)
            all_predictions.extend(pred.cpu().numpy())

    submission['ans'] = all_predictions
    ensemble['ans'] = all_predictions
    ensemble['logit'] = logit_predictions

    cnt = 0
    for idx, ans in enumerate(submission['ans']):
        if ans < 0 or ans > 17: # if answer is out of range, replace it with 0
            cnt += 1
            submission['ans'][idx] = 0
            ensemble['ans'][idx] = 0
    if cnt != 0:
        print(f"[WARNING] {cnt} answers were out of ranges")
    
    submission.to_csv(os.path.join(conf.submission_dir, file_name), index=False)
    ensemble.to_csv(os.path.join(conf.ensemble_dir, file_name), index=False)
    
    print("="*100)
    print("Inference Done")
    print("="*100)