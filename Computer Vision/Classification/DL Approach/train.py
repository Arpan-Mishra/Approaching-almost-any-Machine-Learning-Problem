import os

import numpy as np
import pandas as pd

import albumentations
import torch

from sklearn import metrics
from sklearn.model_selection import train_test_split

import dataset
import engine
from model import get_model

if __name__ == '__main__':
    data_path = 'C:/Users/Arpan/Downloads/Education/ML/Approaching-almost-any-Machine-Learning-Problem/Computer Vision/Data' 

    device = 'cuda'
    df = pd.read_csv(os.path.join(data_path, 'train.csv'))
    epochs = 10
    images = df['ImageId'].values.tolist()
    images = [os.path.join(data_path, 'train_png', i+'.png') for i in images]

    targets = df['target'].values
    
    model = get_model(pretrained=True, model_name='resnet34')

    model.to(device)

    # imagenet stats
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    aug = albumentations.Compose([albumentations.Normalize(mean, std,
    max_pixel_value=255, always_apply=True)])

    train_images, val_images, train_targets, val_targets = train_test_split(
        images, targets, stratify = targets, test_size = 0.2, random_state = 42
    )

    train_dataset = dataset.ClassificationDataset(image_paths=train_images,
    targets = train_targets, resize=(227, 227),
    augmentations=aug)

    train_loader = torch.utils.data.DataLoader(train_dataset,
    batch_size = 16,
    shuffle=True, num_workers=0)

    valid_dataset = dataset.ClassificationDataset(image_paths=val_images,
    targets = val_targets, resize=(227, 227),
    augmentations=aug)

    valid_loader = torch.utils.data.DataLoader(valid_dataset,
    batch_size = 16,
    shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr = 5e-4)

    best_auc = 0

    for epoch in range(epochs):
        engine.train(train_loader, model, optimizer, device)
        predictions, valid_targets = engine.evaluate(valid_loader,
        model, device)
        roc_auc = metrics.roc_auc_score(valid_targets, predictions)

        if roc_auc > best_auc:
            best_auc = roc_auc
            torch.save(model.state_dict(), 'model.bin')
        print(f'Epoch: {epoch}, Valid AUC: {roc_auc}')

