import os
from torch.utils.data import DataLoader
from torchvision import datasets
import torch
import albumentations as T
import cv2
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np

class Transforms:
    def __init__(self, transforms: T.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))

def loaders(traindir, valdir, image_size, batch_size):
    train_transforms = T.Compose([
        T.Resize(image_size, image_size),
        T.Rotate(limit=(-35, 35), p=1.0, border_mode=cv2.BORDER_CONSTANT),
        T.HorizontalFlip(p=0.5),
        T.OneOf([
            T.ColorJitter(brightness=(0.5,2), contrast=(0.5,2), saturation=(0.5,2), p=0.5),
            T.CLAHE(clip_limit=4, tile_grid_size=(8, 8), p=0.5),
        ], p=0.5),
        # T.Affine(scale=(0.9, 1.1), translate_percent=(0.2, 0.2), p=0.5),
        T.OneOf([
            T.GaussianBlur(blur_limit=(1, 3), p=0.2),
            T.GaussNoise(var_limit=(2, 4), mean=0, p=0.2),
        ], p=0.5),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # T.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
        ToTensorV2()
    ])

    val_transforms = T.Compose([
        T.Resize(image_size, image_size),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # T.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
        ToTensorV2()
    ])

    train_dataset = datasets.ImageFolder(
        traindir, transform=Transforms(train_transforms)
    )
    val_dataset = datasets.ImageFolder(
        valdir, transform=Transforms(val_transforms)
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)

    print(train_loader.dataset.classes)
    return train_loader, val_loader