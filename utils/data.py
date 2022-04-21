import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch


def loaders(traindir, valdir, image_size, batch_size):
    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=(-20, 20)),
        transforms.RandomAutocontrast(),
        transforms.ToTensor(),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    train_dataset = datasets.ImageFolder(
        traindir, transform=train_transforms
    )
    val_dataset = datasets.ImageFolder(
        valdir, transform=val_transforms
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(),
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

    print(train_loader.dataset.classes)
    return train_loader, val_loader


