import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch

def create_data(dataset, image_size, batch_size):
# Data loading code
    traindir = os.path.join(dataset, 'train')
    valdir = os.path.join(dataset, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((image_size,image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(-30, 30)),
            transforms.RandomAutocontrast(),
            # transforms.PILToTensor(),
            transforms.ToTensor(),
            normalize

        ]))

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=os.cpu_count(), pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((image_size,image_size)),
            transforms.ToTensor(),
            normalize
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=os.cpu_count(), pin_memory=True)

    print(train_loader.dataset.classes)
    return train_loader, val_loader


if __name__ == "__main__":
    pass