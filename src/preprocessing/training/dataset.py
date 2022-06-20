import os
import cv2
import numpy as np
import albumentations as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class ImagesFromFolder(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, preprocess_input=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.preprocess_input = None 

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])

        # read data
        image = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        image, mask = np.array(image), np.array(mask)
        image = image.transpose((2, 0, 1))
        mask = np.expand_dims(mask, axis=0) / 255
            
        return image, mask 


def loaders(train_imgdir,
            train_maskdir,
            val_imgdir,
            val_maskdir,
            batch_size,
            num_workers=4,
            pin_memory=True,
            preprocess_input=None,
            image_size = 224
            ):

    train_transforms = T.Compose(
        [
            T.Resize(image_size, image_size),
            T.Rotate(limit=(-30, 30), p=0.5),
            T.HorizontalFlip(p=0.5),
            T.VerticalFlip(p=0.5),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]
    )
    val_transforms = T.Compose(
        [
            T.Resize(image_size, image_size),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]
    )
    train_ds = ImagesFromFolder(image_dir=train_imgdir,
                                mask_dir=train_maskdir,
                                transform=train_transforms,
                                preprocess_input=preprocess_input
                                )

    val_ds = ImagesFromFolder(image_dir=val_imgdir,
                              mask_dir=val_maskdir,
                              transform=val_transforms,
                              preprocess_input=preprocess_input
                              )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )
    return train_loader, val_loader

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    _transform = [
        T.Lambda(image=preprocessing_fn),
    ]
    return T.Compose(_transform)
