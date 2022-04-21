import os

import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils.unet import UNET
import torch.nn as nn
from utils.metrics import IoU
import torch.nn.functional as F
from utils.settings import Settings
from utils.dataset import OctDataset
import albumentations as T
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from utils.trainer import validation
import sys
from utils.loss import DiceLoss
from torchsummary import summary

def visualize(n, image, mask, pr_mask, path_save):
    """PLot images in one row."""
    figure, ax = plt.subplots(nrows=n, ncols=3)
    for i in range(n):
        ax[i, 0].imshow(image[i, :, :], cmap='gray')
        ax[0, 0].title.set_text('Test image')
        ax[i, 0].axis('off')
        ax[i, 1].imshow(mask[i, :, :], cmap='jet')
        ax[0, 1].title.set_text('Test mask')
        ax[i, 1].axis('off')
        ax[i, 2].imshow(pr_mask[i, :, :], cmap='jet')
        ax[0, 2].title.set_text('Prediction')
        ax[i, 2].axis('off')
    plt.savefig(path_save + "/prediction_" + str(np.random.randint(0, 100)) + ".png")
    plt.show()



def main():
    """ CUDA device """
    device = torch.device("cuda")
    PATH1 = 'checkpoints/11_00-54_john/model.pth',

    PATH2 = 'checkpoints/BEST/model.pth',

    val_transforms = T.Compose(
        [
            ToTensorV2(),
        ]
    )
    for path in PATH2:
        # load best saved checkpoint
        load = torch.load(path)
        try:
            best_model = load.to(device)
        except:
            key = list(load.keys())[3]
            features_start = load.__getitem__(key).cpu().numpy().shape[0]
            best_model = UNET(num_classes=3,
                              input_channels=1,
                              num_layers=4,
                              features_start=features_start,
                              bilinear=False,
                              dropout=False,
                              dp=0.5,
                              kernel_size=(3, 3),
                              padding=1
                              )
            best_model = best_model.to(device)
            best_model = nn.DataParallel(best_model, device_ids=[0])
            best_model.load_state_dict(load)
        # summary(best_model, input_size=(1, 128, 128), batch_size=1)
        settings = Settings()
        dataset_dir = settings.dataset_dir
        val_dir = dataset_dir + "val_images"
        val_maskdir = dataset_dir + "val_masks"

        val_ds = OctDataset(image_dir=val_dir,
                            mask_dir=val_maskdir,
                            transform=val_transforms,
                            )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            num_workers=os.cpu_count(),
            pin_memory=True,
            shuffle=False,
        )
        loss_fn = DiceLoss(device)
        iou = IoU(device)
        # loss_eval, iou_eval = validation(best_model, val_loader, loss_fn, iou, device)
        # print(loss_eval, iou_eval)
        imgs = []
        mask_true = []
        prds_msk = []
        j = 3
        # np.random.seed(15)  # 2, 10, 20, 42, 32
        for i in range(j):
            randint = np.random.randint(low=0, high=len(val_ds))
            image, mask = val_ds[randint]

            image1 = image.unsqueeze(1).float().to(device)
            mask1 = mask.unsqueeze(0)
            mask1 = mask1.unsqueeze(0).long().to(device)

            pr_mask = best_model(image1)
            metric = iou(pr_mask, mask1)
            metric = np.multiply(metric, 100)
            print(f"{metric[0]:2.4f}, {metric[1]:2.4f}, {metric[2]:2.4f}")
            print(np.mean(metric))
            pr_mask = F.softmax(pr_mask, dim=1)
            pr_mask = torch.argmax(pr_mask, dim=1)
            pr_mask = (pr_mask.squeeze().cpu().float().detach().numpy())

            imgs.append(image1.squeeze(0).squeeze(0).cpu().detach().numpy())
            mask_true.append(mask1.squeeze(0).squeeze(0).cpu().detach().numpy())
            prds_msk.append(pr_mask)

        path_save = os.path.split(path)[0]

        visualize(n=len(imgs), image=np.array(imgs),
                  mask=np.array(mask_true), pr_mask=np.array(prds_msk),
                  path_save=path_save
                  )


if __name__ == '__main__':
    main()
