import os
import sys
import shutil
import numpy as np
import configparser
import multiprocessing as mp
import matplotlib.pyplot as plt

from PIL import Image
from eyepy.core.base import Oct
from sklearn.model_selection import train_test_split
from utils import get_filenames, create_dir

def split_data(path, images_path, masks_path, train_size=0.1):
    extention = 'jpeg'

    train_images_dir = path + 'train_images/'
    val_images_dir = path + 'val_images/'
    train_masks_dir = path + 'train_masks/'
    val_masks_dir = path + 'val_masks/'

    create_dir(train_images_dir)
    create_dir(val_images_dir)
    create_dir(train_masks_dir)
    create_dir(val_masks_dir)

    x = get_filenames(images_path, extention)
    y = get_filenames(masks_path, extention)

    X_train, X_val, y_train, y_val = train_test_split(
        x, y, train_size=train_size, shuffle=True
    )

    for i, j in zip(X_train, y_train):
        shutil.copy(os.path.join(i), train_images_dir)
        shutil.copy(os.path.join(j), train_masks_dir)

    for i, j in zip(X_val, y_val):
        shutil.copy(os.path.join(i), val_images_dir)
        shutil.copy(os.path.join(j), val_masks_dir)

def main():
    split_data('dataset/', 'dataset/Images', 'dataset/Masks', train_size=0.8)

if __name__ == "__main__":
    main()
