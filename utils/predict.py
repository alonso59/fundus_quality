import os
import torch
import numpy as np
from PIL import Image
import torch
import os
import argparse
from tqdm import tqdm
import pandas as pd
import sys
from time import time

def get_filenames(path, ext):
    X0 = []
    for i in sorted(os.listdir(path)):
        if i.endswith(ext):
            X0.append(os.path.join(path, i))
    return X0


def save_csv(csvfile, labels):
    df = pd.read_csv(csvfile)
    df["Qlabel"] = labels
    df.to_csv('test.csv', index=False)
    pass


def predict(file_model, img_file, img_size):
    model = torch.load(file_model)
    img_orig = Image.open(img_file)
    img = img_orig.resize((img_size, img_size))
    img = np.array(img) / 255.
    image = img.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)
    image = torch.tensor(image, device='cuda', dtype=torch.float)
    start = time()
    pred = model(image)
    stop = time()
    et = stop - start
    print('[Elapsed time]: '+ str(et) + ' s')
    pred = torch.softmax(pred, dim=1)
    pred = torch.argmax(pred).detach().cpu().numpy()
    return pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='Source read image path', required=True)
    parser.add_argument('-c', help='CSV File', required=True)
    parser.add_argument('-m', help='Model file', required=True)
    parser.add_argument('-s', help='Image size', required=True, type=int)
    args = parser.parse_args()
    Read_path = args.i
    csvfile = args.c
    file_model = args.m
    image_size = args.s
    exp = 'jpeg', 'jpg', 'JPG', 'png', 'PNG', 'bmp', 'tif', 'tiff'
    files = get_filenames(Read_path, exp)
    pred = []
    for img_file in tqdm(files):
        pred.append(predict(file_model, img_file, image_size))
    
    save_csv(csvfile, pred)

def single():

    predict('logs/version10/checkpoints/model.pth', 'dataset/D2/val/1/11900_right.jpeg' , 224)


if __name__ == '__main__':
    single()
