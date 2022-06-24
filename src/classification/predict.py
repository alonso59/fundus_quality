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
from .models import ClassificationModels

def get_filenames(path, ext):
    X0 = []
    for i in sorted(os.listdir(path)):
        if i.endswith(ext):
            X0.append(os.path.join(path, i))
    return X0

LABELS = []

def save_csv(csvfile, labels):
    df = pd.read_csv(csvfile)
    df["Qlabel"] = labels
    df.to_csv('test.csv', index=False)
    pass

def implement(source, model_name, weights, img_size, n_classes, device):
    dict_weights = torch.load(weights, map_location=device)
    """ Building model """
    model_classifier = ClassificationModels(device, 3, img_size, n_classes, False)
    model, layer, is_inception = model_classifier.model_builder(model_name=model_name) 
    model.load_state_dict(dict_weights, strict=False)
    pred, y_pr = predict(model, source, device)
    return pred, y_pr

def predict(model, image, device):
    image = np.array(image) / 255.
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)
    image = torch.tensor(image, device=device, dtype=torch.float)
    pred = model(image)
    pred = torch.softmax(pred, dim=1)
    y_pr = torch.max(pred).detach().cpu().numpy()
    pred = torch.argmax(pred).detach().cpu().numpy()
    
    return pred, y_pr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='Source read image path', required=True)
    # parser.add_argument('-c', help='CSV File', required=True)
    parser.add_argument('-m', help='Model file', required=True)
    parser.add_argument('-s', help='Image size', required=True, type=int)
    args = parser.parse_args()
    Read_path = args.i
    # csvfile = args.c
    file_model = args.m
    image_size = args.s
    exp = 'jpeg', 'jpg', 'JPG', 'png', 'PNG', 'bmp', 'tif', 'tiff'
    files = get_filenames(Read_path, exp)
    pred = []
    print(len(files))
    for img_file in tqdm(files):
        pred.append(predict(file_model, img_file, image_size))
    pred = np.array(pred)
    print(pred.shape)
    print(np.sum(pred==1))
    # save_csv(csvfile, pred)

def single():

    predict('logs/version0/checkpoints/model.pth', 'dataset/D2/val/1/11900_right.jpeg' , 224)


if __name__ == '__main__':
    single()
