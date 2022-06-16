import numpy as np
from PIL import Image
import torch
import os
import argparse
from tqdm import tqdm
from PIL import ImageFilter
from scipy.ndimage import label
import torch.nn.functional as F
from .models import SegmentationModels

def is_retina_mask_empty(retina_mask):
    for mask in retina_mask:
        if mask.size == 0:
            return True


def add_zerosboxes(img):
    """
    Add zero boxes to the retina image to adjust the size to nxn resolution.

    :param img: numpyarray, The image.
    :return: Numpy array (image) with zeros boxes.
    """
    h, w, c = img.shape
    axis = 0
    b = abs(int(h - w))
    b1 = int(b / 2)
    b2 = b - b1
    if h > w:
        axis = 1
        z1 = np.zeros((h, b1, c))
        z2 = np.zeros((h, b2, c))
    elif w > h:
        z1 = np.zeros((b1, w, c))
        z2 = np.zeros((b2, w, c))
    else:
        return img
    newimg = np.append(img, z1, axis=axis)
    newimg = np.append(z2, newimg, axis=axis)
    return newimg


def remove_remaining(prediction):
    '''
    Remove the remaining groups of prediction pixels.
    :param prediction: Pillow Image, prediction (segmentation) of the model.
    :return: Pillow Image.
    '''
    prediction = np.array(prediction)
    labels, features = label(prediction)
    h, w = prediction.shape
    groups = np.unique(labels)

    # Since label method converts the values of the matrix, we don't know which
    # ones are the black pixels so it will help to identify them by pixel counter.
    # Most of the times zeros of labels means zeros of the prediction but it
    # could change so we must ensure.
    zeros_counter = prediction[(prediction == 0)].size
    zeros_id = 0

    remainings = []
    for group in groups:
        count = labels[(labels == group)].size
        if count == zeros_counter:
            zeros_id = group
        percentage = count/(h*w)
        if percentage < 0.30:
            remainings.append(group)

    for remaining_group in remainings:
        labels = np.where(labels==remaining_group, zeros_id, labels)

    # Reconvert to binary format
    labels = np.where(labels != zeros_id, 255, labels)
    labels = np.where(labels == zeros_id, 0, labels)
    return Image.fromarray(labels.astype(np.uint8))

def implement(source, model_name, weights, img_size, n_classes, device):
    dict_weights = torch.load(weights, map_location=device)
    """ Building model """
    models_class = SegmentationModels(device, in_channels=3, img_size=img_size, n_classes=n_classes)
    model, preprocess_input, name_model = models_class.UNet(feature_start=32, layers=3, kernel_size=3, stride=1, padding=1)
    print(dict_weights)
    # original saved file with DataParallel
    state_dict = torch.load(weights)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    pred = predict(model, img_size, source, device)
    return pred

def predict(model, img_size, source, device):
    """
    Perform the prediction of the retina's mask.

    :param src_path: str, Retina images directory name.
    :param dst_path: str, Destination directory name where the crop will be stored.
    :param model_name: st, Name of the retina crop model.
    :return: Nothing.
    """
    image = Image.open(source)
    gray = image.convert('RGB')
    h, w, _ = np.array(gray).shape
    image = np.array(image)
    print(image.shape)
    gray = gray.resize((img_size, img_size))
    print(np.array(gray).shape)
    gray = np.array(gray).transpose((2,0,1))
    gray = np.expand_dims(gray, axis=0)
    gray = torch.tensor(gray, device=device, dtype=torch.float)
    pred = model(gray)
    pred = torch.sigmoid(pred)
    pred = pred.detach().cpu().squeeze(0).squeeze(0).numpy()
    pred = np.round(pred).astype('float')
    pred = Image.fromarray(pred)
    pred = pred.filter(ImageFilter.MinFilter(3))
    pred = remove_remaining(pred)
    pred = pred.resize((h, w))
    pred = np.array(pred) / 255.
    print(pred.max())
    pos = np.where(pred)
    if is_retina_mask_empty(pos):
        raise RuntimeError('Error image empty!')
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    crop = image[ymin:ymax, xmin:xmax, :]
    crop = add_zerosboxes(crop)
    crop = (crop * 255).astype(np.uint8)
    return crop

def get_filenames(path, ext):
    X0 = []
    for i in sorted(os.listdir(path)):
        if i.endswith(ext):
            X0.append(os.path.join(path, i))
    return X0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='Source read image path', required=True)
    parser.add_argument('-d', help='Destiny write image path', required=True)
    parser.add_argument('-m', help='Model file', required=True)
    parser.add_argument('-s', help='Image size', required=True, type=int)
    args = parser.parse_args()
    Read_path = args.i
    Write_path = args.d
    file_model = args.m
    image_size = args.s
    exp = 'jpeg', 'jpg', 'JPG', 'png', 'PNG', 'bmp', 'tif', 'tiff', 
    files = get_filenames(Read_path, exp)
    for img_file in tqdm(files):
        try:
            predict(file_model, img_file, Write_path, image_size)
        except RuntimeError as exc:
            print(exc)
            print(img_file)
            pass

if __name__ == '__main__':
    main()
