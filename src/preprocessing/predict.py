import torch
import argparse
import numpy as np

from tqdm import tqdm
from scipy.ndimage import label
from PIL import Image, ImageFilter

from .models import SegmentationModels
from common.utils import get_filenames

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

def implement(source, config, weights, img_size, n_classes, device):
    dict_weights = torch.load(weights, map_location=device)
    """ Building model """
    models_class = SegmentationModels(device, in_channels=1, img_sizeh=img_size, img_sizew=img_size, config_file=config, n_classes=n_classes)
    model, name_model = models_class.UNet(feature_start=32, layers=3, kernel_size=3, stride=1, padding=1)
    model.load_state_dict(dict_weights, strict=True)
    pred = predict(model, source, img_size)
    return pred

def predict(model, source, img_size):
    """
    Perform the prediction of the retina's mask.

    :param src_path: str, Retina images directory name.
    :param dst_path: str, Destination directory name where the crop will be stored.
    :param model_name: st, Name of the retina crop model.
    :return: Nothing.
    """
    img_orig = Image.open(source)
    h, w = img_orig.size
    img = img_orig.resize((img_size, img_size)).convert('L')
    img = np.array(img)
    img_orig = np.array(img_orig).astype('float')
    image = np.expand_dims(img, axis=0)
    image = np.expand_dims(image, axis=0)
    image = torch.tensor(image, device='cuda', dtype=torch.float)
    pred = model(image)
    pred = torch.sigmoid(pred)
    pred = pred.detach().cpu().squeeze(0).squeeze(0).numpy()
    pred = np.round(pred).astype('float')
    pred = Image.fromarray(pred)
    pred = pred.filter(ImageFilter.MaxFilter(3))
    pred = remove_remaining(pred)
    pred = pred.resize((h, w))
    pred = np.array(pred) / 255.
    pos = np.where(pred)
    if is_retina_mask_empty(pos):
        print('Invalid image!')
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    pred = np.expand_dims(pred, axis=-1)
    pred = np.repeat(pred, 3, axis=-1)
    img_mult = img_orig / 255
    crop = img_mult[ymin:ymax, xmin:xmax]
    crop = add_zerosboxes(crop)
    crop = (crop * 255).astype(np.uint8)
    pred = (pred * 255).astype(np.uint8)
    return crop, pred

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