import yaml
import argparse
from classification.train import train as clsf_train
from preprocessing.train import train as pre_train
from classification.predict import implement as clsf_impl
from preprocessing.predict import implement as pre_impl
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=str, default='classification', 
                        help='classification, segmentation, detection, implement', required=True)
    parser.add_argument('--mode', type=str, default='eval', help='train, eval, predict', required=False)
    parser.add_argument('--source', type=str, default='dataset/images', help='file or dir/, jpg, png, bmp, tiff')
    parser.add_argument('--config', type=str, default='configs/drnetq.yaml', help='config file')
    opt = parser.parse_args()
    config = opt.config

    stage = opt.stage
    mode = opt.mode

    if stage == 'implement':
        implement('dataset/PulbicDatasets/IDRID/train/')
    elif stage == 'classification':
        if mode == 'train':
            with open(config, "r") as ymlfile:
                cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
            clsf_train(cfg)
    elif stage == 'segmentation':
        assert config == 'configs/drnetq.yaml', 'Select correct config file for segmentation'
        if mode == 'train':
            with open(config, "r") as ymlfile:
                cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
            pre_train(cfg)
    elif stage == 'detection':
        pass
    else:
        raise RuntimeError('Mode/stage combination not implemented!')

def implement(source):
    preprocessing_weights = 'pretrain/unet_weights.pth'
    classifier_weights = 'pretrain/drnetq_weights.pth'
    if len(source) == 1:
        crop, pred = pre_impl(source, 'unet', preprocessing_weights, 512, 1, 'cuda')
        crop, pred = Image.fromarray(crop), Image.fromarray(pred)
        crop.save('crop.png')
        pred.save('pred.png')
        crop = crop.resize((224, 224))
        clsf_impl(np.array(crop), 'drnetq', classifier_weights, 224, 2, 'cuda')
    else:
        X0 = get_filenames(source)
        for i in tqdm(X0):
            crop, pred = pre_impl(i, 'unet', preprocessing_weights, 224, 1, 'cuda')
            crop, pred = Image.fromarray(crop), Image.fromarray(pred)
            # crop.save('crop.png')
            # pred.save('pred.png')
            crop = crop.resize((224, 224))
            clsf_impl(np.array(crop), 'drnetq', classifier_weights, 224, 2, 'cuda')
        # raise RuntimeError('Folder source prediction, not implemented!')

def get_filenames(path):
    X0 = []
    exp = 'jpeg', 'jpg', 'JPG', 'png', 'PNG', 'bmp', 'tif', 'tiff',
    for i in sorted(os.listdir(path)):
        if i.endswith(exp):
            X0.append(os.path.join(path, i))
    return X0



if __name__ == '__main__':
    main()