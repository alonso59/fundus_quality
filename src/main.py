import os
import yaml
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm

from classification.train import train as clsf_train
from preprocessing.train import train as pre_train
from classification.predict import implement as clsf_impl
from preprocessing.predict import implement as pre_impl



def main():
    """
    (*) default
                       Required                                                         
    python src/main.py --stage  class*  --mode  train*   --config   configs/classifier.yaml*            
                                pre             eval                configs/segmenter.yaml                           
                                det             predict             


                        Required        Optional
    python src/main.py --stage  impl    --source    dataset/images/
                                                    dataset/images/input.jpg
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=str, default='class', 
                        help='class, pre, det, impl', required=True)
    parser.add_argument('--config', type=str, default='configs/classifier.yaml', help='config file', required=False)
    parser.add_argument('--mode', type=str, default='eval', help='train, eval, predict', required=False)
    parser.add_argument('--source', type=str, default='dataset/images', help='file or dir/, jpg, png, bmp, tiff')

    opt = parser.parse_args()
    config = opt.config

    stage = opt.stage
    mode = opt.mode
    source = opt.source

    if stage == 'impl':
        implement(source)
    elif stage == 'class':
        assert config == 'configs/classifier.yaml'
        if mode == 'train':
            with open(config, "r") as ymlfile:
                cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
            clsf_train(cfg)
    elif stage == 'pre':
        assert config == 'configs/segmenter.yaml'
        if mode == 'train':
            with open(config, "r") as ymlfile:
                cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
            pre_train(cfg)
    elif stage == 'det':
        pass
    else:
        raise RuntimeError('Mode/stage combination not implemented!')

def implement(source):
    preprocessing_weights = 'pretrain/unet_weights.pth'
    classifier_weights = 'pretrain/drnetq_weights.pth'
    if len(source) == 1:
        crop, pred = pre_impl(source, 'unet', preprocessing_weights, 224, 1, 'cuda')
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


def get_filenames(path):
    X0 = []
    exp = 'jpeg', 'jpg', 'JPG', 'png', 'PNG', 'bmp', 'tif', 'tiff',
    for i in sorted(os.listdir(path)):
        if i.endswith(exp):
            X0.append(os.path.join(path, i))
    return X0



if __name__ == '__main__':
    main()