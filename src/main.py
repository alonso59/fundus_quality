import os
import yaml
import argparse
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

from classification.train import train as clsf_train
from preprocessing.train import train as pre_train
from classification.predict import implement as clsf_impl
from preprocessing.predict import implement as pre_impl
from detection.detect import run as det_impl
from detection.train import run as det_train

def main():
    """
    (*) default
                                                                                
    python src/main.py --stage  class*  --mode  train*   --config   configs/classifier.yaml* --source dataset/images/            
                                pre             eval                configs/segmenter.yaml            dataset/images/input.jpg               
                                det             predict             configs/detector.yaml


                                
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
        assert config == 'configs/detector.yaml'
        if mode == 'train':
            with open(config, "r") as ymlfile:
                cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
            det_train(batch=cfg['batch_size'], epochs=cfg['epochs'],weights=cfg['model_pretrain'], 
                      data=opt.config, imgsz=224, save_dir='logs/detection/train/')
        # python path/to/train.py --data coco128.yaml --weights yolov5s.pt --img 640
        pass
    else:
        raise RuntimeError('Mode/stage combination not implemented!')

def implement(source):
    preprocessing_weights = 'pretrain/unet_weights.pth'
    classifier_weights = 'pretrain/drnetq_weights.pth'
    detector_wieghts = 'pretrain/retina_detection.pt'
    # if len(source) == 1:
    with open('configs/segmenter.yaml', "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    crop, pred = pre_impl(source, cfg, 'unet', preprocessing_weights, 224, 1, 'cuda') # preprocessing
    crop, pred = Image.fromarray(crop), Image.fromarray(pred)
    # print('outputs/' + os.path.split(source)[1])
    crop.save('outputs/preprocessing/' + os.path.split(source)[1])
    # pred.save('pred.png')
    crop = crop.resize((224, 224))

    clsf_impl(np.array(crop), 'drnetq', classifier_weights, 224, 2, 'cuda') # classification
    # else:
    #     X0 = get_filenames(source)
    #     for i in tqdm(X0):
    #         crop, pred = pre_impl(i, 'unet', preprocessing_weights, 224, 1, 'cuda')
    #         crop, pred = Image.fromarray(crop), Image.fromarray(pred)
    #         # crop.save('crop.png')
    #         # pred.save('pred.png')
    #         crop = crop.resize((224, 224))
    #         clsf_impl(np.array(crop), 'drnetq', classifier_weights, 224, 2, 'cuda')
    det_impl('outputs/detection/', detector_wieghts, 'outputs/preprocessing/'+os.path.split(source)[1], imgsz=(224,224)) # detector


def get_filenames(path):
    X0 = []
    exp = 'jpeg', 'jpg', 'JPG', 'png', 'PNG', 'bmp', 'tif', 'tiff',
    for i in sorted(os.listdir(path)):
        if i.endswith(exp):
            X0.append(os.path.join(path, i))
    return X0



if __name__ == '__main__':
    main()