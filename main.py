import yaml
import argparse
from classification.train import train as clsf_train
from preprocessing.train import train as pre_train
from classification.predict import implement as clsf_impl
from preprocessing.predict import implement as pre_impl
import cv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=str, default='classification', 
                        help='classification, segmentation, detection, implement', required=True)
    parser.add_argument('--mode', type=str, default='eval', help='train, eval, predict', required=False)
    parser.add_argument('--source', type=str, default='dataset/images', help='file or dir/, jpg, png, bmp, tiff')
    parser.add_argument('--weights', type=str, default=None, help='model path(s)', required=False)
    parser.add_argument('--config', type=str, default='configs/drnetq.yaml', help='config file')
    opt = parser.parse_args()
    config = opt.config

    stage = opt.stage
    mode = opt.mode

    if stage == 'implement':
        implement('dataset/D2/train/1/5559_right.jpeg')
    elif stage == 'classification':
        if mode == 'train':
            with open(config, "r") as ymlfile:
                cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
            clsf_train(cfg)
    elif stage == 'segmentation':
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
    pred = pre_impl(source, 'unet', preprocessing_weights, 512, 1, 'cuda')
    cv2.imwrite('test.png', pred)
    # clsf_impl(image, 'drnetq', classifier_weights, 224, 2, 'cuda')
if __name__ == '__main__':
    main()