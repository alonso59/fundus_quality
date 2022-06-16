import os
import sys
import logging
import torch
import torch.nn as nn
from common.initialize import initialize as init
import segmentation_models_pytorch as smp
from .training.metrics import mIoU
from .training.trainer import trainer, eval
from .training.dataset import loaders
from .utils.utils import create_dir, seeding
from .models import SegmentationModels
from matplotlib import pyplot as plt
from .training.scheduler import CyclicCosineDecayLR
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from .training.loss import WCEGeneralizedDiceLoss, DiceLoss, CrossEntropyLoss, BinaryCrossEntropyLoss

from pytorch_model_summary import summary as sm
from torchmetrics import Accuracy
import datetime
import yaml


def train(cfg):
    logger, checkpoint_path, version = init(cfg, 'preprocessing')
    paths = cfg['paths']
    hyper = cfg['hyperparameters']
    general = cfg['general']
    """ 
    Hyperparameters 
    """
    batch_size = hyper['batch_size']
    num_epochs = hyper['num_epochs']
    lr = hyper['lr']
    B1 = hyper['b1']
    B2 = hyper['b2']
    weight_decay = hyper['weight_decay']
    gpus_ids = [0]
    """
    Paths
    """
    train_imgdir = paths['train_imgdir']
    train_mskdir = paths['train_mskdir']
    val_imgdir = paths['val_imgdir']
    val_mskdir = paths['val_mskdir']
    """
    General settings
    """
    n_classes = general['n_classes']
    img_size = general['img_size']
    pretrain = general['pretrain']
    name_model = general['model_name']
    device = torch.device("cuda")
    """ 
    Getting loader
    """
    train_loader, val_loader = loaders(train_imgdir=train_imgdir,
                                       train_maskdir=train_mskdir,
                                       val_imgdir=val_imgdir,
                                       val_maskdir=val_mskdir,
                                       batch_size=batch_size,
                                       num_workers=os.cpu_count(),
                                       pin_memory=True,
                                       preprocess_input=None,
                                       image_size=img_size
                                       )
    """ 
    Building model 
    """
    models_class = SegmentationModels(device, in_channels=3, img_size=img_size, n_classes=n_classes)
    model, preprocess_input, name_model = models_class.UNet(feature_start=32, layers=3, kernel_size=3, stride=1, padding=1)
    models_class.summary(logger=logger)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(preprocess_input)
    logger.info(f'Total_params:{pytorch_total_params}')
    if len(gpus_ids) > 1:
        print("Data parallel...")
        model = nn.DataParallel(model, device_ids=gpus_ids)
    """ 
    Prepare training 
    """
    if hyper['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(B1, B2))
    elif hyper['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=B1)
    else:
        raise AssertionError('Optimizer not implemented')
    assert hyper['loss_fn'] == 'binary_crossentropy', "Loss function not implemented"
    loss_fn = nn.BCEWithLogitsLoss()
    metrics = Accuracy()
    scheduler = StepLR(optimizer=optimizer, step_size=cfg['hyperparameters']['scheduler']['step'], gamma=cfg['hyperparameters']['scheduler']['gamma'])
    # if cfg.SCHEDULER == 'step':
    #     scheduler = StepLR(optimizer=optimizer, step_size=cfg.STEP_SIZE, gamma=cfg.GAMMA)
    # if cfg.SCHEDULER == 'cosine':
    #     scheduler = CyclicCosineDecayLR(optimizer,
    #                                 init_decay_epochs=num_epochs // 3,
    #                                 min_decay_lr=lr / 10,
    #                                 restart_interval=num_epochs // 10,
    #                                 restart_lr=lr / 5)
    """ 
    Trainer
    """
    logger.info('**********************************************************')
    logger.info('**************** Initialization sucessful ****************')
    logger.info('**********************************************************')
    logger.info('--------------------- Start training ---------------------')
    trainer(num_epochs=num_epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metric=metrics,
            device=device,
            checkpoint_path=checkpoint_path,
            scheduler=scheduler,
            iter_plot_img=int(num_epochs * 0.1),
            name_model=name_model,
            callback_stop_value=int(num_epochs * 0.15),
            tb_dir = version,
            logger=logger
            )
    logger.info('-------------------- Finished Train ---------------------')
    logger.info('******************* Start evaluation  *******************')
    load_best_model = torch.load(checkpoint_path + 'model.pth')
    loss_eval, acc_eval = eval(load_best_model, val_loader, loss_fn, device)
    logger.info([loss_eval, acc_eval])


if __name__ == '__main__':
    with open('/home/alonso/Documents/fundus_suitable/configs/drnetq.yaml', "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    train(cfg)