import sys
import torch
import numpy as np
import torch.nn as nn
from trainer import *
from retinet import QRetiNet
from data import loaders
from torchsummary import summary
from utils import create_dir, seeding
from swin_transformer import SwinTransformer, swin_t
from torch.optim.lr_scheduler import StepLR
import torchmetrics.functional as M
import os
import configparser
import logging
import torchvision.models as models
from add_pretrain import DRNetQTImageNet
from NAT import *
def main():
    """ Seeding """
    seeding(42)
    config = configparser.ConfigParser()
    config.read('configs/qretinet.ini')
    paths = config['PATHS']
    hyperparameters = config['HYPERPARAMETERS']
    general = config['GENERAL']
    """
    Directories
    """
    ver_ = 0
    while(os.path.exists(f"logs/version{ver_}/")):
        ver_ += 1
    version = f"logs/version{ver_}/"
    checkpoint_path = version + "checkpoints/"
    create_dir(checkpoint_path)
    with open(version + 'config.txt', 'w') as configfile:
        config.write(configfile)
    """
    logging
    """
    logging.basicConfig(filename=version + "log.log",
                        filemode='a',
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger()
    stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stdout_handler)
    """ 
    Hyperparameters 
    """
    batch_size = hyperparameters.getint('batch_size')
    num_epochs = hyperparameters.getint('num_epochs')
    lr = hyperparameters.getfloat('lr')
    B1 = hyperparameters.getfloat('B1')
    B2 = hyperparameters.getfloat('B2')
    weight_decay = hyperparameters.getfloat('weight_decay')
    gpus_ids = [0]
    """
    Paths
    """
    train_imgdir = paths.get('train_imgdir')
    val_imgdir = paths.get('val_imgdir')
    """
    General settings
    """
    n_classes = general.getint('n_classes')
    img_size = 224
    device = torch.device(f"cuda" if torch.cuda.is_available() else 'cpu')
    """ 
    Getting loader
    """
    train_loader, val_loader = loaders(train_imgdir, val_imgdir, img_size, batch_size)
    """ Building model """
    # model = models.resnet18(pretrained=True)
    # model = models.vgg13(pretrained=True)
    # model.classifier[6] = nn.Linear(4096, 2)
    # model = QRetiNet(n_classes=n_classes)  # create object model
    # model = SwinTransformer(
    # hidden_dim=96,
    # layers=(2, 2, 6, 2),
    # heads=(3, 6, 12, 24),
    # channels=3,
    # num_classes=2,
    # head_dim=16,
    # window_size=7,
    # downscaling_factors=(4, 2, 2, 2),
    # relative_pos_embedding=True
    # )
    model = swin_t()
    load = torch.load('pretrain/swint_ep300.pth')
    model.load_state_dict(load, strict=False)
    model.mlp_head = nn.Sequential(
        nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True),
        nn.Linear(in_features=768, out_features=n_classes, bias=True)
    )
    # model = nat_nano(pretrained=True, num_classes=2)
    model.to(device)
    # print(model)
    # sys.exit()
    summary(model, input_size=(3, img_size, img_size), batch_size=-1)
    name_model = 'SwinTiny'
    is_inception = False
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    if len(gpus_ids) > 1:
        print("Data parallel...")
        model = nn.DataParallel(model, device_ids=gpus_ids)
    """ 
    Prepare training 
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(B1, B2))
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=B1)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer=optimizer, step_size=num_epochs * 0.1, gamma=0.8)
    logger.info(f'Total_params:{pytorch_total_params}')
    """ 
    Trainer
    """
    # print()
    # for idx, m in enumerate(model.modules()):
    #     print(idx, '->', m)
    #     layer = m
    #     if idx == 27:
    #         break
    # layer = [model.stage4.layers.modules]
    # print(model)
    # sys.exit()
    # layer = [model.levels[-1].blocks[-1].norm1]
    # layer = [model.stage4.layers[0][1].mlp_block.fn.norm]
    # layer = [model.layers.conv8]
    layer = [model.stage4.layers[0][1].mlp_block.fn.norm]
    print(layer)
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
            metric=None,
            device=device,
            checkpoint_path=checkpoint_path,
            scheduler=scheduler,
            iter_plot_img=int(num_epochs * 0.1),
            name_model=name_model,
            callback_stop_value=int(num_epochs * 0.15),
            tb_dir=version,
            logger=logger,
            layer=layer,
            is_inception=is_inception
            )
    logger.info('-------------------- Finished Train ---------------------')
    logger.info('******************* Start evaluation  *******************')
    load_best_model = torch.load(checkpoint_path + 'model.pth')
    loss_eval, acc_eval = eval(load_best_model, val_loader, loss_fn, device)
    logger.info([loss_eval, acc_eval])

if __name__ == '__main__':
    main()
