import yaml
import torch
import torch.nn as nn

from .training.trainer import *
from .training.dataset import loaders
from torch.optim.lr_scheduler import StepLR

from .models import ClassificationModels
from utils.initialize import initialize as init
def train(cfg):
    paths = cfg['paths']
    hyper = cfg['hyperparameters']
    general = cfg['general']
    logger, checkpoint_path, version = init(cfg, 'classification')
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
    val_imgdir = paths['val_imgdir']
    """
    General settings
    """
    n_classes = general['n_classes']
    img_size = general['img_size']
    pretrain = general['pretrain']
    name_model = cfg['model_name']
    device = torch.device("cuda")
    """ 
    Getting loader
    """
    train_loader, val_loader = loaders(train_imgdir, val_imgdir, img_size, batch_size)
    """ Building model """
    model_classifier = ClassificationModels(device, 3, img_size, n_classes, pretrain)
    model, layer, is_inception = model_classifier.model_builder(model_name=name_model) 
    model_classifier.summary(logger)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
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

    assert hyper['loss_fn'] == 'cross_entropy', "Loss function not implemented"
    loss_fn = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer=optimizer, step_size=cfg['hyperparameters']['scheduler']['step'], gamma=cfg['hyperparameters']['scheduler']['gamma'])
    logger.info(f'Total_params:{pytorch_total_params}')
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
    with open('/home/alonso/Documents/fundus_suitable/configs/drnetq.yaml', "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    train(cfg)