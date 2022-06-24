import torch
import math
from tqdm import tqdm
from .callbacks import TensorboardWriter
from decimal import Decimal
import datetime
from torchmetrics import Accuracy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
from torch.nn.functional import softmax
from sklearn.metrics import roc_curve, auc
import sys


def trainer(num_epochs,
            train_loader,
            val_loader,
            model,
            optimizer,
            loss_fn,
            metric,
            device,
            checkpoint_path,
            scheduler,
            iter_plot_img,
            name_model,
            callback_stop_value,
            tb_dir,
            logger,
            layer,
            is_inception
            ):
    """ Create log interface """
    writer = TensorboardWriter(metric=metric, name_dir=tb_dir + 'tb_' + name_model + '/')
    iter_num = 0.0
    iter_val = 0.0
    stop_early = 0
    best_valid_loss = float("inf")

    # img_sample, labels = next(iter(val_loader))

    for epoch in range(num_epochs):
        lr_ = optimizer.param_groups[0]["lr"]
        str = f"Epoch: {epoch+1}/{num_epochs} --model:{name_model} --lr:{lr_:.3e}"
        logger.info(str)
        train_loss, train_metric, iter_num, lr_ = train(train_loader,
                                                        model,
                                                        writer,
                                                        optimizer,
                                                        loss_fn,
                                                        iter_num,
                                                        device,
                                                        is_inception,
                                                        layer,
                                                        )

        scheduler.step()
        # pick0 = np.random.randint(0, len(img_sample))
        # if epoch % 5 == 0:
        #     print('Saving examples in TensorBoard....')
        #     tb_save_images_figures(model, img_sample[pick0, :, :, :].float().to(device), writer, epoch, device, layer)

        val_loss, val_metric, iter_val = validation(model, val_loader, loss_fn, writer, iter_val, device)

        writer.per_epoch(train_loss=train_loss, val_loss=val_loss,
                         train_metric=train_metric, val_metric=val_metric, step=epoch)

        """ Saving the model """
        if val_loss < best_valid_loss:
            str_print = f"Valid loss improved from {best_valid_loss:2.4f} to {val_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            best_valid_loss = val_loss
            torch.save(model, checkpoint_path + f'model.pth')
            torch.save(model.state_dict(), checkpoint_path + f'/weights.pth')
            stop_early = 0
        else:
            stop_early += 1
            str_print = f"Valid loss not improved: {best_valid_loss:2.4f}, ESC: {stop_early}/{callback_stop_value}"
        if stop_early == callback_stop_value:
            logger.info('+++++++++++++++++ Stop training early +++++++++++++')
            break
        logger.info(f'----> Train Acc: {train_metric:.4f} \t Val. Acc: {val_metric:.4f}')
        logger.info(f'----> Train Loss: {train_loss:.4f} \t Val. Loss: {val_loss:.4f}')
        logger.info(str_print)
    print('Finishing train....')
    load_best_model = torch.load(checkpoint_path + 'model.pth')
    # tb_save_images_figures(model, img_sample[pick0, :, :, :].float().to(device), writer, 0, device, layer)
    tb_save_CM_roc_auc(model, val_loader, writer, device)
    


def train(loader, model, writer, optimizer, loss_fn, iter_num, device, is_inception, layer):
    train_loss = 0.0
    train_iou = 0.0
    loop = tqdm(loader, ncols=120)
    acc = Accuracy()
    model.train()
    for batch_idx, data  in enumerate(loop):
        x = data[0]['image'].type(torch.float).to(device)
        y = data[1].type(torch.long).to(device)
        # forward
        # ------------ is inception ------------
        if is_inception :
            y_pred, aux_outputs = model(x)
            loss1 = loss_fn(y_pred, y)
            loss2 = loss_fn(aux_outputs, y)
            loss = loss1 + 0.4*loss2
        # ------------ is inception ------------
        else:
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
        m0 = acc(y_pred.detach().cpu(), y.detach().cpu())
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # accumulate metrics and loss items
        train_iou += m0.item()
        train_loss += loss.item()
        # update tqdm loop
        loop.set_postfix(Acc=m0.item(), loss=loss.item())
        # tensorboard callbacks
        writer.per_iter(loss.item(), m0.item(), iter_num, name='Train')
        writer.learning_rate(optimizer.param_groups[0]["lr"], iter_num)
        # print(iter_num)
        if iter_num % len(loader)*20 == 0.0:
            print('Save image tensorboard...')
            preds = torch.softmax(y_pred[0, :], dim=0)
            preds = torch.argmax(preds).item()
            writer.save_img_preds(model, layer, x[0, :, :, :], preds, iter_num, device)
        iter_num = iter_num + 1
    return train_loss / len(loader), train_iou / len(loader), iter_num, optimizer.param_groups[0]["lr"]


def validation(model, loader, loss_fn, writer, iter_val, device):
    valid_loss = 0.0
    valid_iou = 0.0
    loop = tqdm(loader, ncols=120)
    model.eval()
    acc = Accuracy()
    with torch.no_grad():
        for batch_idx, data in enumerate(loop):
            x = data[0]['image'].type(torch.float).to(device)
            y = data[1].type(torch.long).to(device)
            y_pred= model(x)
            m0 = acc(y_pred.detach().cpu(), y.detach().cpu())
            loss = loss_fn(y_pred, y)
            # accumulate metrics and loss items
            valid_iou += m0.item()
            valid_loss += loss.item()
            # update tqdm
            loop.set_postfix(Acc=m0.item(), loss=loss.item())
            # tensorboard callbacks
            writer.per_iter(loss.item(), m0.item(), iter_val, name='Val')
            iter_val += 1

    return valid_loss / len(loader), valid_iou / len(loader), iter_val


def eval(model, loader, loss_fn, device):
    eval_metric = 0.0
    eval_loss = 0.0
    loop = tqdm(loader, ncols=120)
    model.eval()
    acc = Accuracy()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loop):
            x = x.type(torch.float).to(device)
            y = y.type(torch.long).to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            m0 = acc(y_pred.detach().cpu(), y.detach().cpu())
            # accumulate metrics and loss items
            eval_metric += m0.item()
            eval_loss += loss.item()
            # update tqdm
            loop.set_postfix(metric=m0.item(), loss=loss.item())

    return eval_loss / len(loader), eval_metric / len(loader)


def tb_save_images_figures(net, img, writer, step, device, layer):
    net.eval()
    pred = net(img.unsqueeze(0))  # Feed Network
    pred = (torch.max(torch.exp(pred), 1)[1]).data.cpu().numpy()
    writer.save_img_preds(net, layer, img.unsqueeze(0), pred, step, device)

def tb_save_CM_roc_auc(net, loader, writer, device):
    net.eval()
    y_pred = []  # save predction
    y_true = []  # save ground truth
    loop = tqdm(loader, ncols=120)
    print('Saving confusion matrix example in TensorBoard....')
    for _, (inputs, labels) in enumerate(loop):
        inputs = inputs.type(torch.float).to(device)
        labels = labels.type(torch.long).to(device)
        output = net(inputs)  # Feed Network
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)  # save prediction
        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # save ground truth
    # constant for classes
    classes = ('Low-Quality', 'High-Quality')
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes],
                         columns=[i for i in classes])
    pd.set_option('display.float_format', lambda x: '%.0f' % x)
    plt.figure()
    writer.save_figure('Confusion Matrix', sn.heatmap(df_cm, annot=True).get_figure(), step=0)

    fpr, tpf, _ = roc_curve(y_true, y_pred)
    auc_ = auc(fpr, tpf)
    fig = plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpf, label='AUC = {:.3f}'.format(auc_))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    writer.save_figure('ROC-AUC', fig, step=0)
