import sys

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchmetrics.functional as M

def tensorboard_checkpoint(step, writer, train_loss, val_loss, train_iou, val_iou):
    results_loss = {'Train': train_loss, 'Validation': val_loss}
    results_acc = {'Train': train_iou, 'Validation': val_iou}
    writer.add_scalars("Loss", results_loss, step)
    writer.add_scalars("Accuracy", results_acc, step)


def fit(num_epochs, train_loader, val_loader,
        model, optimizer, loss_fn, scheduler,
        scaler, device, checkpoint_path):
    best_valid_loss = float("inf")
    hist_train_loss = np.zeros(num_epochs)
    hist_val_loss = np.zeros(num_epochs)
    hist_train_acc = np.zeros(num_epochs)
    hist_val_acc = np.zeros(num_epochs)
    """ Create log interface """
    writer = SummaryWriter()

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch+1}/{num_epochs}")
        train_loss, train_acc = train_fn(train_loader, model, optimizer, loss_fn, scaler, device)
        val_loss, val_acc = validation(model, val_loader, loss_fn, device)
        tensorboard_checkpoint(epoch, writer, train_loss, val_loss, train_acc, val_acc)
        scheduler.step()
        """ Saving the model """
        if val_loss < best_valid_loss:
            str_print = f"Valid loss improved from {best_valid_loss:2.4f} to {val_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            best_valid_loss = val_loss
            torch.save(model, checkpoint_path + '/model.pth')
            torch.save(model.state_dict(), checkpoint_path + "/weights.pth")
        else:
            str_print = f"Valid loss not improved: {best_valid_loss:2.4f}"
        hist_train_loss[epoch] = train_loss
        hist_val_loss[epoch] = val_loss
        hist_train_acc[epoch] = train_acc
        hist_val_acc[epoch] = val_acc
        print(f'--> Train acc: {train_acc:.4f} \tVal. acc: {val_acc:.4f}')
        print(f'--> Train Loss: {train_loss:.4f} \tVal. Loss: {val_loss:.4f}')
        print(str_print)
    np.save(checkpoint_path+'hist_train_loss.npy', hist_train_loss)
    np.save(checkpoint_path+'hist_val_loss.npy', hist_val_loss)
    np.save(checkpoint_path + 'hist_train_acc.npy', hist_train_acc)
    np.save(checkpoint_path + 'hist_val_acc.npy', hist_val_acc)


def train_fn(loader, model, optimizer, loss_fn, scaler, device):
    train_loss = 0.0
    train_acc = 0.0
    loop = tqdm(loader)
    model.train()

    for batch_idx, (x, y) in enumerate(loop):
        x = x.type(torch.float).to(device, non_blocking=True)
        y = y.type(torch.long).to(device, non_blocking=True)

        # forward
        # with torch.cuda.amp.autocast():
        #     y_pred = model(x)
        #     loss = loss_fn(y_pred, y)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        m0 = M.accuracy(y_pred, y)
        # print(f"{m0}:.4f")
        train_acc += m0.item()
        train_loss += loss.item()
        # update tqdm loop
        loop.set_postfix(loss=loss.item(), acc=m0.item())
    return train_loss / len(loader), train_acc / len(loader)


def validation(model, loader, loss_fn, device):
    valid_loss = 0.0
    valid_acc = 0.0
    loop = tqdm(loader)
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loop):
            x = x.requires_grad_(False).type(torch.float).to(device, non_blocking=True)
            y = y.requires_grad_(False).type(torch.long).to(device, non_blocking=True)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            valid_loss += loss.item()
            m0 = M.accuracy(y_pred, y)
            # print(f"{m0}:2.4f")
            valid_acc += m0.item()
            loop.set_postfix(loss=loss.item(), acc=m0.item())
    return valid_loss/len(loader), valid_acc/len(loader)
