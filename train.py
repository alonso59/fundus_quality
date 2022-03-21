import sys
import torch
import datetime
import numpy as np
import torch.nn as nn
from utils.fit import fit
from utils.retinet import QRetiNet
from utils.data import create_data
from utils.settings import Settings
from torchsummary import summary
from matplotlib import pyplot as plt
from utils.utils import create_dir, seeding
from torch.optim.lr_scheduler import StepLR
import torchmetrics.functional as M
def main():
    """ Seeding """
    seeding(42)
    """ Configuration parameters """
    settings = Settings()
    dataset_dir = settings.dataset_dir
    lr = settings.learning_rate
    batch_size = settings.batch_size
    num_epochs = settings.epochs
    image_size = settings.image_size
    checkpoint_path = "checkpoints/" + datetime.datetime.now().strftime("%d_%H-%M_QRetiNet/")
    gpus_ids = settings.gpus_ids
    step_sch = num_epochs * 0.15
    gamma_sch = 0.8
    train_dir = dataset_dir + '/train/'
    val_dir = dataset_dir + '/val/'
    """ Directories """
    create_dir("files")
    create_dir("checkpoints")
    create_dir(checkpoint_path)
    """ CUDA device """
    device = torch.device("cuda")
    """ Dataset and loader """
    train_loader, val_loader = create_data(dataset_dir, image_size, batch_size)
    """ Building model """
    model = QRetiNet()
    model = model.to(device)
    summary(model, input_size=(3, image_size, image_size), batch_size=batch_size)
    model = nn.DataParallel(model, device_ids=gpus_ids)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    """ Prepare training """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer=optimizer, step_size=step_sch, gamma=gamma_sch)
    scaler = torch.cuda.amp.GradScaler()
    # metric = M.accuracy()
    """ Save params """
    with open(checkpoint_path + "LOGS.txt", "w") as text_file:
        text_file.write(f"Learning rate: {lr}\n")
        text_file.write(f"Epochs: {num_epochs}\n")
        text_file.write(f"Scheduler step, gamma: {step_sch, gamma_sch}\n")
        text_file.write(f"Batch size: {batch_size}\n")
        text_file.write(f"No. of Parameters: {pytorch_total_params}\n")
        text_file.write(f"No. of GPUs: {len(gpus_ids)}\n")
        text_file.close()
    """ Training the model """
    fit(num_epochs=num_epochs, train_loader=train_loader, val_loader=val_loader,
        model=model, optimizer=optimizer, loss_fn=loss_fn,
        scheduler=scheduler,
        scaler=scaler, device=device, checkpoint_path=checkpoint_path)

    """ Saving numpy history """
    hist_train_loss = np.load(checkpoint_path + 'hist_train_loss.npy')
    hist_val_loss = np.load(checkpoint_path + 'hist_val_loss.npy')
    hist_train_acc = np.load(checkpoint_path + 'hist_train_acc.npy')
    hist_val_acc = np.load(checkpoint_path + 'hist_val_acc.npy')
    x = np.arange(1, len(hist_train_loss) + 1)

    with open(checkpoint_path + "results.txt", "w") as ff:
        ff.write(f"Best Train Loss: {np.min(hist_train_loss):0.4f}, Best Val Loss:{np.min(hist_val_loss):0.4f},\n "
                 f"Best Train IoU:{np.max(hist_train_acc):0.4f}, Best Val IoU:{np.max(hist_val_acc):0.4f}\n")

    plt.figure()
    plt.plot(x, hist_train_loss)
    plt.plot(x, hist_val_loss)
    plt.title("QRetiNet Loss")
    plt.grid(color='lightgray', linestyle='-', linewidth=2)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='best')
    plt.savefig(checkpoint_path + "loss.png")

    plt.figure()
    plt.plot(x, hist_train_acc)
    plt.plot(x, hist_val_acc)
    plt.title("QRetiNet Accuracy")
    plt.grid(color='lightgray', linestyle='-', linewidth=2)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='best')
    plt.savefig(checkpoint_path + "IoU.png")

if __name__ == '__main__':
    main()
