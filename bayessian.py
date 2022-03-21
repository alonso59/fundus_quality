import sys
import torch
import datetime
import numpy as np
import torch.nn as nn
from utils.unet import UNET
from utils.metrics import IoU
from torchsummary import summary
from utils.utils import create_dir
from utils.data import create_data
from utils.settings import Settings
from utils.fit import fit, validation
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import StepLR
from ax.service.managed_loop import optimize
from utils.loss import DiceLoss
""" Configuration parameters """
settings = Settings()
images_path = settings.images_path
masks_path = settings.masks_path
dataset_dir = settings.dataset_dir
train_size = settings.train_size
data_augmentation = settings.data_augmentation
gpus_ids = settings.gpus_ids
bilinear = False
""" ************************ """
""" Directories """
create_dir("files")
create_dir("checkpoints")
checkpoint_path = "checkpoints/" + datetime.datetime.now().strftime("%d_%H-%M_UNET_OCT/")
create_dir(checkpoint_path)

""" *********** """
""" CUDA device """
device = torch.device(f"cuda:{gpus_ids[0]}" if torch.cuda.is_available() else "cpu")
""" *********** """
exp = 0

def experiment(parameters):
    batch_size = parameters.get("batchsize", 32)
    num_layers = parameters.get("num_layers", 4)
    features_start = parameters.get("features_start", 16)
    lr = parameters.get("lr", 0.001)
    weight_decay = parameters.get("weight_decay", 0.0)
    gamma_sch = parameters.get("gamma_sch", 0.8)
    num_epochs = parameters.get("num_epochs", 500)
    step_sch = parameters.get("step_sch", num_epochs*0.3)
    folder = checkpoint_path + str(np.random.randint(0, 2000)) + "/"
    create_dir(folder)
    """ Dataset and loader """
    train_loader, val_loader = create_data(images_path, masks_path, dataset_dir,
                                           train_size, data_augmentation,
                                           batch_size=batch_size
                                           )
    """ Building model """
    model = UNET(num_classes=3,
                              input_channels=1,
                              num_layers=4,
                              features_start=features_start,
                              bilinear=False,
                              dropout=False,
                              dp=0.5,
                              kernel_size=(5, 5),
                              padding=2
                              )
    model = model.to(device)
    summary(model, input_size=(1, 128, 128), batch_size=batch_size)
    model = nn.DataParallel(model, device_ids=gpus_ids)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    """ Prepare training """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = DiceLoss(device)
    scheduler = StepLR(optimizer=optimizer, step_size=step_sch, gamma=gamma_sch)
    scaler = torch.cuda.amp.GradScaler()
    metrics = IoU(device)
    """ Save params """
    with open(folder + "LOGS.txt", "w") as text_file:
        text_file.write(f"Learning rate: {lr}\n")
        text_file.write(f"weight_decay: {weight_decay}\n")
        text_file.write(f"Epochs: {num_epochs}\n")
        text_file.write(f"Scheduler step, gamma: {step_sch, gamma_sch}\n")
        text_file.write(f"Batch size: {batch_size}\n")
        text_file.write(f"Metric: {metrics.__name__}\n")
        text_file.write(f"No. of Layers: {num_layers}\n")
        text_file.write(f"Features start: {features_start}\n")
        text_file.write(f"No. of Parameters: {pytorch_total_params}\n")
        text_file.write(f"Bilinear decoder?: {bilinear}\n")
        text_file.write(f"No. of GPUs: {len(gpus_ids)}\n")
        text_file.close()
    """ Training the model """
    fit(num_epochs=num_epochs, train_loader=train_loader, val_loader=val_loader,
        model=model, optimizer=optimizer, loss_fn=loss_fn,
        metric=metrics, scheduler=scheduler,
        scaler=scaler, device=device, checkpoint_path=folder)
    """ Saving numpy history """
    hist_train_loss = np.load(folder + 'hist_train_loss.npy')
    hist_val_loss = np.load(folder + 'hist_val_loss.npy')
    hist_train_IoU = np.load(folder + 'hist_train_IoU.npy')
    hist_val_IoU = np.load(folder + 'hist_val_IoU.npy')
    x = np.arange(1, len(hist_train_loss) + 1)
    with open(folder + "results.txt", "w") as ff:
        ff.write(f"Best Train Loss: {np.min(hist_train_loss):0.4f}, Best Val Loss:{np.min(hist_val_loss):0.4f},\n "
                 f"Best Train IoU:{np.max(hist_train_IoU):0.4f}, Best Val IoU:{np.max(hist_val_IoU):0.4f}\n")
    plt.figure()
    plt.plot(x, hist_train_loss)
    plt.plot(x, hist_val_loss)
    plt.title("UNet Loss")
    plt.grid(color='lightgray', linestyle='-', linewidth=2)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='best')
    plt.savefig(folder + "loss.png")

    plt.figure()
    plt.plot(x, hist_train_IoU)
    plt.plot(x, hist_val_IoU)
    plt.title("UNet IoU")
    plt.grid(color='lightgray', linestyle='-', linewidth=2)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='best')
    plt.savefig(folder + "IoU.png")
    load_best_model = torch.load(folder + '/model.pth')

    _, iou_eval = validation(load_best_model, val_loader, loss_fn, metrics, device)
    return iou_eval


best_parameters, values, exp, model = optimize(
    parameters=[
        {"name": "lr", "type": "range", "bounds": [1e-3, 0.5], "log_scale": True},
        {"name": "batchsize", "type": "range", "bounds": [64, 256]},
        # {"name": "weight_decay", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "num_layers", "type": "range", "bounds": [3, 5]},
        # {"name": "num_epochs", "type": "range", "bounds": [300, 500]},
    ],
    total_trials=25,
    evaluation_function=experiment,
    objective_name='iou',
)

print(best_parameters)
print(exp)
means, covariances = values
print(means)
print(covariances)