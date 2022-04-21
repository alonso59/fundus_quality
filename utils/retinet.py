import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch
from collections import OrderedDict


class QRetiNet(nn.Module):
    def __init__(self, n_classes):
        super(QRetiNet, self).__init__()

        self.layers = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=0)),
            ('bn1', nn.BatchNorm2d(32)),
            ('relu1', nn.ReLU()),
            ('maxpool1', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv2', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0)),
            ('conv3', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=0)),
            ('bn2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU()),
            ('maxpool2', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv4', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=0)),
            ('conv5', nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=0)),
            ('bn3', nn.BatchNorm2d(64)),
            ('relu3', nn.ReLU()),
            ('maxpool3', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv6', nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=0)),
            ('bn4', nn.BatchNorm2d(32)),
            ('relu4', nn.ReLU()),
            ('maxpool4', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('flatten', nn.Flatten()),
            ('fc1', nn.Linear(in_features=2048, out_features=128)),
            ('relu5', nn.ReLU()),
            ('fc2', nn.Linear(in_features=128 , out_features=n_classes)),
        ]))

    @property
    def __name__(self):
        return "QRetiNet"

    def forward(self, x):
        out = self.layers(x)
        return out



def test():
    model = QRetiNet(n_classes=2).to(torch.device('cuda'))
    summary(model, input_size=(3, 224, 224), batch_size=-1)


if __name__ == "__main__":
    test()
