import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch
from collections import OrderedDict


class DRNetQ(nn.Module):
    def __init__(self, n_classes):
        super(DRNetQ, self).__init__()

        self.layers = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)),
            ('relu1',nn.ReLU()),
            ('maxpool1', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('bn1', nn.BatchNorm2d(32)),
            ('conv2', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)),
            ('relu2',nn.ReLU()),
            ('conv3', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)),
            ('relu3',nn.ReLU()),
            ('maxpool2', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('bn2', nn.BatchNorm2d(64)),
            ('conv4', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)),
            ('relu4',nn.ReLU()),
            ('conv5', nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)),
            ('relu5',nn.ReLU()),
            ('maxpool3', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('bn3', nn.BatchNorm2d(64)),
            ('conv6', nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)),
            ('relu6',nn.ReLU()),
            ('conv7', nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)),
            ('relu7',nn.ReLU()),
            ('maxpool4', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('bn4', nn.BatchNorm2d(32)),
            ('conv8', nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)),
            ('relu8',nn.ReLU()),
            ('maxpool5', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('bn5', nn.BatchNorm2d(16)),
            ('flatten', nn.Flatten()),
            ('fc1', nn.Linear(in_features=784, out_features=512)),
            ('relufc',nn.ReLU()),
            ('fc2', nn.Linear(in_features=512 , out_features=n_classes)),
        ]))

    @property
    def __name__(self):
        return "QRetiNet"

    def forward(self, x):
        out = self.layers(x)
        return out



def test():
    model = DRNetQ(n_classes=2).to(torch.device('cuda'))
    summary(model, input_size=(3, 224, 224), batch_size=-1)


if __name__ == "__main__":
    test()
