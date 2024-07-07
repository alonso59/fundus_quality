import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding),
                        nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out
  
class InceptionResidualBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_1x1,
        red_3x3,
        out_3x3,
        red_5x5,
        out_5x5,
        out_pool,
    ):
        super(InceptionResidualBlock, self).__init__()
        self.branch1 = ResidualBlock(in_channels, out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            ResidualBlock(in_channels, red_3x3, kernel_size=1, padding=0),
            ResidualBlock(red_3x3, out_3x3, kernel_size=3),
        )
        self.branch3 = nn.Sequential(
            ResidualBlock(in_channels, red_5x5, kernel_size=1),
            ResidualBlock(red_5x5, out_5x5, kernel_size=5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            ResidualBlock(in_channels, out_pool, kernel_size=1),
        )
    
    def forward(self, x):
        branches = (self.branch1, self.branch2, self.branch3, self.branch4)
        return torch.cat([branch(x) for branch in branches], 1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.dropout = nn.Dropout(p=0.7)
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = ResidualBlock(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class InceptionV1(nn.Module):
    def __init__(self, aux_logits=True, num_classes=1000):
        super(InceptionV1, self).__init__()
        self.aux_logits = aux_logits
        self.conv1 = ResidualBlock(
            in_channels=3, 
            out_chanels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
        )
        self.conv2 = ResidualBlock(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception3a = InceptionResidualBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionResidualBlock(256, 128, 128, 192, 32, 96, 64)
        self.inception4a = InceptionResidualBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionResidualBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionResidualBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionResidualBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionResidualBlock(528, 256, 160, 320, 32, 128, 128)
        self.inception5a = InceptionResidualBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionResidualBlock(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, num_classes)
        
        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        else:
            self.aux1 = self.aux2 = None
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool(x)
        x = self.inception4a(x)
        
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)
        
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)
        
        x = self.inception4e(x)
        x = self.maxpool(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        if self.aux_logits and self.training:
            return aux1, aux2, x
        return x