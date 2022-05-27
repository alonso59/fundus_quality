import torch.nn as nn

class DRNetQTImageNet(nn.Module):
    def __init__(self, my_pretrained_model):
        super(DRNetQTImageNet, self).__init__()
        self.pretrained = my_pretrained_model
        self.new_layers = nn.Sequential(
                                        nn.Linear(128, 2),
                                    )
    
    def forward(self, x):
        x = self.pretrained(x)
        x = self.new_layers(x)
        return x