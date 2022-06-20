import torch
import torch.nn as nn
import torchvision.models as models
from common.summary import summary
from .networks import drnetq, swin_transformer, NAT

class ClassificationModels(nn.Module):
    def __init__(self, device, in_channels, img_size, n_classes=2, pretrain=True) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_channels
        self.img_size = img_size
        self.n_classes = n_classes
        self.pretrain = pretrain
    
    def summary(self, logger=None):
        summary(self.model, input_size=(self.in_channels, self.img_size, self.img_size), batch_size=-1, logger=logger)

    def model_builder(self, model_name):
        self.is_inception = False
        if model_name == 'inceptionv3':
            self.model, self.layer_target = self.inceptionv3()
            self.is_inception = True
        if model_name == 'resnet18':
            self.model, self.layer_target = self.resnet18()
        if model_name == 'vgg13':
            self.model, self.layer_target = self.vgg13()
        if model_name == 'drnetq':
            self.model, self.layer_target = self.DRNetq()
        if model_name == 'swin_custom':
            self.model, self.layer_target = self.swin_custom()
        if model_name == 'swin_tiny':
            self.model, self.layer_target = self.swin_tiny()
        if model_name == 'nat_mini':
            self.model, self.layer_target = self.nat_mini()
        if model_name == 'nat_custom':
            self.model, self.layer_target = self.nat_custom()
        return self.model, self.layer_target, self.is_inception

    def inceptionv3(self):
        self.model = models.inception_v3(pretrained=self.pretrain)
        self.layer_target = [self.model.Mixed_7c]  
        self.model.AuxLogits.fc = nn.Linear(768, self.n_classes)
        self.model.fc = nn.Linear(2048, self.n_classes)
        return self.model.to(self.device), self.layer_target

    def resnet18(self):      
        self.model = models.resnet18(pretrained=self.pretrain)
        self.layer_target = [self.model.layer4[-1]]
        self.model.fc = nn.Linear(512, self.n_classes)
        return self.model.to(self.device), self.layer_target

    def vgg13(self):
        self.model = models.vgg13(pretrained=self.pretrain)
        self.layer_target = [self.model.features]
        self.model.classifier[6] = nn.Linear(4096, self.n_classes)
        return self.model.to(self.device), self.layer_target

    def DRNetq(self):
        self.model = drnetq.DRNetQ(n_classes=self.n_classes)  # create object model
        self.layer_target = [self.model.layers.maxpool5]
        self.is_inception = False 
        return self.model.to(self.device), self.layer_target
    
    def swin_custom(self):
        self.model = swin_transformer.SwinTransformer(
                hidden_dim=24,
                layers=(2, 2, 2, 2),
                heads=(2, 2, 2, 2),
                channels=3,
                num_classes=2,
                head_dim=32,
                window_size=7,
                downscaling_factors=(2, 2, 2, 2),
                relative_pos_embedding=True
                )
        self.layer_target = [self.model.stage4.layers[0][1].mlp_block.fn.norm]
        return self.model.to(self.device), self.layer_target

    def swin_tiny(self): 
        self.model = swin_transformer.swin_t()
        load = torch.load('pretrain/swint_ep300.pth')
        self.model.load_state_dict(load, strict=False)
        self.model.mlp_head = nn.Sequential(
        nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True),
        nn.Linear(in_features=768, out_features=self.n_classes, bias=True)
        )
        self.layer_target = [self.model.stage4.layers[0][1].mlp_block.fn.norm]
        return self.model.to(self.device), self.layer_target  

    def nat_mini(self): 
        self.model = NAT.nat_mini(pretrained=True, num_classes=self.n_classes)
        self.layer_target = [self.model.levels[-1].blocks[-1].norm1]
        return self.model.to(self.device), self.layer_target

    def nat_custom(self):
        self.model = NAT.nat_custom(num_classes=self.n_classes)
        self.layer_target = [self.model.levels[-1].blocks[-1].norm1]
        return self.model.to(self.device), self.layer_target

    """
    you can add your own network here
    .
    .
    .
    """
