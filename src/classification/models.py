import torch
import torch.nn as nn
import torchvision.models as models
from common.summary import summary
from .networks import drnetq, swin_transformer #, NAT 
from timm.models.inception_v4 import inception_v4
import timm
import sys
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
        if model_name == 'inceptionv4':
            self.model, self.layer_target = self.incepv4()
        if model_name == 'resnet18':
            self.model, self.layer_target = self.resnet18()
        if model_name == 'resnet152':
            self.model, self.layer_target = self.resnet152()
        if model_name == 'vgg13':
            self.model, self.layer_target = self.vgg13()
        if model_name == 'vgg19':
            self.model, self.layer_target = self.vgg19()
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
        model = models.inception_v3(pretrained=self.pretrain)
        layer_target = [model.Mixed_7c]
        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs, self.n_classes)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.n_classes)
        return model.to(self.device), layer_target

    def incepv4(self):
        model = timm.models.create_model('inception_resnet_v2', pretrained=True, num_classes=self.n_classes)
        # print(model)
        # layer_target = [model.features]
        # print(model)
        layer_target = [model.conv2d_7b]
        return model.to(self.device), layer_target

    def resnet18(self):      
        model = models.resnet18(pretrained=self.pretrain)
        layer_target = [model.layer4[-1]]
        model.fc = nn.Linear(512, self.n_classes)
        return model.to(self.device), layer_target

    def resnet152(self):      
        model = models.resnet152(pretrained=self.pretrain)
        layer_target = [model.layer4[-1]]
        model.fc = nn.Linear(2048, self.n_classes)
        return model.to(self.device), layer_target

    def vgg13(self):
        model = models.vgg13(pretrained=self.pretrain)
        layer_target = [model.features]
        model.classifier[6] = nn.Linear(4096, self.n_classes)
        return model.to(self.device), layer_target

    def vgg19(self):
        model = models.vgg19(pretrained=self.pretrain)
        layer_target = [model.features]
        model.classifier[6] = nn.Linear(4096, self.n_classes)
        return model.to(self.device), layer_target

    def DRNetq(self):
        model = drnetq.DRNetQ(n_classes=self.n_classes)  # create object model
        layer_target = [model.layers.relu8]
        self.is_inception = False 
        return model.to(self.device), layer_target
    
    def swin_custom(self):
        model = swin_transformer.SwinTransformer(
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
        layer_target = [model.stage4.layers[0][1].mlp_block.fn.norm]
        return model.to(self.device), layer_target

    def swin_tiny(self): 
        model = swin_transformer.swin_t()
        load = torch.load('pretrain/swint_ep300.pth')
        model.load_state_dict(load, strict=False)
        model.mlp_head = nn.Sequential(
        nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True),
        nn.Linear(in_features=768, out_features=self.n_classes, bias=True)
        )
        layer_target = [model.stage4.layers[0][1].mlp_block.fn.norm]
        return model.to(self.device), layer_target  

    def nat_mini(self): 
        model = NAT.nat_mini(pretrained=True, num_classes=self.n_classes)
        layer_target = [model.levels[-1].blocks[-1].norm1]
        return model.to(self.device), layer_target

    def nat_custom(self):
        model = NAT.nat_custom(num_classes=self.n_classes)
        layer_target = [model.levels[-1].blocks[-1].norm1]
        return model.to(self.device), layer_target

    """
    you can add your own network here
    .
    .
    .
    """
