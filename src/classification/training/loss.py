import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import sys

class weighted_cross_entropy_generalized_dice_loss(nn.Module):
    def __init__(self, class_weights):
        super(weighted_cross_entropy_generalized_dice_loss, self).__init__()

        self.class_weights = torch.tensor(class_weights).float().to("cuda")
        self.CE = nn.CrossEntropyLoss(weight=self.class_weights)

    @property
    def __name__(self):
        return "entropy_loss"

    def forward(self, inputs, targets, eps=1e-7):
        num_classes = inputs.shape[1]
        dice_loss = np.zeros(3)
        w = torch.ones(inputs.shape).type(inputs.type()).to("cuda")

        for c in range(inputs.shape[1]):
            w[:, c, :, :] = self.class_weights[c]
        # One Hot ground thrut
        true_1_hot = torch.eye(num_classes)[targets.squeeze(1).long()]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()

        true_1_hot = true_1_hot.type(inputs.type())
        # Getting probabilities
        probas = F.softmax(inputs, dim=1)

        # Compute DiceLoss
        mult = probas * true_1_hot
        dims = (0, 2, 3)
        intersection = 2 * torch.sum(mult, dim=(0, 2, 3)) + eps
        cardinality = torch.sum(probas, dim=dims) + torch.sum(true_1_hot, dim=dims) + eps
        dice_loss = 1 - (intersection / cardinality).mean()
        # print((intersection / cardinality))
        # Compute CrossEntropy
        target1 = targets.squeeze(1).long()
        cross = self.CE(inputs, target1)

        return dice_loss #* 0.3 + cross


class DiceLoss(nn.Module):
    def __init__(self, device):
        super(DiceLoss, self).__init__()
        self.device = device
    @property
    def __name__(self):
        return "dice_loss"

    def forward(self, inputs, targets, eps=1e-7):
        num_classes = inputs.shape[1]
        # One Hot ground thrut
        true_1_hot = torch.eye(num_classes)[targets.squeeze(1).long()]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot = true_1_hot.type(inputs.type()).to(self.device)
        # Getting probabilities
        probas = F.softmax(inputs, dim=1).to(self.device)
        # Compute DiceLoss
        mult = (probas * true_1_hot).to(self.device)

        dims = (0, 2, 3)
        intersection = 2 * torch.sum(mult, dim=(0, 2, 3)) + eps
        cardinality = torch.sum(probas, dim=dims) + torch.sum(true_1_hot, dim=dims) + eps
        dice_score = 1 - (intersection / cardinality).mean()
        return dice_score
