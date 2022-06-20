import sys

import torch
import torch.nn.functional as F
import torch.nn as nn


class Accuracy(nn.Module):

    def __init__(self):
        super().__init__()

    @property
    def __name__(self):
        return "accuracy"

    def forward(self, y_pr, y_gt):
        y_pr = torch.sigmoid(y_pr.reshape(-1))
        y_pr = torch.round(y_pr)
        tp = torch.sum(y_pr == y_gt.reshape(-1))
        score = tp / y_gt.reshape(-1).shape[0]
        return score


class mIoU(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device
    @property
    def __name__(self):
        return "mIoU"

    def forward(self, logits, true, eps=1e-5):
        num_classes = logits.shape[1]
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot = true_1_hot.type(true.type()).to(self.device)
        probas = F.softmax(logits, dim=1).to(self.device)
        dims = (0, 2, 3)
        mult = (probas * true_1_hot).to(self.device)
        sum = (probas + true_1_hot).to(self.device)
        intersection = torch.sum(mult, dim=dims)
        cardinality = torch.sum(sum, dim=dims)
        union = cardinality - intersection
        iou_score = (intersection / (union + eps))
        # print(iou_score)
        return iou_score.cpu().detach().numpy()
