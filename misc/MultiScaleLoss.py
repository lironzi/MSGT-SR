import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.vgg_perceptual_loss import VGGPerceptualLoss
class MultiScaleLoss(nn.Module):
    def __init__(self,device, alpha=0.1, beta=0.0,weights=[1,1,1]):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1 = nn.L1Loss()
        self.vgg_loss=VGGPerceptualLoss(device)
        self.weights=weights
    def forward(self, outputs, HRs):
        loss_x2 = self.weights[0]*self.l1(outputs[0], HRs[0])
        loss_x4 = self.weights[1]*self.l1(outputs[1], HRs[1])
        loss_main = self.weights[2]*self.l1(outputs[2], HRs[2]) + self.alpha * self.vgg_loss(outputs[2], HRs[2])
        loss = loss_main + loss_x2 + loss_x4
        return loss