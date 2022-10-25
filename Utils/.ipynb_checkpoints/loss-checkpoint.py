import torch 
import torch.nn as nn
from torch.nn import functional as F

def dice_score(inputs,targets,activation,smooth=1):
    if activation is not None:
        inputs = activation(inputs)
    
    #inputs = inputs.view(-1)
    #targets = targets.view(-1)
        
    intersection = (inputs * targets).sum()                            
    return (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
    
    

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
 
    def forward(self, inputs, targets, smooth=1):        
        
        activation = torch.sigmoid       
        
        dice = dice_score(inputs,targets,activation)
        
        return 1 - dice
    
class BCEDiceLoss(DiceLoss):
    __name__ = 'bce_dice_loss'

    def __init__(self, lambda_dice=1.0, lambda_bce=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.lambda_dice=lambda_dice
        self.lambda_bce=lambda_bce

    def forward(self, y_pr, y_gt):
        dice = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return (self.lambda_dice*dice) + (self.lambda_bce* bce)