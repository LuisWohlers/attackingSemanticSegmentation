from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch

#input: (N,C,H,W)
#output: (N,num_classes,H,W) 


class TestBlock(Module):
    def __init__(self,inChannels,outChannels):
        super().__init__()
        self.conv1 = Conv2d(inChannels,outChannels,(3,3),padding='same')
        self.relu = ReLU()
        self.conv2 = Conv2d(outChannels,outChannels,(3,3),padding='same')
        
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))
    
class Model(Module):
    def __init__(self,inChannels,outChannels,image_height,image_width):
        super().__init__()
        
        