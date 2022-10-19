from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch
import torchvision
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
    
class Block(Module):
    def __init__(self,inChannels,outChannels):
        super().__init__()
        self.conv1 = Conv2d(inChannels,outChannels,3)
        self.relu = ReLU()
        self.conv2 = Conv2d(outChannels,outChannels,3)
        
    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))
    
class Encoder(Module):
    def __init__(self):
        super().__init__()
        self.channels = (3,64,128,256,512,1024)
        self.blocks = ModuleList([Block(self.channels[i],self.channels[i+1]) for i in range(len(self.channels)-1)])
        self.pool = MaxPool2d(2)
        
    def forward(self,x):
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
            x = self.pool(x)
        return features
    
class Decoder(Module):
    def __init__(self):
        super().__init__()
        self.channels = (1024,512,256,128,64)
        self.upconvs = ModuleList([ConvTranspose2d(self.channels[i],self.channels[i+1],2,2) for i in range(len(self.channels)-1)])
        self.blocks = ModuleList([Block(self.channels[i],self.channels[i+1]) for i in range(len(self.channels)-1)])
        
    def forward(self,x,features):
        for i in range(len(self.channels)-1):
            x = self.upconvs[i](x)
            _,_,H,W = x.shape
            encoded_features = torchvision.transforms.CenterCrop([H, W])(features[i])
            x = torch.cat([x,encoded_features],dim=1)
            x = self.blocks[i](x)
        return x
            
class UNet(Module):
    def __init__(self,num_classes=3):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.tail = Conv2d(64,num_classes,1)
        #self.softmax = torch.nn.Softmax(dim = 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.float()
        
    def forward(self,x):
        x = F.pad(x,(98,98,98,98),'constant',0)
        encoded_features = self.encoder(x)
        out = self.decoder(encoded_features[::-1][0], encoded_features[::-1][1:])
        out = self.tail(out)
        out = self.sigmoid(out)
        return torchvision.transforms.CenterCrop([1080, 1920])(out)
            
        