from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import BatchNorm2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torch.nn import Dropout
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch
import torchvision
#input: (N,C,H,W)
#output: (N,num_classes,H,W) 

class Block(Module):
    def __init__(self,inChannels,outChannels,p_dropout=0.05):
        super().__init__()
        self.bnorm1 = BatchNorm2d(inChannels)
        self.conv1 = Conv2d(inChannels,outChannels,3)
        self.relu = ReLU()
        self.bnorm2 = BatchNorm2d(outChannels)
        self.conv2 = Conv2d(outChannels,outChannels,3)
        #self.bnorm3 = BatchNorm2d(outChannels)
        
        self.dropout = Dropout(p_dropout)
        
    def forward(self, x):
        x = self.bnorm1(x)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.bnorm2(x)
        x = self.conv2(x)
        x = self.dropout(x)
        #x = self.bnorm3(x)
        return x#self.bnorm3(self.conv2(self.bnorm2(self.relu(self.conv1(self.bnorm1(x))))))
    
class Encoder(Module):
    def __init__(self,channels,p_dropout=0.05):
        super().__init__()
        self.channels = channels#(3,64,128,256,512,1024)
        self.blocks = ModuleList([Block(self.channels[i],self.channels[i+1],p_dropout) for i in range(len(self.channels)-1)])
        self.pool = MaxPool2d(2)
        
    def forward(self,x):
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
            x = self.pool(x)
        return features
    
class Decoder(Module):
    def __init__(self,channels,p_dropout=0.05):
        super().__init__()
        self.channels = channels#(1024,512,256,128,64)
        self.upconvs = ModuleList([ConvTranspose2d(self.channels[i],self.channels[i+1],2,2) for i in range(len(self.channels)-1)])
        self.blocks = ModuleList([Block(self.channels[i],self.channels[i+1],p_dropout) for i in range(len(self.channels)-1)])
        
    def forward(self,x,features):
        for i in range(len(self.channels)-1):
            x = self.upconvs[i](x)
            _,_,H,W = x.shape
            encoded_features = torchvision.transforms.CenterCrop([H, W])(features[i])
            x = torch.cat([x,encoded_features],dim=1)
            x = self.blocks[i](x)
        return x
            
class UNet(Module):
    def __init__(self,num_classes=3,enc_channels=(3,64,128,256,512,1024),dec_channels=(1024,512,256,128,64),img_size=(512,1024),p_dropout=0.05):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = Encoder(enc_channels,p_dropout)
        self.decoder = Decoder(dec_channels,p_dropout)
        self.tail = Conv2d(self.decoder.channels[-1],num_classes,1)
        self.sigmoid = torch.nn.Sigmoid()
        self.img_size = img_size
        self.float()
        
    def forward(self,x):
        #x = F.pad(x,(98,98,98,98),'reflect')
        encoded_features = self.encoder(x)        
        out = self.decoder(encoded_features[::-1][0], encoded_features[::-1][1:])
        out = self.tail(out)
        #out = self.sigmoid(out)
        out = F.interpolate(out,self.img_size)
        #out = self.sigmoid(out)
        return out
    
    def predict(self,x):        
        raw = self.forward(x)
        sm = torch.nn.Softmax(dim=1)
        confidences = sm(raw)
        segmentation = torch.argmax(confidences,dim=1)
        channels = [(segmentation == c).int() for c in range(self.num_classes)]
        segmentation_channels = torch.stack(channels,dim=-1)
        segmentation_channels = segmentation_channels.permute(0,3,1,2)
        return segmentation, segmentation_channels, confidences, raw
            
        