from torch.nn import BCEWithLogitsLoss, BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os
import numpy as np

from Utils import loader, loss
from Train import train
from Model import Model

def FGSM_singleImage(model:torch.nn.Module=None, 
         loss:torch.nn.Module=None,
         img:torch.tensor=None, 
         target_mask:torch.tensor=None, 
         eps:float=0.5) -> tuple[torch.tensor,torch,tensor]:
    
    if model == None or img == None or target_mask == None or loss == None:
        return None,None
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch device('cpu')
    img, mask = img.unsqueeze(0).to(device), mask.unsqueeze(0).to(device)
    pred = model(img)

    
    
    model.zero_grad()
    loss = loss(pred,target_mask)
    loss.backward()
    
    
    
    