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
import gc

from Utils import loader, loss
from Train import train
from Model import Model

def FGSM_singleImage(model:torch.nn.Module=None, 
         loss:torch.nn.Module=None,
         img:torch.tensor=None, 
         target_mask:torch.tensor=None, 
         epsilon:float=0.5) -> torch.tensor:#-> tuple[torch.tensor,torch.tensor]:
    
    if model == None or img == None or target_mask == None or loss == None:
        return None,None
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    img, target_mask = img.unsqueeze(0).to(device), target_mask.unsqueeze(0).to(device)
    img.requires_grad = True
    out = model(img)
    
    
    loss = loss(out,target_mask)
    model.zero_grad()
    loss.backward()
    data_grad = img.grad.data
    sign_data_grad = data_grad.sign()
    perturbed_img = img - epsilon*sign_data_grad
    perturbed_img = torch.clamp(perturbed_img,0,1)
    return perturbed_img
    
def I_FGSM_singleImage(model:torch.nn.Module=None, 
         lossf:torch.nn.Module=None,
         img:torch.tensor=None, 
         target_mask:torch.tensor=None, 
         alpha:float=0.5,
         num_iters=50) -> [torch.tensor,torch.tensor]:
    if model == None or img == None or target_mask == None or lossf == None:
        return None,None
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #device = 'cpu'
    
    img, target_mask = img.unsqueeze(0).to(device), target_mask.unsqueeze(0).to(device)
    img.requires_grad = True
    img_ = img
    
    for i in range(num_iters):
        img_ = img_.detach().cpu().clone().to(device)
        img_.requires_grad_()
        img_.retain_grad()        
        #if i>0:
        #    loss.detach_()
            #loss.requires_grad_()
            #loss.retain_grad()
        #    out.detach_()
            #loss = torch.autograd.Variable(loss.data,requires_grad = True)
        out = model(img_)
        loss = lossf(out,target_mask)
        model.zero_grad()
        loss.backward(retain_graph = True)
        data_grad = img_.grad.data
        sign_data_grad = data_grad.sign()
        img_ = img_ - alpha*sign_data_grad
        img_ = torch.clamp(img_,0,1)
        del loss
        del out
        del data_grad
        del sign_data_grad
        torch.cuda.empty_cache()
        gc.collect()
        
        #time.sleep(0.1)
        
        
    perturbed_image = img_
    perturbation = img_-img
    
    return perturbed_image,perturbation

def Irestricted_FGSM_singleImage(model:torch.nn.Module=None, 
         lossf:torch.nn.Module=None,
         img:torch.tensor=None, 
         target_mask:torch.tensor=None, 
         alpha:float=0.5,
         num_iters=50) -> [torch.tensor,torch.tensor]:
    if model == None or img == None or target_mask == None or lossf == None:
        return None,None
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #device = 'cpu'
    
    img, target_mask = img.unsqueeze(0).to(device), target_mask.unsqueeze(0).to(device)
    img.requires_grad = True
    img_ = img
    prediction = model.predict(img_)[1][0][1]
    plt.imshow(prediction.detach().cpu().numpy())
    
    for i in range(num_iters):
        img_ = img_.detach().cpu().clone().to(device)
        img_.requires_grad_()
        img_.retain_grad()        
        out = model(img_)
        loss = lossf(out,target_mask)
        model.zero_grad()
        loss.backward(retain_graph = True)
        data_grad = img_.grad.data
        sign_data_grad = data_grad.sign()
        img_ = img_ - alpha*sign_data_grad*prediction
        img_ = torch.clamp(img_,0,1)
        del loss
        del out
        del data_grad
        del sign_data_grad
        torch.cuda.empty_cache()
        gc.collect()
        
        #time.sleep(0.1)
        
        
    perturbed_image = img_
    perturbation = img_-img
    
    return perturbed_image,perturbation
    
def I_FGSMLeastLikely_singleImage(model:torch.nn.Module=None, 
         lossf:torch.nn.Module=None,
         img:torch.tensor=None, 
         num_classes=3, 
         alpha:float=0.5,
         num_iters=50) -> [torch.tensor,torch.tensor]:
    if model == None or img == None or lossf == None:
        return None,None
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    img = img.unsqueeze(0).to(device)
    img.requires_grad = True
    img_ = img
    
    sm = torch.nn.Softmax(dim=1)
    
    for i in range(num_iters):
        img_.requires_grad_()
        img_.retain_grad()        
        out = model(img_)
        confidences = sm(out)
        target_mask = torch.argmin(confidences,dim=1)
        target_mask = [(target_mask == c).int() for c in range(num_classes)]
        target_mask = torch.stack(target_mask,dim=-1)
        target_mask = target_mask.permute(0,3,1,2).float()
        loss = lossf(out,target_mask)
        model.zero_grad()
        loss.backward(retain_graph = True)
        data_grad = img_.grad.data
        sign_data_grad = data_grad.sign()
        img_ = img_ - alpha*sign_data_grad
        img_ = torch.clamp(img_,0,1)
        del loss
        del out
        torch.cuda.empty_cache()
        
    perturbed_image = img_
    perturbation = img_-img
    
    return perturbed_image,perturbation

def atanh(x, eps=1e-6):
    x = x*(1-eps)
    return 0.5 * torch.log((1.0+x)/(1.0-x))

def to_tanh_space(x,box):
    return atanh((x - (box[1]+box[0])*0.5) / (boc[1]-box[0])*0.5)

def from_tanh_space(x,box):
    return torch.tanh(x)*(boc[1]-box[0])*0.5) + (box[1]+box[0])*0.5)

def CarliniWagner(model:torch.nn.Module=None, 
         lossf:torch.nn.Module=None,
         img:torch.tensor=None, 
         target_mask:torch.tensor=None, 
         alpha:float=0.5,
         num_iters=50) -> [torch.tensor,torch.tensor]:
    if model == None or img == None or target_mask == None or lossf == None:
        return None,None
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #device = 'cpu'
    
    img, target_mask = img.unsqueeze(0).to(device), target_mask.unsqueeze(0).to(device)
    
    