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

from typing import Union, Callable, Tuple

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

def PGD_batch(model:torch.nn.Module=None, 
         lossf:torch.nn.Module=None,
         img_batch:torch.tensor=None, 
         mask_batch:torch.tensor=None,
         target_mask_batch:torch.tensor=None, 
         num_iters:int=50,
         step_norm:Union[str,float]='inf',
         step_size=1.0,
         eps:float=0.0,
         eps_norm:Union[str,float]='inf',
         clamp:tuple=(0,1)) -> [torch.tensor,torch.tensor]:
    if model == None or img_batch == None or target_mask_batch == None or lossf == None:
        return None,None
    ### https://towardsdatascience.com/know-your-enemy-7f7c5038bdf3
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #device = 'cpu'
    
    img_batch, target_mask_batch = img_batch.to(device), target_mask_batch.to(device)
    #img_batch.requires_grad = True
    
    img_batch_adv = img_batch.clone().detach().requires_grad_(True).to(device)
    targeted = target_mask_batch is not None
    num_channels = img_batch.shape[1]
    
    for i in range(num_iters):
        _img_adv = img_batch_adv.clone().detach().requires_grad_(True)
        
        prediction = model(_img_adv)
        loss = lossf(prediction,target_mask_batch if targeted else mask_batch)
        loss.backward()
        
        with torch.no_grad():
            if step_norm == 'inf':
                gradients = _img_adv.grad.sign() * step_size
            else:
                gradients = _img_adv.grad * step_size / _img_adv.grad.view(_img_adv.shape[0],-1).norm(step_norm,dim=-1).view(-1,1,1,1)
                
            if targeted:
                img_batch_adv -= gradients
            else:
                img_batch_adv += gradients
                
        if eps_norm == 'inf':
            img_batch_adv = torch.max(torch.min(img_batch_adv,img_batch+eps),img_batch-eps)
        else:
            delta = img_batch_adv - img_batch
            
            mask = delta.view(delta.shape[0], -1).norm(eps_norm, dim=1) <= eps
            
            scaling_factor = delta.view(delta.shape[0],-1).norm(eps_norm,dim=1)
            scaling_factor[mask] = eps
            
            delta *= eps/scaling_factor.view(-1,1,1,1)
            
            img_batch_adv = img_batch + delta
            
        img_batch_adv = img_batch_adv.clamp(*clamp)
        
        del loss
        torch.cuda.empty_cache()
        gc.collect()
        
    return img_batch_adv.detach()
                
            
        
    

#def atanh(x, eps=1e-6):
#    x = x*(1-eps)
#    return 0.5 * torch.log((1.0+x)/(1.0-x))

#def to_tanh_space(x,box):
#    return atanh((x - (box[1]+box[0])*0.5) / (boc[1]-box[0])*0.5)

#def from_tanh_space(x,box):
#    return torch.tanh(x)*((box[1]-box[0])*0.5) + ((box[1]+box[0])*0.5)

#def CarliniWagner(model:torch.nn.Module=None, 
#         lossf:torch.nn.Module=None,
#         img:torch.tensor=None, 
#         target_mask:torch.tensor=None, 
#         alpha:float=0.5,
#         num_iters=50) -> [torch.tensor,torch.tensor]:
#    if model == None or img == None or target_mask == None or lossf == None:
#        return None,None
#    
#    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#    #device = 'cpu'
#    
#    img, target_mask = img.unsqueeze(0).to(device), target_mask.unsqueeze(0).to(device)
 
    
#class AdversarialTransformationNetwork():
    
    