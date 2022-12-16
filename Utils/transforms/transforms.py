import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset 
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import PIL

class ToTensor(object):

    def __call__(self, image, mask, img_size, class_values):
        image = image.resize(img_size)
        
        masks = transforms.functional.to_tensor(mask).squeeze()*255

        mask = [transforms.functional.to_tensor(transforms.functional.to_pil_image((masks==c).int()).resize(img_size))\
                for c in class_values]
        
        
        
        mask = torch.stack(mask,dim=-1).squeeze()
        
        mask = mask.permute(2,0,1)

        return {'image': transforms.functional.to_tensor(image),
                'mask': mask.float()}