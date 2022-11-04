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

    def __call__(self, image, mask):
        image = image.resize((1024,512))
        mask = mask.resize((1024,512))
        return {'image': transforms.functional.to_tensor(image),
                'mask': transforms.functional.to_tensor(mask).squeeze()*255}#torch.from_numpy(mask.squeeze(axis=0).transpose(2,0,1))}