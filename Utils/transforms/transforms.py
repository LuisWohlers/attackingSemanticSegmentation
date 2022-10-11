import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset 
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

class ToTensor(object):

    def __call__(self, image, mask):
        return {'image': transforms.functional.to_tensor(image),
                'mask': torch.from_numpy(mask.squeeze().transpose(2,0,1))}