import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset 
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

class ToTensor(object):
    """Convert ndarrays of images and masks to Tensors."""

    def __call__(self, image, mask):
        # swap color axis
        image = image.transpose((1,2,0))
        return {'image': transforms.functional.to_tensor(image),
                'mask': torch.from_numpy(mask)}