import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset 
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from .transforms import transforms as Tr

class dataset(Dataset):
    
    classlist = ['road',
            'sidewalk',
            'construction',
            'tram-track',
            'fence',
            'pole',
            'traffic-light',
            'traffic-sign',
            'vegetation',
            'terrain',
            'sky',
            'human',
            'rail-track',
            'car',
            'truck',
            'trackbed',
            'on-rails',
            'rail-raised',
            'rail-embedded']
    
    def __init__(self, imdir, maskdir, numstart=0, numend=-2, classes=None,transforms=Tr.ToTensor()):
        self.image_filenames = sorted(os.listdir(imdir))[numstart:numend+1]
        self.mask_filenames = sorted(os.listdir(maskdir))[numstart:numend+1]
        self.image_filedirections = [os.path.join(imdir,name) for name in self.image_filenames]
        self.mask_filedirections = [os.path.join(maskdir,name) for name in self.mask_filenames]
        self.class_values = [self.classlist.index(c.lower()) for c in classes]
        self.transforms = transforms
        
    def __getitem__(self,i):
        image = Image.open(self.image_filedirections[i])
        mask = np.array(Image.open(self.mask_filedirections[i]))[...,np.newaxis].transpose((2,0,1))
        
        masks = [(mask==c) for c in self.class_values]
        mask = np.stack(masks,axis=-1).astype('float')
        
        if self.transforms:
            sample = self.transforms(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask.float()
    
    def __len__(self):
        return len(self.image_filenames)
    
    def visualize_mask(self,i,_class):
        mask = cv2.imread(self.mask_filedirections[i])
        image = np.array(Image.open(self.image_filedirections[i]))
        class_value = self.classlist.index(_class.lower())
        
        mask = (mask==class_value)
        mask = mask.astype('float')
        width, height, channels = mask.shape

        black = np.array([0,0,0])
        white = np.array([255,255,255])
        magenta = np.array([186,85,211])

        masked_img = image
        
        for x in range(width):
            for y in range(height):
                if (mask[x,y,:]>=0.8).all():
                    masked_img[x,y,:]=0.4*image[x,y,:]+0.6*magenta
        
        plt.imshow(masked_img)
        
def visualize_result(image,result_mask):
    mask = result_mask.cpu().detach().numpy()#.transpose(2,0,1)
    image = image.cpu().detach().numpy().transpose(1,2,0)
    
    width, height = mask.shape

    magenta = np.array([186,85,211])/255

    masked_img = image
        
    for x in range(width):
        for y in range(height):
            if (mask[x,y]==1.).all():
                masked_img[x,y,:]=0.4*image[x,y,:]+0.6*magenta
        
    plt.imshow(masked_img)
        
        


        
    
        
