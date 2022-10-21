from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os

class TrainSeg():
    def __init__(self,model,
                traindata,
                testdata,
                valdata,
                classes,
                loss,
                batchsize,
                lr,
                num_epochs,
                model_path):
        self.model = model
        self.train_data = traindata
        self.test_data = testdata
        self.val_data = valdata
        self.classes = classes
        self.lossFunc = loss  
        self.learning_rate = lr
        self.batch_size = batchsize
        self.num_epochs = num_epochs
        self.model_path = model_path
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model.to(self.device)
        self.optim = Adam(self.model.parameters(),lr=self.learning_rate)
        
        self.trainloader = DataLoader(self.train_data, batch_size = self.batch_size, shuffle = True, num_workers = os.cpu_count())
        self.testloader = DataLoader(self.test_data, batch_size = self.batch_size, shuffle = True, num_workers = os.cpu_count())
        self.valloader = DataLoader(self.val_data, batch_size = self.batch_size, shuffle = True, num_workers = os.cpu_count())
        
        self.train_steps = len(traindata)//self.batch_size
        self.test_steps = len(testdata)//self.batch_size
        self.val_steps = len(valdata)//self.batch_size
        
        self.Loss_dict = {"train_loss":[],"test_loss":[]}
        
    def train(self):
        start_time = time.time()
        for e in range(self.num_epochs):
            self.model.train()
            
            totalTrainLoss = 0
            totalTestLoss = 0
            
            for (idx,(img,mask)) in enumerate(self.trainloader):
                (img,mask) = (img.to(self.device),mask.to(self.device))
                
                pred = self.model(img)
                loss = self.lossFunc(pred,mask)
                
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
                totalTrainLoss += loss
                
            with torch.no_grad():
                self.model.eval()
                
                for (img,mask) in self.testloader:
                    (img,mask) = (img.to(self.device),mask.to(self.device))
                    pred = self.model(img)
                    totalTestLoss += self.lossFunc(pred,mask)
                    
            avgTrainLoss = totalTrainLoss / self.train_steps
            avgTestLoss = totalTestLoss / self.test_steps
                
            self.Loss_dict["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
            self.Loss_dict["test_loss"].append(avgTestLoss.cpu().detach().numpy())
            
            print("Epoch: {}/{}".format(e + 1, self.num_epochs))
            print("Train loss: {:.6f}, Test loss: {:.4f}".format(avgTrainLoss, avgTestLoss))

        end_time = time.time()
        print("Total time taken to train the model: {:.2f}s".format(end_time - start_time))
                
    def val(self):
        with torch.no_grad():
            self.model.eval()
            totalValLoss = 0
            
            for (img,mask) in self.valloader:
                (img,mask) = (img.to(self.device),mask.to(self.device))
                pred = self.model(img)
                totalValLoss += self.lossFunc(pred,mask)
                
            avgValLoss = totalValLoss / self.val_steps
            
        print("Average Validation Loss: {:.4f}\n".format(avgValLoss))
        
    def save(self):
        torch.save(self.model.state_dict(), self.model_path)
        
    def load(self):
        self.model.load_state_dict(torch.load(self.model_path))
        
    
        
    
        

