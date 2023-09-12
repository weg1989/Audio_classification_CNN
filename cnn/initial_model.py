#!/usr/bin/env python
# coding: utf-8

# In[19]:


get_ipython().system('pip install tqdm')


# In[20]:


import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm


# download dataset
# create dataloader
# build model 
# train
# save trained model

# In[5]:


def download_mnist_datasets():
    train_data = datasets.MNIST(
    root='data',
    download=True,
    train=True,
    transform=ToTensor())
    test_data = datasets.MNIST(
    root='data',
        download=True,
        train=False,
        transform=ToTensor()
    )
    return train_data , test_data


# In[6]:


train_dataset , test_dataset = download_mnist_datasets()


# In[7]:


# Create a dataloader
Batch=128

train_data_loader = DataLoader(train_dataset, batch_size=Batch)
test_data_loader = DataLoader(test_dataset, batch_size=Batch)


# In[8]:


# Build model
class FedForNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(in_features=28*28,out_features=256),
            nn.ReLU(),
            nn.Linear(256,10))
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        return self.softmax(self.dense_layers(self.flatten(x)))
    


# In[9]:


# Steup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device


# In[11]:


model_0 = FedForNet().to(device)
model_0


# In[24]:


def train_step(model, data_loader, loss_fn, optimizer, device):
    
    model.train()
    for X,y in data_loader:
        X , y = X.to(device) , y.to(device)
        
        y_pred = model(X)
        
        loss = loss_fn(y_pred,y)
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
    print(f"Loss: {loss}")


# In[28]:


def test_step(model, data_loader, loss_fn, device):
    
    model.eval()
    
    with torch.inference_mode():
        
        for X,y in data_loader:
            X , y = X.to(device) , y.to(device)

            test_pred = model(X)

            test_loss = loss_fn(test_pred,y)

        
    print(f"Test Loss: {test_loss}")


# In[29]:


loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(params=model_0.parameters(),
                            lr=0.01)


# In[30]:


# Set the number of epochs
epochs = 3

for epoch in tqdm(range(epochs)):
      print(f"Epoch: {epoch} \n-----------")
      ### Training
      train_step(model=model_0,
                   data_loader=train_data_loader,
                   loss_fn=loss_fn,
                   optimizer=optimizer,
                   device=device)

      ###testing
      test_step(model=model_0,
                   data_loader=test_data_loader,
                   loss_fn=loss_fn,
                   device=device)


# In[31]:


model_0.state_dict()


# In[32]:


torch.save(model_0.state_dict(), "FedFornet.pth")
print("Model stored")


# In[ ]:




