#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install torchsummary')


# In[5]:


from torch import nn
from torchsummary import summary


# In[9]:


class CNNNetwork(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=1,
                          out_channels=16,
                         kernel_size=3,
                         stride=1,
                         padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=16,
                          out_channels=32,
                         kernel_size=3,
                         stride=1,
                         padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.conv3 = nn.Sequential(
                nn.Conv2d(in_channels=32,
                          out_channels=64,
                         kernel_size=3,
                         stride=1,
                         padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.conv4 = nn.Sequential(
                nn.Conv2d(in_channels=64,
                          out_channels=128,
                         kernel_size=3,
                         stride=1,
                         padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.flatten = nn.Flatten()
        
        self.linear = nn.Linear(in_features=128*5*4, out_features=10)
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.softmax(self.linear(self.flatten(x)))


# In[10]:


cnn = CNNNetwork()
summary(cnn.cuda(), (1,64,44))


# In[ ]:




