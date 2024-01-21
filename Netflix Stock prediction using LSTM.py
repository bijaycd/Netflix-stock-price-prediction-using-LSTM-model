#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import tensorflow
import keras


# In[8]:


import opendatasets as od
od.download("https://www.kaggle.com/datasets/jainilcoder/netflix-stock-price-prediction/data")


# In[9]:


df = pd.read_csv("netflix-stock-price-prediction/NFLX.csv")
df.head()


# In[10]:


df.info()


# In[11]:


df.shape


# In[12]:


closed_price = df['Close']
closed_price


# In[13]:


from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
scaled_price = mm.fit_transform(np.array(closed_price)[...,None]).squeeze()


# In[14]:


X=[]
Y=[]


# In[15]:


seq_len = 15
for i in range(len(scaled_price) - seq_len):
    X.append(scaled_price[i : i+ seq_len])
    Y.append(scaled_price[i+seq_len])


# In[16]:


X = np.array(X)[... , None]
Y = np.array(Y)[... ,None]


# In[17]:


import torch
import torch.nn as nn
train_x = torch.from_numpy(X[:int(0.8*X.shape[0])]).float()
train_y = torch.from_numpy(Y[:int(0.8*Y.shape[0])]).float()
test_x = torch.from_numpy(X[:int(0.8*X.shape[0])]).float()
test_y = torch.from_numpy(Y[:int(0.8*Y.shape[0])]).float()


# In[18]:


class Model(nn.Module):
    def __init__(self,input_size,hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size , hidden_size,batch_first=True)
        self.fc=nn.Linear(hidden_size, 1)
    def forward(self , x):
        output,(hidden,cell)=self.lstm(x)
        return self.fc(hidden[-1,:])


# In[19]:


model = Model(1,64)


# In[20]:


optimizer=torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
num_epochs=100


# In[21]:


for epoch in range(num_epochs):
    output = model(train_x)
    loss=loss_fn(output,train_y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10==0 and epoch !=0:
        print(epoch , "epoch loss", loss.detach().numpy())


# In[22]:


model.eval()
with torch.no_grad():
    output=model(test_x)


# In[23]:


pred=mm.inverse_transform(output.numpy())
real=mm.inverse_transform(test_y.numpy())


# In[25]:


plt.plot(pred.squeeze(),color="red",label="predicted")
plt.plot(real.squeeze(),color="green",label="real")
plt.title('Netflix Stock prediction using LSTM')
plt.legend()
plt.show()

