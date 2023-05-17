# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 06:42:45 2021

@author: RPL 2020
"""


from lib import loaddata,plot,ae,citra
from sklearn.model_selection import train_test_split
import numpy as np

# In[]: Load data rekon dan miyawaki
stimtrain,stimtest=loaddata.Data28()

# In[]:
stim0=np.reshape(stimtest[0],(28,28)).T
stim1=np.reshape(stimtest[1],(28,28)).T
stim2=np.reshape(stimtest[2],(28,28)).T
stim3=np.reshape(stimtest[3],(28,28)).T

stimtrain0=np.reshape(stimtrain[0],(28,28)).T
stimtrain1=np.reshape(stimtrain[1],(28,28)).T
stimtrain2=np.reshape(stimtrain[2],(28,28)).T
stimtrain3=np.reshape(stimtrain[3],(28,28)).T

# In[]:
input_train,input_test=train_test_split(stimtrain, test_size=0.1, random_state=42)
output_train,output_test=train_test_split(stimtrain, test_size=0.1, random_state=42) 
# In[]
plot.tigaKolomGambar('Autoencoder CNN Denoising','Stimulus',stimtrain,'Rekonstruksi',stimtrain,'Recovery',stimtrain) 
                          

# In[]:
cnnautoencoder=ae.trainCNNDenoise(np.array(input_train), np.array(output_train),np.array(input_test), np.array(output_test))
         
# In[]: Reconstructed Data
reconstructedcnn = cnnautoencoder.predict(np.array(stimtest))

# In[]: Plot gambar
plot.tigaKolomGambar('Autoencoder CNN Denoising','Stimulus',stimtest,'Rekonstruksi',stimtest,'Recovery',reconstructedcnn) 
                          
 