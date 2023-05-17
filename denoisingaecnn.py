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
