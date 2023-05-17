# In[]:
from scipy.io import savemat, loadmat
import numpy as np
from matplotlib import pyplot as plt

# In[]:
handwriten_69=loadmat('digit69_28x28.mat')
#ini fmri 10 test 90 train satu baris berisi 3092
Y_train = handwriten_69['fmriTrn'].astype('float32')
Y_test = handwriten_69['fmriTest'].astype('float32')

# ini stimulus semua
X_train = handwriten_69['stimTrn']#90 gambar dalam baris isi per baris 784 kolom
X_test = handwriten_69['stimTest']#10 gambar dalam baris isi 784 kolom
# normalisasi agar hasil 0-1
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# In[]:
resolution=28
X_train = X_train.reshape([X_train.shape[0], resolution, resolution])
X_test = X_test.reshape([X_test.shape[0], resolution, resolution])
# %%
X_train_t = [np.transpose(matrix) for matrix in X_train]

X_test_t = [np.transpose(matrix) for matrix in X_test]
# %%
X_train_t = np.transpose(X_train, (0, 2, 1))
X_test_t = np.transpose(X_test, (0, 2, 1))
