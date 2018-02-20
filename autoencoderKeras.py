from keras.models import Model#type of the ANN model
from keras.layers import Dense,Input#fully connected layer
from keras import losses
from keras import optimizers
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io
#read data
spectrum_path="/home/gurkan/Desktop/mlann/CWRU/NormalBaseline/1772/"
faulty_spectrum_path="/home/gurkan/Desktop/mlann/CWRU/48DriveEndFault/1772/"
faulty_data=np.load(faulty_spectrum_path+'0.021-InnerRace_fft.npy')
faulty_data=np.transpose(faulty_data)
data=np.load(spectrum_path+'Normal_fft.npy')
data=data
print(np.size(data,axis=0))
print(np.size(data,axis=1))

data_tr=np.transpose(data)
training_idx=np.random.rand(np.size(data_tr,0))<0.8
training_set=data_tr[training_idx,:]
test_set=data_tr[~training_idx,:]

#model initialization
input_size=16385
inputs=Input(shape=(input_size,))
layer_1=Dense(256,activation='sigmoid')(inputs)
predictions=Dense(input_size)(layer_1)

stacked_ae1=Model(inputs=inputs,outputs=predictions)
stacked_ae1.compile(optimizer=optimizers.SGD(lr=0.01, momentum=0.9),
              loss='mean_squared_error')
stacked_ae1.fit(training_set, training_set,epochs=15)
