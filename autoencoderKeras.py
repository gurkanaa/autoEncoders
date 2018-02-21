from keras.models import Model#type of the ANN model
from keras.layers import Dense,Input#fully connected layer
from keras import losses
from keras import optimizers
from keras import backend as K
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io
from sklearn.svm import SVC
#read data
spectrum_path="/home/gurkan/Desktop/mlann/CWRU/NormalBaseline/1772/"
faulty_spectrum_path="/home/gurkan/Desktop/mlann/CWRU/48DriveEndFault/1772/"
files=os.listdir(faulty_spectrum_path)
for file_id in files:
    if('fft'in file_id):
        if 'faulty_data' in globals():
            faulty_data=np.concatenate((faulty_data,np.load(faulty_spectrum_path+file_id)),axis=1)
        else:
            faulty_data=np.load(faulty_spectrum_path+file_id)

faulty_data=np.transpose(faulty_data)
print(np.size(faulty_data,axis=0),np.size(faulty_data,axis=1))
data=np.load(spectrum_path+'Normal_fft.npy')
data=data
print(np.size(data,axis=0))
print(np.size(data,axis=1))

data_tr=np.transpose(data)
training_idx=np.random.rand(np.size(data_tr,0))<0.8
training_set=data_tr[training_idx,:]
test_set=data_tr[~training_idx,:]

#model initialization
num_of_layers=2
input_size=16385
h_layer1=256
h_layer2=64
inputs=Input(shape=(input_size,))
layer_1=Dense(h_layer1,activation='sigmoid')(inputs)
predictions=Dense(input_size)(layer_1)

stacked_ae1=Model(inputs=inputs,outputs=predictions)
stacked_ae1.compile(optimizer=optimizers.SGD(lr=0.1, momentum=0.9),
              loss='mean_squared_error')
stacked_ae1.fit(training_set, training_set,epochs=25)

get_3rd_layer_output = K.function([stacked_ae1.layers[0].input],
                                  [stacked_ae1.layers[1].output])
layer_1_predictions = get_3rd_layer_output([training_set, 0])[0]
inputs2=Input(shape=(h_layer1,))
layer_2=Dense(h_layer2,activation='sigmoid')(inputs2)
predictions2=Dense(h_layer1,activation='sigmoid')(layer_2)
stacked_ae2=Model(inputs=inputs2,outputs=predictions2)
stacked_ae2.compile(optimizer=optimizers.SGD(lr=0.1,momentum=0.9),
               loss='mean_squared_error')
stacked_ae2.fit(layer_1_predictions,layer_1_predictions,epochs=25)

w1=stacked_ae1.get_weights()
w2=stacked_ae2.get_weights()
weights=[w1[0],w1[1],w2[0],w2[1],w2[2],w2[3],w1[2],w1[3]]

inputs=Input(shape=(input_size,))
layer1=Dense(h_layer1,activation='sigmoid')(inputs)
layer2=Dense(h_layer2,activation='sigmoid')(layer1)
layer3=Dense(h_layer1,activation='sigmoid')(layer2)
predictions=Dense(input_size)(layer3)
ae=Model(inputs=inputs,outputs=predictions)
ae.set_weights(weights)

ae.compile(optimizer=optimizers.SGD(lr=0.01, momentum=0.9),
              loss='mean_squared_error')
ae.fit(training_set, training_set,epochs=15)
get_feature = K.function([ae.layers[0].input],
                                  [ae.layers[2].output])

healthy_data_feature = get_feature([data_tr, 0])[0]
faulty_data_feature=get_feature([faulty_data,0])[0]
training_set_size=250
training_set_hid=np.random.randint(healthy_data_feature.shape[0],size=training_set_size)
training_set_fid=np.random.randint(faulty_data_feature.shape[0],size=training_set_size)
classification_training_set=np.concatenate((healthy_data_feature[training_set_hid,:],faulty_data_feature[training_set_fid,:]),axis=0)
classification_training_set_classes=np.concatenate((np.zeros((training_set_size,1)),np.ones((training_set_size,1))))

classification_test_set=np.concatenate((healthy_data_feature[~training_set_hid,:],faulty_data_feature[~training_set_fid,:]),axis=0)
classification_test_set_classes=np.concatenate((np.zeros((data_tr.shape[0]-training_set_size,1)),np.ones((faulty_data.shape[0]-training_set_size,1))))



print(np.size(classification_test_set,axis=0),np.size(classification_test_set,axis=1),np.size(classification_test_set_classes,axis=0),np.size(classification_test_set_classes,axis=1))

clf=SVC()
clf.fit(classification_training_set,classification_training_set_classes)






'''get_3rd_layer_output = K.function([stacked_ae1.layers[0].input],
                                  [stacked_ae1.layers[1].output])
layer_1_predictions = get_3rd_layer_output([training_set, 0])[0]
'''
'''
print(np.size(layer_1_predictions,axis=0),np.size(layer_1_predictions,axis=1))
plt.figure(1)
plt.plot(layer_1_predictions[1,:])
plt.figure(2)
plt.plot(training_set[1,:])
plt.show()
'''
