#Modules
from keras.models import Model#type of the ANN model
from keras.layers import Dense,Input#fully connected layer
from keras.models import Sequential
from keras import losses
from keras import optimizers
from keras import backend as K
import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import os
import scipy.io
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from  sklearn.svm import LinearSVC
import json

#Load data
def load_data(name_of_file):
    with open(name_of_file) as json_file:
        data = json.load(json_file)
    return data

#select training and test data
def cwru_train_test_sets(data):
    #takes the data with special format and returns training and test datasets and their labels respectively
    training_data=[]
    test_data=[]
    #Data:
    #   4 baseline-  Training 3 Test:1
    normal_index = list(index for (index, d) in enumerate(data) if d["Health_status"] == "Normal")
    normal_perm=np.random.permutation(normal_index)
    for i in normal_perm[:3]:
        training_data.append(data[i])
    for i in normal_perm[3:]:
        test_data.append(data[i])

    #   28 Outer Race fault- Training 21 Test:7
    outer_index = list(index for (index, d) in enumerate(data) if d["Health_status"] == "Outer")
    outer_perm=np.random.permutation(outer_index)
    for i in outer_perm[:21]:
        training_data.append(data[i])
    for i in outer_perm[21:]:
        test_data.append(data[i])
    #   12 Inner Race fault- Training 9 Test:3
    inner_index = list(index for (index, d) in enumerate(data) if d["Health_status"] == "Inner")
    inner_perm=np.random.permutation(inner_index)
    for i in inner_perm[:9]:
        training_data.append(data[i])
    for i in inner_perm[9:]:
        test_data.append(data[i])
    #   12 Ball fault- Training 9 Test:3
    ball_index = list(index for (index, d) in enumerate(data) if d["Health_status"] == "Ball")
    ball_perm=np.random.permutation(ball_index)
    for i in ball_perm[:9]:
        training_data.append(data[i])
    for i in ball_perm[9:]:
        test_data.append(data[i])
    tr_data_array=training_data[0]['Data']
    tr_data_label=np.ones(np.size(training_data[0]['Data'],axis=1))
    print(len(training_data))
    for i in range(1,len(training_data)):
        tr_data_array=np.concatenate((tr_data_array,training_data[i]['Data']),axis=1)
        tr_data_label=np.concatenate((tr_data_label,np.ones(np.size(training_data[i]['Data'],axis=1))))
    test_data_array=test_data[0]['Data']
    test_data_label=np.ones(np.size(test_data[0]['Data'],axis=1))
    for i in range(1,len(test_data)):
        test_data_array=np.concatenate((test_data_array,test_data[i]['Data']),axis=1)
        test_data_label=np.concatenate((test_data_label,np.ones(np.size(test_data[i]['Data'],axis=1))))
    print('Size of training data:',np.size(tr_data_label))
    print('Size of test data:',np.size(test_data_label))
    return tr_data_array,test_data_array,tr_data_label,test_data_label

def abc(num_of_layers,input_size,*args):
    for i in range(len(args)):
        if type(args[i])!=int:
            raise Exception('Number of neurons in hidden layers must be integer')
    if len(args)!=num_of_layers and len(args)!=0:
        raise Exception('Number of layers is ',num_of_layers)
    else:
        if len(args)==num_of_layers:
            model_param=args
        else:
            model_param=np.zeros(num_of_layers)
            for i in range(num_of_layers):
                model_param[i]=int(input_size/pow(4,i+1))
        model=Sequential()
        model_param=[int(x) for x in model_param]
        print(model_param)
        #encoder
        for i in range(num_of_layers):
            if i==0:
                model.add(Dense(model_param[i],activation='sigmoid',input_dim=input_size))
            else:
                model.add(Dense(model_param[i],activation='sigmoid'))
        #decoder
        for i in range(1,num_of_layers):
            if i==num_of_layers-1:
                model.add(Dense(input_size))
            else:
                model.add(Dense(model_param[num_of_layers-i-1],activation='sigmoid'))
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9)
        model.compile(loss='mean_squared_error',
              optimizer='sgd')
    return model
#Pretraining of stacked autoencoder
def pretrain(model,data):
    weights=model.get_weights()
    layer_input=data
    for i in range(len(weights)/2):
        

'''
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
stacked_ae1.fit(training_set, training_set,epochs=15)

get_3rd_layer_output = K.function([stacked_ae1.layers[0].input],
                                  [stacked_ae1.layers[1].output])
layer_1_predictions = get_3rd_layer_output([training_set, 0])[0]
inputs2=Input(shape=(h_layer1,))
layer_2=Dense(h_layer2,activation='sigmoid')(inputs2)
predictions2=Dense(h_layer1,activation='sigmoid')(layer_2)
stacked_ae2=Model(inputs=inputs2,outputs=predictions2)
stacked_ae2.compile(optimizer=optimizers.SGD(lr=0.1,momentum=0.9),
               loss='mean_squared_error')
stacked_ae2.fit(layer_1_predictions,layer_1_predictions,epochs=15)

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
ae.fit(training_set, training_set,epochs=5)
get_feature = K.function([ae.layers[0].input],
                                  [ae.layers[2].output])

healthy_data_feature = get_feature([data_tr, 0])[0]
faulty_data_feature=get_feature([faulty_data,0])[0]
training_set_hid=np.random.rand(healthy_data_feature.shape[0])<0.5
training_set_fid=np.random.rand(faulty_data_feature.shape[0])<0.05
classification_training_set=np.concatenate((healthy_data_feature[training_set_hid,:],
                                            faulty_data_feature[training_set_fid,:]),axis=0)
classification_training_set_classes=np.concatenate((np.zeros((sum(training_set_hid),1)),np.ones((sum(training_set_fid),1))))
classification_test_set=np.concatenate((healthy_data_feature[~training_set_hid,:],faulty_data_feature[~training_set_fid,:]),
                                                                axis=0)
classification_test_set_classes=np.concatenate((np.zeros((data_tr.shape[0]-sum(training_set_hid),1)),
                                        np.ones((faulty_data.shape[0]-sum(training_set_fid),1))),axis=0)

print(sum(training_set_hid),sum(training_set_fid))
clf=SVC(class_weight='balanced')
clf.fit(classification_training_set,classification_training_set_classes.ravel())
test_result=clf.predict(classification_test_set)
g_tru=classification_test_set_classes.ravel()
print(sum(abs(test_result-g_tru)))
clfKnn=KNeighborsClassifier(n_neighbors=5)
clfKnn.fit(classification_training_set,classification_training_set_classes)
test_resultKnn=clfKnn.predict(classification_test_set)
print(sum(abs(test_resultKnn-g_tru)))
clfLinear=LinearSVC()
clfLinear.fit(classification_training_set,classification_training_set_classes)
test_resultlinear=clfLinear.predict(classification_test_set)
print(sum(abs(test_resultlinear-g_tru)))
'''



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
