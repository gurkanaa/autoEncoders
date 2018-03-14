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
def cwru_tra_test_sets(data):
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
    tr_data_label=training_data[0]['label']*np.ones(np.size(training_data[0]['Data'],axis=1))
    print(len(training_data))
    for i in range(1,len(training_data)):
        tr_data_array=np.concatenate((tr_data_array,training_data[i]['Data']),axis=1)
        tr_data_label=np.concatenate((tr_data_label,training_data[i]['label']*np.ones(np.size(training_data[i]['Data'],axis=1))))
    test_data_array=test_data[0]['Data']
    test_data_label=test_data[0]['label']*np.ones(np.size(test_data[0]['Data'],axis=1))
    for i in range(1,len(test_data)):
        test_data_array=np.concatenate((test_data_array,test_data[i]['Data']),axis=1)
        test_data_label=np.concatenate((test_data_label,test_data[i]['label']*np.ones(np.size(test_data[i]['Data'],axis=1))))
    tr_data_array=np.asarray(tr_data_array)
    test_data_array=np.asarray(test_data_array)
    for i in range(np.size(tr_data_array,axis=1)):
        tr_data_array[:,i]=tr_data_array[:,i]/np.max(tr_data_array[:,i])
    for i in range(np.size(test_data_array,axis=1)):
        test_data_array[:,i]=test_data_array[:,i]/np.max(test_data_array[:,i])
    print('Size of training data:',np.size(tr_data_label))
    print('Size of test data:',np.size(test_data_label))
    return tr_data_array,test_data_array,tr_data_label,test_data_label

def create_model(num_of_layers,input_size,*args):
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
        for i in range(0,num_of_layers):
            if i==num_of_layers-1:
                model.add(Dense(input_size,activation='linear'))
            else:
                model.add(Dense(model_param[num_of_layers-i-2],activation='sigmoid'))
    return model
#Pretraining of stacked autoencoder
def pretrain(model,data,learning_rate):
    weights=model.get_weights()
    weigts_dummy=weights
    layer_input=data
    for i in range(int(len(weights)/4)):
        if i==0:
            inputs=Input(shape=(np.size(weights[2*i],axis=0),))
            layer_1=Dense(np.size(weights[2*i],axis=1),activation='sigmoid')(inputs)
            predictions=Dense(np.size(weights[2*i],axis=0),activation='linear')(layer_1)
        else:
            inputs=Input(shape=(np.size(weights[2*i],axis=0),))
            layer_1=Dense(np.size(weights[2*i],axis=1),activation='sigmoid')(inputs)
            predictions=Dense(np.size(weights[2*i],axis=0),activation='sigmoid')(layer_1)
        stacked_ae=Model(inputs=inputs,outputs=predictions)
        stacked_ae.compile(optimizer=optimizers.SGD(lr=learning_rate, momentum=0.9),
                      loss='mean_squared_error')
        print(i)
        stacked_ae.fit(layer_input,layer_input,shuffle=True,epochs=150,batch_size=32)
        get_h_layer_output = K.function([stacked_ae.layers[0].input],
                                          [stacked_ae.layers[1].output])
        layer_1_predictions = get_h_layer_output([layer_input, 0])[0]
        w=stacked_ae.get_weights()
        del layer_input,layer_1,inputs,predictions,get_h_layer_output
        layer_input=layer_1_predictions
        weigts_dummy[2*i]=w[0]
        weigts_dummy[2*i+1]=w[1]
        weigts_dummy[len(weights)-2*i-2]=w[2]
        weigts_dummy[len(weights)-2*i-1]=w[3]
        del w
    model.set_weights(weigts_dummy)
    return model

def overall_train(model,data,learning_rate):
    sgd = optimizers.SGD(lr=learning_rate, momentum=0.9)
    model.compile(loss='mean_squared_error',
          optimizer=sgd)
    model.fit(data,data,shuffle=True,epochs=150,batch_size=32)
    return model
