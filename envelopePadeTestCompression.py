#Paderborn data autoencoder tests envelope
#used Modules
import numpy as np
from scipy.io import savemat
#import autoencoderKeras as aK
from scipy.io import loadmat#reading mat file
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis#linear discriminant classifier
from sklearn.metrics import roc_curve,auc#ROC curves
from sklearn.svm import LinearSVC
import os
from keras.models import Model#type of the ANN model
from keras.layers import Dense,Input#fully connected layer
from keras.models import Sequential
import matplotlib.pyplot as plt
import autoencoderKeras
from envelopePadeTest import readPadeEnvelope,normalize,denormalize,create_test_train_set
from padeVibrationEnvelopeAE import ready_to_classify,create_test_train_set
#dummies
path='/home/gurkan/Desktop/mlann/matlabfilesenvelopeanalysis/'
save_path='/home/gurkan/Desktop/mlann/pythonfiles/autoencoder/stacked/autoEncoders/compdecomp/'
#parameters
num_of_hidden_layers=3
num_of_neurons=[256,64,16]
num_of_neurons = list(map(int, num_of_neurons))
layer_activation='relu'
output_activation='sigmoid'
#preprocessing functions


#compression function
def compression(tr_data,test_data):
    #normalize data
    #Create autoencoder models
    model=autoencoderKeras.create_model(num_of_hidden_layers,len(tr_data[0,:]),num_of_neurons,layer_activation,output_activation)
    #pretraining
    model=autoencoderKeras.pretrain(model,tr_data,0.8,layer_activation,output_activation,'RMSprop')
    weights1=model.get_weights()
    #training
    model=autoencoderKeras.overall_train(model,tr_data,0.1,'RMSprop')
    #test of ae on training set
    tr_pred = model.predict(tr_data)
    #test of ae on test set
    test_pred = model.predict(test_data)
    #coding model
    code_model=Sequential()
    for i in range(num_of_hidden_layers):
        if i==0:
            code_model.add(Dense(num_of_neurons[i],activation='relu',input_dim=len(tr_data[0,:])))
        else:
            code_model.add(Dense(num_of_neurons[i],activation='relu'))

    #copy trained weights
    weights=model.get_weights()
    code_model.set_weights(weights[0:2*num_of_hidden_layers])
    #codes
    test_code=code_model.predict(test_data)
    tr_code=code_model.predict(tr_data)
    return tr_pred,test_pred,tr_code,test_code,weights1


#main of compression
data=readPadeEnvelope(path)
train_set,test_set=create_test_train_set(data)
print(len(train_set))
for i in range(len(train_set)):
    training,test=ready_to_classify(data,train_set[i],test_set[i])
    tr_k,tr_nors=normalize(training['data'])
    test_k,test_nors=normalize(test['data'])
    tr_pred,test_pred,tr_code,test_code,weights2=compression(tr_k,test_k)
    training['data']=denormalize(tr_pred,tr_nors)
    test['data']=denormalize(test_pred,test_nors)
    training['code']=tr_code
    test['code']=test_code
    training['weights']=weights2
    savemat(save_path+'tr'+str(i),training)
    savemat(save_path+'test'+str(i),test)
