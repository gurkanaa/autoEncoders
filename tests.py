import numpy as np
#import autoencoderKeras as aK
from scipy.io import loadmat#reading mat file
import os
import matplotlib.pyplot as plt
from keras.models import Model#type of the ANN model
from keras.layers import Dense,Input,Dropout#fully connected layer
from keras.models import Sequential
from keras.utils import to_categorical

def readPadeEnvelope(path):
    data={}
    for folders in os.listdir(path):
        env=[]
        if folders.startswith('K'):
            for files in os.listdir(path+folders):
                dummy=loadmat(path+folders+'/'+files)
                if len(env)==0:
                    env=dummy['env_mat']
                else:
                    env=np.append(env,dummy['env_mat'],axis=1)
            data.update({folders:env})
    return data

def create_test_train_set(data):
    '''
    inner=[]
    healthy=[]
    outer=[]
    in_out=[]
    for key in data.keys():
        if '00' in key:
            healthy.append(key)
        elif 'A' in key:
            outer.append(key)
        elif 'I' in key:
            inner.append(key)
        #elif 'B' in key:
        #    in_out.append(key)
    test_set=[]
    '''
    test_set=[['K001','KA04','KI04'],['K002','KA15','KI14'],['K003','KA16','KI16'],['K004','KA22','KI18'],['K005','KA30','KI21']]
    train_set=[]
    for i in range(len(test_set)):
        train_set.append(list(set(data.keys())-set(test_set[i])))
    return train_set,test_set

def ready_to_classify(data,training_set,test_set):
    training_data=[]
    training_labels=[]
    test_data=[]
    test_labels=[]
    for key in data.keys():
        for i in range(np.size((data[key]),axis=1)):
            if key in training_set:
                training_data.append(data[key][:,i])
                if key.startswith('KI'):
                    training_labels.append(1)
                elif key.startswith('KA'):
                    training_labels.append(2)
                else:
                    training_labels.append(0)
            else:
                test_data.append(data[key][:,i])
                if key.startswith('KI'):
                    test_labels.append(1)
                elif key.startswith('KA'):
                    test_labels.append(2)
                else:
                    test_labels.append(0)
    data=(np.asarray(training_data))
    data_t=(np.asarray(test_data))
    training={'data':data,'labels':np.asarray(training_labels)}
    test={'data':data_t,'labels':np.asarray(test_labels)}
    return training,test

#confusion matrix
def confusion_matrix(true_labels,predicted_labels):
    conf_mat=np.zeros([3,3])
    for i in range(len(true_labels)):
        for j in range(3):
            for k in range(3):
                if true_labels[i]==j and predicted_labels[i]==k:
                    conf_mat[j][k]+=1
    for j in range(3):
        conf_mat[j][:]/=sum(conf_mat[j][:])
    return conf_mat

#read Data
save_path='/home/gurkan/Desktop/mlann/pythonfiles/autoencoder/stacked/autoEncoders/compdecomp/'
path='/home/gurkan/Desktop/mlann/matlabfilesenvelopeanalysis/'

#initialize
conf_raw_mlp=np.zeros((3,3))
conf_comp_mlp=np.zeros((3,3))
num_of_neurons=[256,64,16]
training_pred={}
test_pred={}
#main code
for i in range(5):
    data=readPadeEnvelope(path)
    train_set,test_set=create_test_train_set(data)
    training,test=ready_to_classify(data,train_set[i],test_set[i])
    training_pred['data']=loadmat(save_path+'tr'+str(i))['data']
    test_pred['data']=loadmat(save_path+'test'+str(i))['data']
    training_pred['labels']=loadmat(save_path+'tr'+str(i))['labels'][0]
    test_pred['labels']=loadmat(save_path+'test'+str(i))['labels'][0]





    #raw MLP
    model = Sequential()
    model.add(Dense(num_of_neurons[0], input_dim=len(training['data'][1,:]), activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(num_of_neurons[1], activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(num_of_neurons[2], activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(3,activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    labels = to_categorical(training['labels'], num_classes=3)
    model.fit(training['data'],labels,shuffle=True,epochs=5,batch_size=32,verbose=0)
    predicted_labels=np.argmax(model.predict(test['data']),axis=1)
    conf_raw_mlp1=confusion_matrix(test['labels'],predicted_labels)
    conf_raw_mlp+=conf_raw_mlp1

    #compressed mlp
    model = Sequential()
    model.add(Dense(num_of_neurons[0], input_dim=len(training['data'][1,:]), activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(num_of_neurons[1], activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(num_of_neurons[2], activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(3,activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    labels = to_categorical(training_pred['labels'], num_classes=3)
    model.fit(training_pred['data'],labels,shuffle=True,epochs=5,batch_size=32,verbose=0)
    predicted_labels=np.argmax(model.predict(test_pred['data']),axis=1)
    conf_comp_mlp1=confusion_matrix(test_pred['labels'],predicted_labels)
    conf_comp_mlp+=conf_comp_mlp1
conf_raw_mlp/=5
conf_comp_mlp/=5
print(conf_raw_mlp)
print(conf_comp_mlp)
