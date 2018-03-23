#Paderborn data autoencoder tests envelope
#used Modules
import numpy as np
#import autoencoderKeras as aK
from scipy.io import loadmat#reading mat file
import os
import matplotlib.pyplot as plt

#dummies
f=np.arange(8191)*(4000/8190)#frequency axis values of envelope signal


#read Data
path='/home/gurkan/Desktop/mlann/matlabfilesenvelopeanalysis/'
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


#create training and test sets
def create_test_train_set(data):
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
        elif 'B' in key:
            in_out.append(key)
    test_set=[]
    train_set=[]
    for key_i in inner:
        for key_o in outer:
            for key_h in healthy:
                test_set.append(in_out+[key_h]+[key_i]+[key_o])
                train_set.append(list(set(data.keys())-set(in_out+[key_h]+[key_i]+[key_o])))
    return train_set,test_set

#extracting features for data
def feature_extract(data,f):
    inner_f=73.98
    f_turn=15
    outer_f=45.75
    data_stats={}
    f_inner=np.argmin(abs(f-inner_f))
    f_inner2=np.argmin(abs(f-inner_f+f_turn))
    print(data['K001'][:,1][f_inner2-1:f_inner2+2])
    print(data['K001'][:,1][f_inner-1:f_inner+2])
    f_outer=np.argmin(abs(f-outer_f))
    print(np.concatenate(data['K001'][:,1][f_inner-1:f_inner+2],data['K001'][:,1][f_inner2-1:f_inner2+2]))
    for key in data.keys():
        print(len(data[key][:,1]))
        inner=[np.sum(np.concatenate(data[key][:,i][f_inner-1:f_inner+2],data[key][:,i][f_inner2-1:f_inner2+2])) for i in range(len(data[key][0,:]))]
        outer=[np.sum(data[key][:,i][f_outer-1:f_outer+2],data[key][:,i][2*f_outer-1:2*f_outer+2]) for i in range(len(data[key][0,:]))]
        data_stats.update({key:{'inner':inner,'outer':outer}})
    return data_stats
#main code
data=readPadeEnvelope(path)
train_set,test_set=create_test_train_set(data)
data_stats=feature_extract(data,f)
print(data_stats['K001'])
