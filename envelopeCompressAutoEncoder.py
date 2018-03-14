'''
Envelope Signals are obtained by applying following operations on the raw data:
Demodulation, FIR Low Pass Filter, Absolute Value
This code compresses fft of envelope signal, and assesses the performance of
the compression via comparing the results of several classification methods
on CWRU(CaseWesternReserveUniversity) bearing dataset.
3 different tests are designed:
    1. Performance comparison of rule based classifiers on raw envelope append
       decoded envelope.
    2. Performance comparison of MLP classifer on raw and decoded data.
    3. Performance comparison of MLP classifiers that use the features
       that are obtained using stacked autoencoder with those in 2.
    4. Performance comparison of two different MLP classifers: One of them uses
       the features obtained via stacked ae, and the other uses the stacked ae
       as only pretraining of the classifer.
'''
from scipy.io import loadmat#necessary for reading mat file
import numpy as np #for mathematical calculations
import random
import scipy.stats as st
import autoencoderKeras
import matplotlib.pyplot as plt
#reading data
#There are 5 fields: 'env':Envelope spectrum
#                    'label_ar'   : Fault types
#                    'severity_ar': Severity of fault: 0-7-14-21
#                    'setting_ar': Rpm of motor: 1730-1750-1772-1797
#                    'record_ar' : Each record is labeled with different id from 1 to n
#                    'f'         : Frequency values of envelope spectrum
def readData(filename):
    data=[]
    mat_obj=loadmat(filename)
    return mat_obj

#classification and test sets
def train_test_sets(data):
    settings=[1730,1750,1772,1797]
    test_id=np.random.choice((settings))
    training_id=list(set(settings)-set([test_id]))
    training_set=[]
    training_labels=[]
    test_set=[]
    test_labels=[]
    for i in range(np.size(data['env'],axis=1)):
        if data['setting_ar'][0][i]==test_id:
            test_set.append(data['env'][:,i])
            test_labels.append(data['label_ar'][0][i])
        else:
            training_set.append(data['env'][:,i])
            training_labels.append(data['label_ar'][0][i])
    return training_set,test_set,training_labels,test_labels

#create feature vectors for rule based classification
def feature_extract(data,f):
    inner_f=159.93
    outer_f=105.87
    ball_f=68.72
    ball_f2=17.77
    rms=[np.sqrt(np.mean((data[i,:]*data[i,:]))) for i in range(len(data[:,0]))]  #rms
    max_=[np.max(data[i,:]) for i in range(len(data[:,0]))] #max
    skew_=[st.skew(data[i,:]) for i in range(len(data[:,0]))] #skewness
    kurtosis_=[st.kurtosis(data[i,:]) for i in range(len(data[:,0]))] #kurtosis
    f_inner=np.argmin(abs(f-inner_f))
    f_outer=np.argmin(abs(f-outer_f))
    f_ball=np.argmin(abs(f-ball_f))
    f2_ball=np.argmin(abs(f-ball_f2))
    inner=[np.sum(data[i,:][f_inner-2:f_inner+2]) for i in range(len(data[:,0]))]
    outer=[np.sum(data[i,:][f_outer-2:f_outer+2]) for i in range(len(data[:,0]))]
    ball=[np.sum(data[i,:][f_ball-2:f_ball+2])+np.sum(data[i][f2_ball-2:f2_ball+2]) for i in range(len(data[:,0]))]
    data_stats={'inner':inner,'outer':outer,'ball':ball,'rms':rms,'max':max_,'skew':skew_,'kurtosis':kurtosis_}
    return data_stats
#compute the level of rule based values for healthy motor
def healthy_level(tr_label,data_stats):
    rms_s=.0
    max_s=.0
    skew_s=.0
    kurt_s=.0
    inner_s=.0
    outer_s=.0
    ball_s=.0
    count=.0
    for i in range(len(tr_label)):
        if tr_label[i]==1:
            rms_s+=data_stats['rms'][i]
            inner_s+=data_stats['inner'][i]
            outer_s+=data_stats['outer'][i]
            ball_s+=data_stats['ball'][i]
            count+=1
    healthy_val={'inner':3*inner_s/count,'outer':3*outer_s/count,'ball':ball_s/count,'rms':1.2*rms_s/count}
    return healthy_val
#rule based classifier for CWRU data
def rule_based_classification(data_stats,healthy_val):
    pred_label=[]
    for i in range(len(data_stats['rms'])):
        if data_stats['inner'][i]>healthy_val['inner'] and data_stats['inner'][i]>data_stats['outer'][i]:#inner fault indicator
            pred_label.append(2)
        elif data_stats['outer'][i]>healthy_val['outer'] and data_stats['outer'][i]>data_stats['inner'][i]:#outer fault indicator
            pred_label.append(3)
        elif data_stats['rms'][i]>healthy_val['rms']:#unclassified fault indicator
            pred_label.append(4)
        else:#healthy
            pred_label.append(1)
    return pred_label
#stats for predictions
def confusion_matrix(true_labels,predicted_labels):
    conf_mat=np.zeros([4,4])
    for i in range(len(true_labels)):
        for j in range(4):
            for k in range(4):
                if true_labels[i]==j+1 and predicted_labels[i]==k+1:
                    conf_mat[j][k]+=1
    return conf_mat

#normalization of data
def normalize(data):
    dc_vec=np.zeros(len(data[:,0]))
    mean_vec=np.zeros(len(data[:,0]))
    var_vec=np.zeros(len(data[:,0]))
    for i in range(len(dc_vec)):
        dc_vec[i]=data[i,0]
        mean_vec[i]=np.mean(data[i,:])
        var_vec[i]=np.std(data[i,:])
        data[i,0]=0
        data[i,:]=(data[i,:]-mean_vec[i])/var_vec[i]
    data_standards={'dc':dc_vec,'mean':mean_vec,'std':var_vec}
    return data,data_standards

def denormalize(data,data_standards):
    for i in range(len(data_standards['dc'])):
        data[i,:]=data[i,:]*data_standards['std'][i]+data_standards['mean'][i]
        data[i,0]=data_standards['dc'][i]
    return data

###main calls####
#Read data
data=readData('small_set.mat')
#Create training and test data
tr_set,test_set,tr_labels,test_labels=train_test_sets(data)
tr_set=np.asarray(tr_set)
test_set=np.asarray(test_set)
#normalize data
tr_set_st,tr_data_stands=normalize(tr_set)
test_set_st,test_data_stands=normalize(test_set)
#Create autoencoder model
model=autoencoderKeras.create_model(2,len(tr_set_st[0,:]),512,64)
#pretraining
model=autoencoderKeras.pretrain(model,tr_set_st,0.5)
#training
model=autoencoderKeras.overall_train(model,tr_set_st,0.5)
#test on training
test = model.predict(test_set_st)
test=denormalize(test,test_data_stands)
plt.figure(1)
test_error_figure,=plt.plot(test_set[0,:],label='Training')
plt.legend(handles=[test_error_figure])
plt.figure(2)
test_error_figure,=plt.plot(test[0,:],label='test')
plt.legend(handles=[test_error_figure])

plt.figure(3)
test_error_figure,=plt.plot(test_set[100,:],label='Training')
plt.legend(handles=[test_error_figure])
plt.figure(4)
test_error_figure,=plt.plot(test[100,:],label='test')
plt.legend(handles=[test_error_figure])
plt.show()

print(test_labels[100],test_labels[0])

#tr_data_stats=feature_extract(tr_set,data['f'][0])
