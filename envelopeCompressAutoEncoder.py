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
    rms=[np.sqrt(np.mean((data[i]))) for i in range(len(data))]  #rms
    max_=[np.max(data[i]) for i in range(len(data))] #max
    skew_=[st.skew(data[i]) for i in range(len(data))] #skewness
    kurtosis_=[st.kurtosis(data[i]) for i in range(len(data))] #kurtosis
    f_inner=np.argmin(abs(f-inner_f))
    f_outer=np.argmin(abs(f-outer_f))
    f_ball=np.argmin(abs(f-ball_f))
    inner=[np.sum(data[i][f_inner-1:f_inner+1]) for i in range(len(data))]
    outer=[np.sum(data[i][f_outer-1:f_outer+1]) for i in range(len(data))]
    ball=[np.sum(data[i][f_ball-1:f_ball+1]) for i in range(len(data))]
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
        if tr_label[i]==4:
            rms_s+=data_stats['rms'][i]
            max_s+=data_stats['max'][i]
            skew_s+=data_stats['skew'][i]
            kurt_s+=data_stats['kurtosis'][i]
            inner_s+=data_stats['inner'][i]
            outer_s+=data_stats['outer'][i]
            ball_s+=data_stats['ball'][i]
            count+=1
    healthy_val={'inner':inner_s/count,'outer':outer_s/count,'ball':ball_s/count,'rms':rms_s/count,'max':max_s/count,'skew':skew_s/count,'kurtosis':kurt_s/count}
    return healthy_val

#main calls
data=readData('small_set.mat')
tr_set,test_set,tr_label,test_label=train_test_sets(data)
tr_data_stats=feature_extract(tr_set,data['f'][0])
tr_healthy_val=healthy_level(tr_label,tr_data_stats)
print(tr_healthy_val['inner'],tr_healthy_val['outer'],tr_healthy_val['ball'],tr_healthy_val['rms'],tr_healthy_val['skew'],tr_healthy_val['kurtosis'],tr_healthy_val['max'])
