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
    for i in range(np.max(np.transpose(mat_obj['record_ar']))):
        env=[]
        for j in range(len(np.transpose(mat_obj['label_ar']))):
            if mat_obj['record_ar'][0][j]==i+1:
                if(len(env)==0):
                    env=np.asarray(mat_obj['env'][:,j])
                    label=mat_obj['label_ar'][0][j]
                    severity=mat_obj['severity_ar'][0][j]
                    setting=mat_obj['setting_ar'][0][j]
                else:
                    env=np.vstack((env,mat_obj['env'][:,j]))
        data.append({'env':env,'Id':i, 'label':label, 'severity':severity, 'setting':setting, 'f':mat_obj['f'][0]})
    return data
klm=readData('small_set.mat')
for i in range(len(klm)):
    print(klm[i]['Id'],klm[i]['label'],klm[i]['setting'],klm[i]['severity'])
#classification and test sets
def train_test_sets(data,test_type):
    classification_config={"severity":[7,14,21], "settings":[1730,1750,1772,1797]}
    if test_type=="severity":
        training_id=np.random.choice(classification_config["severity"],2)
        test_id=classification_config["severity"]-training_id
    return test_id,training_id
print(train_test_sets(klm,"severity"))
