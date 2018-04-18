#Paderborn data autoencoder tests envelope
#used Modules
import numpy as np
#import autoencoderKeras as aK
from scipy.io import loadmat#reading mat file
import os
import matplotlib.pyplot as plt
from keras.models import Model#type of the ANN model
from keras.layers import Dense,Input,Dropout#fully connected layer
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.svm import LinearSVC
#dummies
num_of_hidden_layers=3
num_of_neurons=[2048,512,128]
f=np.arange(8189)*(8000/8188)#frequency axis values of envelope signal
fault_t='inner'
sample_test=37
if fault_t=='outer':
    fault_l=1
else:
    fault_l=0

#read Data
save_path='/home/gurkan/Desktop/mlann/pythonfiles/autoencoder/stacked/autoEncoders/compdecomp1/'
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
#normalization of data
def normalize(data_in):
    data=np.copy(data_in)
    data[:,0]=0
    min_vec=np.min(data,axis=1)
    max_vec=np.max(data,axis=1)-min_vec
    for i in range(len(max_vec)):
        data[i,:]=(data[i,:]-min_vec[i])/max_vec[i]
    data_nors={'min':min_vec,'max':max_vec}
    return data,data_nors

def denormalize(data_in,data_nors):
    data=np.copy(data_in)
    for i in range(len(data_nors['max'])):
        data[i,:]=data[i,:]*data_nors['max'][i]+data_nors['min'][i]
    return data
#create training and test sets
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

#extracting features for data
def feature_extract(data,f):
    inner_f=123.3
    f_turn=25
    outer_f=76.25
    f_inner=np.argmin(abs(f-inner_f))
    f_inner2=np.argmin(abs(f-inner_f+f_turn))
    f_outer=np.argmin(abs(f-outer_f))
    inner=[np.sum(np.concatenate((data['data'][i,:][f_inner-1:f_inner+2],data['data'][i,:][f_inner2-1:f_inner2+2]),axis=0))/np.max(data['data'][i,:]) for i in range(len(data['data'][:,0]))]
    outer=[np.sum(np.concatenate((data['data'][i,:][f_outer-1:f_outer+2],data['data'][i,:][2*f_outer-1:2*f_outer+2]),axis=0))/np.max(data['data'][i,:]) for i in range(len(data['data'][:,0]))]
    data_stats={'inner':np.asarray(inner),'outer':np.asarray(outer)}
    return data_stats

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

#healthy values
def healthy_average_std(data_features,data_labels,fault_type):
    if fault_type=='inner':
        _index_h=np.where(data_labels!=1)
        _index_f=np.where(data_labels==1)
    else:
       _index_h=np.where(data_labels!=2)
       _index_f=np.where(data_labels==2)
    _mean=np.mean(data_features[fault_type][_index_h])
    _std=np.std(data_features[fault_type][_index_h])
    _mean_f=np.mean(data_features[fault_type][_index_f])
    _std_f=np.std(data_features[fault_type][_index_f])
    return _mean,_std,_std_f,_mean_f

#Rule Based classifier
def my_classifier(test_data_features,thr_i,thr_o):
    p_l=np.zeros(len(test_data_features['inner']))
    for i in range(len(test_data_features['inner'])):
        if test_data_features['inner'][i]>thr_i:
            if test_data_features['inner'][i]>test_data_features['outer'][i]:
                p_l[i]=1
            else:
                p_l[i]=2
        elif test_data_features['outer'][i]>thr_o:
            p_l[i]=2

    return p_l

#ROC curve
def my_roc(data_features,data_labels,h_mu,h_std,fault_type):
    _k=np.linspace((np.max(data_features)-h_mu)/h_std,(np.min(data_features)-h_mu)/h_std,num=1000)
    _acc=np.ones(len(_k))
    _tpr=np.ones(len(_k))
    _fpr=np.ones(len(_k))
    for i in range(len(_k)):
        if fault_type=='inner':
            _tpr[i]=np.sum([a and b for a, b in zip(data_features>h_mu+_k[i]*h_std, data_labels==1)])/len(data_labels)
            _fpr[i]=np.sum([a and b for a, b in zip(data_features<h_mu+_k[i]*h_std, data_labels!=1)])/len(data_labels)
            _acc[i]=_tpr[i]+_fpr[i]
        elif fault_type=='outer':
            _tpr[i]=np.sum([a and b for a, b in zip(data_features>h_mu+_k[i]*h_std, data_labels==2)])/len(data_labels)
            _fpr[i]=np.sum([a and b for a, b in zip(data_features<h_mu+_k[i]*h_std, data_labels!=2)])/len(data_labels)
            _acc[i]=_tpr[i]+_fpr[i]
    op_point=np.argmax(_acc)
    return _k[op_point]

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

#main code
data=readPadeEnvelope(path)
train_set,test_set=create_test_train_set(data)


#will be computed
conf_raw_rule=np.zeros((3,3))
conf_raw_mlp=np.zeros((3,3))
conf_code_mlp=np.zeros((3,3))
conf_full_pre=np.zeros((3,3))
training={}
test={}
for i in range(len(train_set)):
    #raw rule based
    print(test_set[i])
    training,test=ready_to_classify(data,train_set[i],test_set[i])
    tr_data_features=feature_extract(training,f)
    test_data_features=feature_extract(test,f)
    h_mu,h_std,f_std,f_mu=healthy_average_std(tr_data_features,training['labels'],'inner')
    ki_=my_roc(tr_data_features['inner'],training['labels'],h_mu,h_std,'inner')
    h_mu,h_std,f_std,f_mu=healthy_average_std(tr_data_features,training['labels'],'outer')
    ko_=my_roc(tr_data_features['outer'],training['labels'],h_mu,h_std,'outer')
    predicted_labels=my_classifier(test_data_features,ki_,ko_)
    conf_raw_rule1=confusion_matrix(test['labels'],predicted_labels)
    conf_raw_rule+=conf_raw_rule1

    #raw MLP
    model = Sequential()
    model.add(Dense(num_of_neurons[0], input_dim=len(training['data'][1,:]), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_of_neurons[1], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_of_neurons[2], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3,activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    labels = to_categorical(training['labels'], num_classes=3)
    model.fit(training['data'],labels,shuffle=True,epochs=10,batch_size=32,verbose=2)
    predicted_labels=np.argmax(model.predict(test['data']),axis=1)
    conf_raw_mlp1=confusion_matrix(test['labels'],predicted_labels)
    conf_raw_mlp+=conf_raw_mlp1

    #code MLP
    training['code']=loadmat(save_path+'tr'+str(i))['code']
    test['code']=loadmat(save_path+'test'+str(i))['code']
    training['labels']=loadmat(save_path+'tr'+str(i))['labels'][0]
    test['labels']=loadmat(save_path+'test'+str(i))['labels'][0]
    weights=loadmat(save_path+'tr'+str(i))['weights'][0]
    code_model = Sequential()
    code_model.add(Dense(32, input_dim=num_of_neurons[num_of_hidden_layers-1], activation='relu'))
    code_model.add(Dropout(0.2))
    code_model.add(Dense(8, activation='relu'))
    code_model.add(Dropout(0.2))
    code_model.add(Dense(3,activation='softmax'))

    code_model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    labels = to_categorical(training['labels'], num_classes=3)
    code_model.fit(training['code'],labels,shuffle=True,epochs=10,batch_size=32,verbose=2)
    predicted_labels=np.argmax(code_model.predict(test['code']),axis=1)
    conf_code_mlp1=confusion_matrix(test['labels'],predicted_labels)
    conf_code_mlp+=conf_code_mlp1

    #code full train
    full_model=Sequential()
    for j in range(num_of_hidden_layers):
        if j==0:
            full_model.add(Dense(num_of_neurons[j],activation='relu',input_dim=len(training['data'][0,:])))
        else:
            full_model.add(Dense(num_of_neurons[j],activation='relu'))
    deneme=full_model.get_weights()
    for klm in range(len(deneme)):
        np.copyto(deneme[klm],weights[klm])
    full_model.set_weights(deneme)
    full_model.add(Dense(32, activation='relu'))
    full_model.add(Dropout(0.2))
    full_model.add(Dense(8, activation='relu'))
    full_model.add(Dropout(0.2))
    full_model.add(Dense(3,activation='softmax'))
    full_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    full_model.fit(training['data'],labels,shuffle=True,epochs=10,batch_size=32,verbose=2)
    predicted_labels=np.argmax(full_model.predict(test['data']),axis=1)
    conf_full_pre1=confusion_matrix(test['labels'],predicted_labels)
    conf_full_pre+=conf_full_pre1

    print(conf_raw_rule1)
    print(conf_raw_mlp1)
    print(conf_code_mlp1)
    print(conf_full_pre1)

conf_raw_rule/=len(test_set)
conf_raw_mlp/=len(test_set)
conf_code_mlp/=len(test_set)
conf_full_pre/=len(test_set)

print(conf_raw_rule)
print(conf_raw_mlp)
print(conf_code_mlp)
print(conf_full_pre)
