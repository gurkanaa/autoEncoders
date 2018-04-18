#Paderborn data autoencoder tests envelope
#used Modules
import numpy as np
#import autoencoderKeras as aK
from scipy.io import loadmat#reading mat file
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis#linear discriminant classifier
from sklearn.metrics import roc_curve,auc,roc_auc_score#ROC curves
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
import os
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#dummies
f=np.arange(2048)*(2000/2047)#frequency axis values of envelope signal
fault_t='inner'
sample_test=37
if fault_t=='outer':
    fault_l=1
else:
    fault_l=0

#read Data
save_path='/home/gurkan/Desktop/mlann/pythonfiles/autoencoder/stacked/autoEncoders/compdecomp/'
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
    train_set=[]
    for i in range(len(inner)):
        test_set.append(in_out+[healthy[i]]+[inner[i]]+[outer[i]])
        train_set.append(list(set(data.keys())-set(in_out+[healthy[i]]+[inner[i]]+[outer[i]])))
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
                    training_labels.append([1,0])
                elif key.startswith('KA'):
                    training_labels.append([0,1])
                elif key.startswith('KB'):
                    training_labels.append([1,1])
                else:
                    training_labels.append([0,0])
            else:
                test_data.append(data[key][:,i])
                if key.startswith('KI'):
                    test_labels.append([1,0])
                elif key.startswith('KA'):
                    test_labels.append([0,1])
                elif key.startswith('KB'):
                    test_labels.append([1,1])
                else:
                    test_labels.append([0,0])
    data=(np.asarray(training_data))
    data_t=(np.asarray(test_data))
    training={'data':data,'labels':np.asarray(training_labels)}
    test={'data':data_t,'labels':np.asarray(test_labels)}
    return training,test

#healthy values
def healthy_average_std(data_features,data_labels,fault_type):
    if fault_type=='inner':
        _index_h=np.where(data_labels[:,0]==0)
        _index_f=np.where(data_labels[:,0]==1)
    else:
        _index_h=np.where(data_labels[:,1]==0)
        _index_f=np.where(data_labels[:,1]==1)
    _mean=np.mean(data_features[fault_type][_index_h])
    _std=np.std(data_features[fault_type][_index_h])
    _mean_f=np.mean(data_features[fault_type][_index_f])
    _std_f=np.std(data_features[fault_type][_index_f])
    return _mean,_std,_std_f,_mean_f

#Rule Based classifier
def my_classifier(test_data_features,thr):
    p_l=test_data_features>thr
    return p_l
#ROC curve
def my_roc(data_features,data_labels,h_mu,h_std,fault_type):
    _k=np.linspace((np.max(data_features)-h_mu)/h_std,(np.min(data_features)-h_mu)/h_std,num=2210)
    _fpr=np.ones(len(_k))
    _tpr=np.ones(len(_k))
    _acc=np.ones(len(_k))
    for i in range(len(_k)):
        _tpr[i]=np.sum([a and b for a, b in zip(data_features>h_mu+_k[i]*h_std, data_labels==1)])/np.sum(data_labels)
        _fpr[i]=np.sum([a and b for a, b in zip(data_features>h_mu+_k[i]*h_std, data_labels==0)])/np.sum(1-data_labels)
        _acc[i]=np.sum((data_features>h_mu+_k[i]*h_std) == data_labels)/len(data_labels)
    _auc=np.sum([(_tpr[i]*(_fpr[i+1]-_fpr[i])) for i in range(len(_tpr)-1)])
    op_point=np.argmax(_tpr-_fpr)

    return _fpr,_tpr,_auc,_k,op_point
#confusion matrix
def confusion_matrix(true_labels,predicted_labels):
    conf_mat=np.zeros([2,2])
    for i in range(len(true_labels)):
        for j in range(2):
            for k in range(2):
                if true_labels[i]==j and predicted_labels[i]==k:
                    conf_mat[j][k]+=1
    for j in range(2):
        conf_mat[j][:]/=sum(conf_mat[j][:])
    return conf_mat
#main code
data=readPadeEnvelope(path)
train_set,test_set=create_test_train_set(data)

'''
fpr_raw_rule=[]
tpr_raw_rule=[]
conf_raw_rule=np.zeros((2,2))
fpr_codec_rule=[]
tpr_codec_rule=[]
conf_codec_rule=np.zeros((2,2))
fpr_raw_svm=[]
tpr_raw_svm=[]
conf_raw_svm=np.zeros((2,2))
fpr_codec_svm=[]
tpr_codec_svm=[]
conf_codec_svm=np.zeros((2,2))
fpr_code_svm=[]
tpr_code_svm=[]
conf_code_svm=np.zeros((2,2))
training={}
test={}
for i in range(len(train_set)):

    #raw rule based
    training,test=ready_to_classify(data,train_set[i],test_set[i])
    tr_data_features=feature_extract(training,f)
    test_data_features=feature_extract(test,f)
    h_mu,h_std,f_std,f_mu=healthy_average_std(tr_data_features,training['labels'],fault_t)
    fpr_,tpr_,auc_,k_,op_point=my_roc(test_data_features[fault_t],test['labels'][:,fault_l],h_mu,h_std,fault_t)
    predicted_labels=my_classifier(test_data_features[fault_t],k_[op_point])
    conf_raw_rule+=confusion_matrix(test['labels'][:,fault_l],predicted_labels)
    tpr_raw_rule.append(tpr_[0:2210])
    fpr_raw_rule.append(fpr_[0:2210])

    #raw svm
    sel=SelectKBest(k=15)
    sel.fit(training['data'],training['labels'][:,fault_l])
    features=sel.transform(training['data'])
    clf = LinearSVC()
    clf.fit(features,training['labels'][:,fault_l])
    y_score = clf.fit(sel.transform(training['data']),training['labels'][:,fault_l]).decision_function(sel.transform(test['data']))
    fpr_, tpr_, _ = roc_curve(test['labels'][:,fault_l],y_score,drop_intermediate=False)
    predicted_labels=clf.predict(sel.transform(test['data']))
    conf_raw_svm+=confusion_matrix(test['labels'][:,fault_l],predicted_labels)
    fpr_raw_svm.append(fpr_[0:2210])
    tpr_raw_svm.append(tpr_[0:2210])

    #compressed rule based
    training['data']=loadmat(save_path+'tr'+str(i))['data']
    training['labels']=loadmat(save_path+'tr'+str(i))['labels']
    test['data']=loadmat(save_path+'test'+str(i))['data']
    test['labels']=loadmat(save_path+'test'+str(i))['labels']
    tr_data_features=feature_extract(training,f)
    test_data_features=feature_extract(test,f)
    h_mu,h_std,f_std,f_mu=healthy_average_std(tr_data_features,training['labels'],fault_t)
    fpr_,tpr_,auc_,k_,op_point=my_roc(test_data_features[fault_t],test['labels'][:,fault_l],h_mu,h_std,fault_t)
    predicted_labels=my_classifier(test_data_features[fault_t],k_[op_point])
    conf_codec_rule+=confusion_matrix(test['labels'][:,fault_l],predicted_labels)
    tpr_codec_rule.append(tpr_[0:2210])
    fpr_codec_rule.append(fpr_[0:2210])

    #compressed Svm
    sel=SelectKBest(k=15)
    sel.fit(training['data'],training['labels'][:,fault_l])
    features=sel.transform(training['data'])
    clf = LinearSVC()
    clf.fit(features,training['labels'][:,fault_l])
    y_score = clf.fit(sel.transform(training['data']),training['labels'][:,fault_l]).decision_function(sel.transform(test['data']))
    fpr_, tpr_, _ = roc_curve(test['labels'][:,fault_l],y_score,drop_intermediate=False)
    predicted_labels=clf.predict(sel.transform(test['data']))
    conf_codec_svm+=confusion_matrix(test['labels'][:,fault_l],predicted_labels)
    fpr_codec_svm.append(fpr_[0:2210])
    tpr_codec_svm.append(tpr_[0:2210])

    #Code SVM
    training['data']=loadmat(save_path+'tr'+str(i))['code']
    training['labels']=loadmat(save_path+'tr'+str(i))['labels']
    test['data']=loadmat(save_path+'test'+str(i))['code']
    test['labels']=loadmat(save_path+'test'+str(i))['labels']
    clf = LinearSVC()
    clf.fit(training['data'],training['labels'][:,fault_l])
    y_score = clf.fit(training['data'],training['labels'][:,fault_l]).decision_function(test['data'])
    fpr_, tpr_, _ = roc_curve(test['labels'][:,fault_l],y_score,drop_intermediate=False)
    predicted_labels=clf.predict(test['data'])
    conf_code_svm+=confusion_matrix(test['labels'][:,fault_l],predicted_labels)
    fpr_code_svm.append(fpr_[0:2210])
    tpr_code_svm.append(tpr_[0:2210])
    print(i)
conf_raw_rule/=5
conf_raw_svm/=5
conf_codec_rule/=5
conf_codec_svm/=5
conf_code_svm/=5

print(conf_raw_rule)
print(conf_raw_svm)
print(conf_codec_rule)
print(conf_codec_svm)
print(conf_code_svm)
plt.figure(1)
lw = 2
plt.step(np.mean(fpr_raw_rule,axis=0), np.mean(tpr_raw_rule,axis=0), color='darkorange',
         lw=lw, label='Raw Rule-ROC curve (area = %0.4f)' % np.mean(auc(np.mean(fpr_raw_rule,axis=0), np.mean(tpr_raw_rule,axis=0))))
plt.step(np.mean(fpr_raw_svm,axis=0), np.mean(tpr_raw_svm,axis=0), color='yellowgreen',
         lw=lw, label='Raw Svm-ROC curve (area = %0.4f)' % np.mean(auc(np.mean(fpr_raw_svm,axis=0), np.mean(tpr_raw_svm,axis=0))))
plt.step(np.mean(fpr_codec_rule,axis=0), np.mean(tpr_codec_rule,axis=0), color='blue',
         lw=lw, label='Codec Rule-ROC curve (area = %0.4f)' % np.mean(auc(np.mean(fpr_codec_rule,axis=0), np.mean(tpr_codec_rule,axis=0))))
plt.step(np.mean(fpr_codec_svm,axis=0), np.mean(tpr_codec_svm,axis=0), color='red',
         lw=lw, label='Codec Svm-ROC curve (area = %0.4f)' % np.mean(auc(np.mean(fpr_codec_svm,axis=0), np.mean(tpr_codec_svm,axis=0))))
plt.step(np.mean(fpr_code_svm,axis=0), np.mean(tpr_code_svm,axis=0), color='gray',
         lw=lw, label='Code Svm-ROC curve (area = %0.4f)' % np.mean(auc(np.mean(fpr_code_svm,axis=0), np.mean(tpr_code_svm,axis=0))))

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
'''
